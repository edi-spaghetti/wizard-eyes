"""Download-on-demand asset cache backed by remote storage."""
import fnmatch
import json
import logging
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .manifest import (
    MANIFEST_KEY, IncludeRules, Manifest, blob_key, hash_bytes, hash_file,
    to_key,
)
from .storage import NotFound, Storage

logger = logging.getLogger('wizard_eyes.assets')

CACHE_DIRNAME = '.assets'


@dataclass
class PublishResult:
    new: List[str] = field(default_factory=list)
    changed: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    unchanged: int = 0
    total_files: int = 0
    total_bytes: int = 0
    uploaded_blobs: int = 0
    uploaded_bytes: int = 0
    manifest: Optional[Manifest] = None


@dataclass
class StatusResult:
    missing: List[str] = field(default_factory=list)
    modified: List[str] = field(default_factory=list)
    local_only: List[str] = field(default_factory=list)
    ok: int = 0


class AssetManager:
    """Keeps the local data directory in step with the remote manifest.

    :param data_dir: Local data root (the engine's ``data`` folder).
    :param storage: Remote backend. If None, every method is a no-op.
    :param cache_dir: Where the cached manifest and hash index live.
        Defaults to ``<data_dir>/.assets``.
    """

    def __init__(self, data_dir: str, storage: Optional[Storage],
                 cache_dir: Optional[str] = None):
        self.data_dir = os.path.abspath(data_dir)
        self.storage = storage
        self.cache_dir = cache_dir or os.path.join(
            self.data_dir, CACHE_DIRNAME)
        self._manifest: Optional[Manifest] = None
        self._refreshed = False
        self._index: Optional[Dict[str, list]] = None
        self._index_dirty = False
        self._defer_index_save = False
        self._lock = threading.RLock()

    # ---------------------------------------------------------------- cache

    @property
    def _manifest_path(self):
        return os.path.join(self.cache_dir, 'manifest.json')

    @property
    def _etag_path(self):
        return os.path.join(self.cache_dir, 'manifest.etag')

    @property
    def _index_path(self):
        return os.path.join(self.cache_dir, 'index.json')

    def _read(self, path) -> Optional[bytes]:
        try:
            with open(path, 'rb') as f:
                return f.read()
        except OSError:
            return None

    def _write(self, path, data: bytes):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f'{path}.{os.getpid()}.{threading.get_ident()}.tmp'
        with open(tmp, 'wb') as f:
            f.write(data)
        os.replace(tmp, path)

    def _load_cached_manifest(self) -> Optional[Manifest]:
        data = self._read(self._manifest_path)
        if data is None:
            return None
        try:
            return Manifest.from_bytes(data)
        except ValueError:
            logger.warning('Cached asset manifest is corrupt, ignoring it')
            return None

    # ------------------------------------------------------------- manifest

    def refresh_manifest(self, force: bool = False) -> Optional[Manifest]:
        """Fetch the remote manifest if it changed since last time.

        Only hits the network once per process unless ``force`` is set.
        Uses a conditional GET, so an unchanged manifest costs one tiny
        request. On any failure the cached copy is used.
        """
        with self._lock:
            if self.storage is None:
                return None
            if self._refreshed and not force:
                return self._manifest

            cached = self._load_cached_manifest()
            etag = None
            if cached is not None:
                etag = (self._read(self._etag_path) or b'').decode() or None

            try:
                data, new_etag = self.storage.get_bytes(
                    MANIFEST_KEY, if_none_match=etag)
            except NotFound:
                logger.warning('No remote asset manifest found')
                self._manifest = cached
            except Exception as e:  # network trouble must never stop the bot
                logger.warning(f'Asset manifest refresh failed: {e!r}; '
                               f'using cached copy')
                self._manifest = cached
            else:
                if data is None:
                    logger.debug('Asset manifest unchanged')
                    self._manifest = cached
                else:
                    self._manifest = Manifest.from_bytes(data)
                    self._write(self._manifest_path, data)
                    self._write(self._etag_path, (new_etag or '').encode())
                    logger.info(f'Asset manifest updated '
                                f'({len(self._manifest.files)} files)')

            self._refreshed = True
            return self._manifest

    @property
    def manifest(self) -> Optional[Manifest]:
        return self.refresh_manifest()

    # ---------------------------------------------------------------- index

    def _get_index(self) -> Dict[str, list]:
        if self._index is None:
            data = self._read(self._index_path)
            try:
                self._index = json.loads(data) if data else {}
            except ValueError:
                self._index = {}
        return self._index

    def _save_index(self, force: bool = False):
        if self._defer_index_save and not force:
            return
        if self._index is not None and self._index_dirty:
            self._write(self._index_path, json.dumps(self._index).encode())
            self._index_dirty = False

    @contextmanager
    def _bulk(self):
        """Save the hash index once at the end instead of per file."""
        self._defer_index_save = True
        try:
            yield
        finally:
            self._defer_index_save = False
            self._save_index()

    def local_sha(self, key: str, update_index: bool = True) -> Optional[str]:
        """sha256 of the local file, using a (size, mtime) cache."""
        path = self.local_path(key)
        try:
            st = os.stat(path)
        except OSError:
            return None
        index = self._get_index()
        cached = index.get(key)
        if cached and cached[0] == st.st_size and cached[1] == st.st_mtime_ns:
            return cached[2]
        sha = hash_file(path)
        if update_index:
            index[key] = [st.st_size, st.st_mtime_ns, sha]
            self._index_dirty = True
            self._save_index()
        return sha

    def local_path(self, key: str) -> str:
        return os.path.join(self.data_dir, *key.split('/'))

    # --------------------------------------------------------------- ensure

    def ensure(self, path: str) -> str:
        """Make sure the file at ``path`` matches the manifest.

        Downloads it if it's missing or out of date. Paths outside the data
        directory, or not in the manifest, are left alone. Never raises:
        on failure whatever is on disk is used.

        :returns: The path that was passed in.
        """
        if self.storage is None:
            return path
        try:
            key = to_key(path, self.data_dir)
            if key is not None:
                self._ensure_key(key)
        except Exception as e:
            logger.warning(f'Could not ensure asset {path}: {e!r}')
        return path

    def ensure_glob(self, pattern: str) -> List[str]:
        """Ensure every manifest entry matching a glob pattern path."""
        if self.storage is None:
            return []
        try:
            key_pattern = to_key(pattern, self.data_dir)
            manifest = self.manifest
            if key_pattern is None or manifest is None:
                return []
            keys = [k for k in manifest.files
                    if fnmatch.fnmatchcase(k, key_pattern)]
            for key in keys:
                self._ensure_key(key)
            return keys
        except Exception as e:
            logger.warning(f'Could not ensure assets {pattern}: {e!r}')
            return []

    def _ensure_key(self, key: str) -> bool:
        """:returns: True if the file was downloaded."""
        with self._lock:
            manifest = self.manifest
            if manifest is None:
                return False
            expected = manifest.sha(key)
            if expected is None:
                return False
            if self.local_sha(key) == expected:
                return False

            data, _ = self.storage.get_bytes(blob_key(expected))
            actual = hash_bytes(data)
            if actual != expected:
                raise ValueError(
                    f'hash mismatch for {key}: expected {expected}, '
                    f'got {actual}')
            path = self.local_path(key)
            self._write(path, data)
            st = os.stat(path)
            self._get_index()[key] = [st.st_size, st.st_mtime_ns, expected]
            self._index_dirty = True
            self._save_index()
            logger.info(f'Downloaded asset {key} ({len(data)} bytes)')
            return True

    # ----------------------------------------------------------- bulk tools

    def sync(self, prefix: str = '') -> Tuple[int, int]:
        """Download everything in the manifest under ``prefix``.

        :returns: (downloaded, failed) counts.
        """
        manifest = self.refresh_manifest(force=True)
        if manifest is None:
            return 0, 0
        downloaded = failed = 0
        with self._bulk():
            for key in sorted(manifest.files):
                if not key.startswith(prefix):
                    continue
                try:
                    if self._ensure_key(key):
                        downloaded += 1
                except Exception as e:
                    failed += 1
                    logger.warning(f'Failed to sync {key}: {e!r}')
        return downloaded, failed

    def status(self, rules: Optional[IncludeRules] = None) -> StatusResult:
        """Compare local files against the manifest."""
        result = StatusResult()
        manifest = self.refresh_manifest(force=True) or Manifest()
        with self._bulk():
            for key, entry in sorted(manifest.files.items()):
                sha = self.local_sha(key)
                if sha is None:
                    result.missing.append(key)
                elif sha != entry['sha256']:
                    result.modified.append(key)
                else:
                    result.ok += 1
        if rules is not None:
            result.local_only = [
                k for k in rules.scan(self.data_dir)
                if k not in manifest.files]
        return result

    def publish(self, rules: IncludeRules,
                dry_run: bool = False) -> PublishResult:
        """Hash local files, upload new blobs and a new manifest.

        Blobs are never deleted, so old manifests stay restorable.
        In dry-run mode nothing is written locally or remotely.
        """
        result = PublishResult()
        if self.storage is None and not dry_run:
            raise ValueError('No asset storage configured')

        previous = Manifest()
        if self.storage is not None:
            try:
                data, _ = self.storage.get_bytes(MANIFEST_KEY)
                previous = Manifest.from_bytes(data)
            except NotFound:
                pass
        known_blobs = {e['sha256'] for e in previous.files.values()}

        manifest = Manifest(generated_at=Manifest.now())
        with self._bulk():
            for key in rules.scan(self.data_dir):
                path = self.local_path(key)
                size = os.path.getsize(path)
                sha = self.local_sha(key, update_index=not dry_run)
                manifest.files[key] = {'sha256': sha, 'size': size}
                result.total_files += 1
                result.total_bytes += size

                old = previous.sha(key)
                if old is None:
                    result.new.append(key)
                elif old != sha:
                    result.changed.append(key)
                else:
                    result.unchanged += 1

                if sha in known_blobs:
                    continue
                known_blobs.add(sha)
                if not dry_run:
                    if self.storage.exists(blob_key(sha)):
                        continue
                    with open(path, 'rb') as f:
                        self.storage.put_bytes(blob_key(sha), f.read())
                result.uploaded_blobs += 1
                result.uploaded_bytes += size

        result.removed = sorted(set(previous.files) - set(manifest.files))
        result.manifest = manifest

        if not dry_run:
            self.storage.put_bytes(MANIFEST_KEY, manifest.to_bytes())
            # local files now match the manifest by definition
            self._write(self._manifest_path, manifest.to_bytes())
            self._manifest = manifest
            self._refreshed = False  # pick up the new etag next time
        return result
