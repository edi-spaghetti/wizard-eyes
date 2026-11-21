"""Asset manifest: maps data-relative paths to content hashes.

A manifest looks like::

    {
        "version": 1,
        "generated_at": "2026-11-21T12:00:00+00:00",
        "files": {"tabs/inventory/frame.npy": {"sha256": "...", "size": 123}}
    }

Blobs are stored remotely by content hash, see :func:`blob_key`.
"""
import fnmatch
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional

MANIFEST_KEY = 'manifest.json'
MANIFEST_VERSION = 1


def blob_key(sha256: str) -> str:
    """Remote key for a blob with the given content hash."""
    return f'blobs/{sha256[:2]}/{sha256}'


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def to_key(path: str, data_dir: str) -> Optional[str]:
    """Convert an absolute (or data-relative) path to a manifest key.

    Returns None if the path is outside the data directory.
    """
    if not os.path.isabs(path):
        path = os.path.join(data_dir, path)
    rel = os.path.relpath(os.path.abspath(path), os.path.abspath(data_dir))
    if rel == '.' or rel.startswith('..') or os.path.isabs(rel):
        return None
    return rel.replace(os.sep, '/')


@dataclass
class Manifest:
    files: Dict[str, Dict] = field(default_factory=dict)
    generated_at: str = ''
    version: int = MANIFEST_VERSION

    @classmethod
    def from_bytes(cls, data: bytes) -> 'Manifest':
        raw = json.loads(data.decode('utf-8'))
        return cls(
            files=raw.get('files', {}),
            generated_at=raw.get('generated_at', ''),
            version=raw.get('version', MANIFEST_VERSION),
        )

    def to_bytes(self) -> bytes:
        return json.dumps({
            'version': self.version,
            'generated_at': self.generated_at,
            'files': dict(sorted(self.files.items())),
        }, indent=1, ensure_ascii=False).encode('utf-8')

    def sha(self, key: str) -> Optional[str]:
        entry = self.files.get(key)
        return entry['sha256'] if entry else None

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec='seconds')


@dataclass
class IncludeRules:
    """Decides which files in the data directory are published."""
    include: List[str] = field(default_factory=list)
    exclude_segments: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)

    def __post_init__(self):
        self._regexes = [re.compile(p) for p in self.exclude_patterns]
        self._segments = set(self.exclude_segments)

    @classmethod
    def load(cls, path: str) -> 'IncludeRules':
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        return cls(
            include=raw.get('include', []),
            exclude_segments=raw.get('exclude_segments', []),
            exclude_patterns=raw.get('exclude_patterns', []),
        )

    def matches(self, key: str) -> bool:
        parts = key.split('/')
        if self._segments.intersection(parts[:-1]):
            return False
        if not any(fnmatch.fnmatchcase(parts[-1], p) for p in self.include):
            return False
        return not any(r.search(key) for r in self._regexes)

    def scan(self, data_dir: str) -> Iterator[str]:
        """Yield manifest keys of all matching files under data_dir."""
        for dirpath, dirnames, filenames in os.walk(data_dir):
            dirnames[:] = sorted(
                d for d in dirnames if d not in self._segments)
            for name in sorted(filenames):
                key = to_key(os.path.join(dirpath, name), data_dir)
                if key and self.matches(key):
                    yield key
