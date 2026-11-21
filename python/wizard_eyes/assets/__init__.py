"""Remote asset storage for the data directory.

The engine calls :func:`ensure_local` before loading any file from ``data/``.
If remote assets are configured (see :mod:`wizard_eyes.assets.config`) the
file is downloaded when missing or out of date; otherwise it's a no-op.
"""
import logging
import threading
from typing import List, Optional

from .client import AssetManager
from .config import AssetConfig, default_data_dir
from .manifest import IncludeRules, Manifest
from .storage import LocalStorage, NotFound, R2Storage, Storage

logger = logging.getLogger('wizard_eyes.assets')

_manager: Optional[AssetManager] = None
_manager_lock = threading.Lock()


def get_manager() -> AssetManager:
    """The process-wide asset manager, created from config on first use."""
    global _manager
    with _manager_lock:
        if _manager is None:
            try:
                storage = AssetConfig.load().create_storage()
            except Exception as e:
                logger.warning(f'Remote assets disabled: {e!r}')
                storage = None
            _manager = AssetManager(default_data_dir(), storage)
        return _manager


def set_manager(manager: Optional[AssetManager]):
    """Replace the process-wide manager (None resets to config default)."""
    global _manager
    with _manager_lock:
        _manager = manager


def ensure_local(path: str) -> str:
    """Make sure an asset is present and current on disk.

    Safe to call on any path; returns the path unchanged.
    """
    return get_manager().ensure(str(path))


def ensure_local_glob(pattern: str) -> List[str]:
    """Like :func:`ensure_local` for every manifest file matching a glob."""
    return get_manager().ensure_glob(str(pattern))
