import pickle
from os.path import dirname, join, realpath


def get_root():
    """Centralised method to define the root directory."""
    return realpath(join(dirname(__file__), '..', '..'))


def ensure_local(path):
    """Download an asset from remote storage if missing or out of date.

    No-op when remote assets aren't configured. Returns the path.
    """
    from .assets import ensure_local as _ensure_local
    return _ensure_local(path)


def ensure_local_glob(pattern):
    """Ensure all remote assets matching a glob pattern are on disk."""
    from .assets import ensure_local_glob as _ensure_local_glob
    return _ensure_local_glob(pattern)


def load_pickle(path):
    """Load pickled data from disk."""
    ensure_local(path)
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data
