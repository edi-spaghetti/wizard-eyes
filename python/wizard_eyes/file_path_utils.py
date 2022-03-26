import pickle
from os.path import dirname, join, realpath


def get_root():
    """Centralised method to define the root directory."""
    return realpath(join(dirname(__file__), '..', '..'))


def load_pickle(path):
    """Load pickled data from disk."""
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data
