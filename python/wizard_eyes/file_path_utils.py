from os.path import dirname, join, realpath


def get_root():
    """Centralised method to define the root directory."""
    return realpath(join(dirname(__file__), '..', '..'))
