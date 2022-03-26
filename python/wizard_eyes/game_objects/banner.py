from .game_objects import GameObject


class Banner(GameObject):
    """Represents to top banner on the client window."""

    def __init__(self, client):
        super(Banner, self).__init__(
            client, client, config_path='banner'
        )
