from unittest import mock

from ...dynamic_menus.container import AbstractContainer
from .interface import CountersInterface


class Counters(AbstractContainer):

    PATH_TEMPLATE = '{root}/data/game_screen/counters/{name}.npy'

    def __init__(self, client):

        super().__init__(
            client, client, config_path='mouse_options.counters',
            container_name='mouse_options',
        )

    @property
    def widget_class(self):
        return mock.MagicMock()

    @property
    def interface_class(self):
        return CountersInterface

    @property
    def interface_init_params(self):
        return (self.client, self.client), dict()
