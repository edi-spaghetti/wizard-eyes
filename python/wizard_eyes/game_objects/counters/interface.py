from ...dynamic_menus.interface import AbstractInterface
from .icon import CountersInterfaceIcon


class CountersInterface(AbstractInterface):

    PATH_TEMPLATE = '{root}/data/game_screen/counters/{name}.npy'

    def __init__(self, client, widget):
        super().__init__(
            client, widget, config_path='mouse_options.counters.interface'
        )

    @property
    def icon_class(self):
        return CountersInterfaceIcon
