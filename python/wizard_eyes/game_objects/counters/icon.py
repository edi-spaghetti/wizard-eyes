from ...dynamic_menus.icon import AbstractIcon


class CountersInterfaceIcon(AbstractIcon):
    """
    Class to represent counters and timers in top left.
    """

    PATH_TEMPLATE = '{root}/data/game_screen/counters/{name}.npy'
