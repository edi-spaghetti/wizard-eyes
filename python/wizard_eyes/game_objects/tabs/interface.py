from .icon import InterfaceIcon
from ...dynamic_menus.interface import AbstractInterface

import numpy


class TabInterface(AbstractInterface):
    """
    The interface area that becomes available on clicking on the tabs on
    the main screen. For example, inventory, prayer etc.
    Note, these interfaces, and the icons they contain, are intended to be
    generated dynamically, rather than the rigid structure enforced by e.g.
    :class:`Inventory`.
    """

    ALPHA_MAPPING = {
        'equipment': {
            (1, 1, 255, 255): ['helmet', 1],
            (0, 255, 255, 255): ['cape', 1],
            (255, 0, 255, 255): ['amulet', 1],
            (255, 255, 0, 255): ['ammo', 1],
            (100, 100, 255, 255): ['weapon', 1],
            (0, 255, 0, 255): ['body', 1],
            (255, 100, 100, 255): ['shield', 1],
            (255, 0, 0, 255): ['legs', 1],
            (100, 100, 100, 255): ['gloves', 1],
            (0, 100, 255, 255): ['boots', 1],
            (255, 255, 255, 255): ['ring', 1],
        },
        'inventory': {
            (0, 0, 255, 255): ['item', 28],
            (0, 0, 254, 255): ['item', 28],
            (0, 0, 253, 255): ['item', 28],
            (0, 0, 252, 255): ['item', 28],
            (0, 0, 251, 255): ['item', 28],
            (0, 0, 250, 255): ['item', 28],
            (0, 0, 249, 255): ['item', 28],
            (0, 0, 248, 255): ['item', 28],
            (0, 0, 247, 255): ['item', 28],
            (0, 0, 246, 255): ['item', 28],
            (0, 0, 245, 255): ['item', 28],
            (0, 0, 244, 255): ['item', 28],
            (0, 0, 243, 255): ['item', 28],
            (0, 0, 242, 255): ['item', 28],
            (0, 0, 241, 255): ['item', 28],
            (0, 0, 240, 255): ['item', 28],
            (0, 0, 239, 255): ['item', 28],
            (0, 0, 238, 255): ['item', 28],
            (0, 0, 237, 255): ['item', 28],
            (0, 0, 236, 255): ['item', 28],
            (0, 0, 235, 255): ['item', 28],
            (0, 0, 234, 255): ['item', 28],
            (0, 0, 233, 255): ['item', 28],
            (0, 0, 232, 255): ['item', 28],
            (0, 0, 231, 255): ['item', 28],
            (0, 0, 230, 255): ['item', 28],
            (0, 0, 229, 255): ['item', 28],
            (0, 0, 228, 255): ['item', 28],
        },
        'prayer': {
            (193, 191, 34, 255): ['protect_magic', 1],
            (193, 191, 33, 255): ['protect_missiles', 1],
            (193, 191, 32, 255): ['protect_melee', 1],
            (193, 191, 31, 255): ['piety', 1],
            (193, 191, 30, 255): ['rigour', 1],
            (193, 191, 29, 255): ['augury', 1],
        }
        # TODO: other tabs
    }

    def __init__(self, client, widget):
        super().__init__(client, widget)
        # TODO: add common icons

    @property
    def icon_class(self):
        return InterfaceIcon

    def sort_unique_alpha(self, unique):
        """All inventory icons are in one channel, with the highest red value
        in top left, to lowest in bottom right."""
        return sorted(unique, key=lambda x: numpy.sum(x), reverse=True)
