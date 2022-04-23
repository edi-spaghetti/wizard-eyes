from .icon import InterfaceIcon
from ...dynamic_menus.interface import AbstractInterface


class TabInterface(AbstractInterface):
    """
    The interface area that becomes available on clicking on the tabs on
    the main screen. For example, inventory, prayer etc.
    Note, these interfaces, and the icons they contain, are intended to be
    generated dynamically, rather than the rigid structure enforced by e.g.
    :class:`Inventory`.
    """

    PATH_TEMPLATE = '{root}/data/tabs/{name}.npy'

    def __init__(self, client, widget):
        super(TabInterface, self).__init__(
            client, widget, config_path='tabs.interface',
            container_name='dynamic_tabs')

    @property
    def icon_class(self):
        return InterfaceIcon

    def draw(self):
        if f'tab_interface_bbox' in self.client.args.show:
            self.draw_bbox()
