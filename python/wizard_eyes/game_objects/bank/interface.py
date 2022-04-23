from ...dynamic_menus.interface import AbstractInterface
from .icon import BankInterfaceIcon


class BankInterface(AbstractInterface):

    PATH_TEMPLATE = '{root}/data/bank/{name}.npy'

    def __init__(self, client, widget):
        super().__init__(client, widget,
                         config_path='bank.interface')

    @property
    def icon_class(self):
        return BankInterfaceIcon

    def draw(self):
        if f'bank_interface_bbox' in self.client.args.show:
            self.draw_bbox()

    def get_bbox(self):
        """
        Custom bounding box for the main bank interface. Unlike other
        interfaces, where the widget is in a container separate from the
        interface, in the bank the widgets are in a sub-container, but the
        interface is within the main container.

        Determine the interface bbox relative to the main container.
        """

        if self._bbox:
            return self._bbox

        x1, y1, x2, y2 = super().get_bbox()
        ox, oy, _, _ = self.client.localise(*self.widget.get_bbox())

        x1 = x1 + ox + self.margin_left
        y1 = y1 + oy + self.margin_top
        x2 = x1 + self.width
        y2 = y1 + self.height

        self._bbox = x1, y1, x2, y2
        return x1, y1, x2, y2
