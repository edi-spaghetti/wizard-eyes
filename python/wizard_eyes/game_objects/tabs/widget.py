from .interface import TabInterface
from ...dynamic_menus.widget import AbstractWidget


class TabWidget(AbstractWidget):

    PATH_TEMPLATE = '{root}/data/tabs/{name}.npy'

    @property
    def interface_class(self):
        return TabInterface

    @property
    def interface_init_params(self):
        return (self.client, self), dict()

    def get_mask(self, name):
        return self.masks.get('tab')
