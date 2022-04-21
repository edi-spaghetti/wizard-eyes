from .interface import ChatInterface
from ...dynamic_menus.widget import AbstractWidget


class ChatWidget(AbstractWidget):

    PATH_TEMPLATE = '{root}/data/chat/{name}.npy'

    @property
    def interface_class(self):
        return ChatInterface

    @property
    def interface_init_params(self):
        return (self.client, self), dict()
