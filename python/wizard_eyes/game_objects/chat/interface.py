from ...dynamic_menus.interface import AbstractInterface
from .icon import ChatInterfaceIcon


class ChatInterface(AbstractInterface):

    PATH_TEMPLATE = '{root}/data/chat/{name}.npy'

    def __init__(self, client, widget):
        super().__init__(client, widget,
                         config_path='chat.interface', container_name='chat')

    @property
    def icon_class(self):
        return ChatInterfaceIcon

    def draw(self):

        if f'chat_interface_bbox' in self.client.args.show:
            self.draw_bbox()
