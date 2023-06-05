from PIL import Image

from ...dynamic_menus.interface import AbstractInterface
from .icon import ChatInterfaceIcon

import wizard_eyes.client


class ChatInterface(AbstractInterface):

    PATH_TEMPLATE = '{root}/data/chat/{name}.npy'

    def __init__(self, client: "wizard_eyes.client.Client", widget):
        super().__init__(client, widget,
                         config_path='chat.interface', container_name='chat')
        self.text = None
        self.text_changed_at = -float('inf')

    @property
    def icon_class(self):
        return ChatInterfaceIcon

    def read_text(self):

        img = Image.fromarray(self.img)
        self.client.ocr.SetImage(img)
        text = str(self.client.ocr.GetUTF8Text())
        if text != self.text:
            self.text_changed_at = self.client.time

        self.text = text

    def draw(self):

        if f'chat_interface_bbox' in self.client.args.show:
            self.draw_bbox()
