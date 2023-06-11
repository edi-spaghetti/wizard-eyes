from PIL import Image

from ...dynamic_menus.interface import AbstractInterface
from .icon import ChatInterfaceIcon

import wizard_eyes.client


class ChatInterface(AbstractInterface):

    def __init__(self, client: 'wizard_eyes.client.Client', widget):
        super().__init__(client, widget)
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
