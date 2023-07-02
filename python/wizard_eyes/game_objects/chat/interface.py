import cv2
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

        if not self.located:
            return

        img = Image.fromarray(self.img)
        self.client.ocr.SetImage(img)
        text = str(self.client.ocr.GetUTF8Text())
        if text != self.text:
            self.text_changed_at = self.client.time

        self.text = text
        self.client.add_draw_call(self.draw_text)

    def draw_text(self):
        """Draws the text on the client image."""

        if not self.located:
            return

        x1, y1, x2, y2 = self.client.localise(*self.get_bbox())
        w = x2 - x1
        h = y2 - y1

        # TODO: configurable and dynamically fitting text
        font_size = .45
        thickness = 1

        triggers = {'*text', f'{self.widget.name}_text'}
        if self.client.args.show.intersection(triggers):

            lines = str(self.text).split('\n')
            for i, line in enumerate(lines):

                (tw, th), _ = cv2.getTextSize(
                    text=line,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_size,
                    thickness=thickness,
                )

                cv2.putText(
                    self.client.original_img,
                    line,
                    (int(x1 + w * 0.1), int(y1 + h * 0.1 + i * th)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_size,
                    self.colour,
                    thickness=thickness,
                )
