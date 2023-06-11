import cv2

from .interface import ChatInterface
from ...dynamic_menus.widget import AbstractWidget


class ChatWidget(AbstractWidget):

    METHOD_MAPPING = {
        'all_selected': cv2.TM_SQDIFF_NORMED,
    }

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

        # report button has a different shape to other chat widgets,
        # so it needs its own special mask
        if name == 'report':
            alias = 'report'
        else:
            alias = 'chat'
        self.load_masks([alias])
        for template in self.templates:
            self.alias_mask(alias, template)

    @property
    def interface_class(self):
        return ChatInterface
