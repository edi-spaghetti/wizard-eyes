from .interface import BankInterface
from ...dynamic_menus.widget import AbstractWidget

import cv2


class BankWidget(AbstractWidget):

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.is_tag = False

        # each permutation should use the same mask
        for template in self.templates:
            self.alias_mask(self.name, template)

    @property
    def interface_class(self):
        return BankInterface

    def is_selected(self, name):
        """Check if the given tab is selected."""

        if self.is_tag:

            mask = cv2.bitwise_not(self.masks[self.name])
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)  # channel match

            img = self.client.get_img_at(
                self.get_bbox(), mode=self.client.BGRA)
            img = cv2.bitwise_and(img, mask)
            # TODO: parameterise this yellow-ish colour
            img = cv2.inRange(img, (40, 75, 90, 255), (50, 85, 100, 255))
            return img.any()

        else:
            return super().is_selected(name)
