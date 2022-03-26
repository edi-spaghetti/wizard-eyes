import cv2
import numpy

from .interface import TabInterface
from ..game_objects import GameObject


class TabItem(GameObject):

    PATH_TEMPLATE = '{root}/data/tabs/{name}.npy'

    def __str__(self):
        return f'TabItem<{self.name}>'

    def __repr__(self):
        return f'TabItem<{self.name}>'

    def __init__(self, name, *args, selected=False, **kwargs):
        super(TabItem, self).__init__(*args, **kwargs)
        self.name = name
        self.selected = selected
        self.interface = TabInterface(self.client, self)

    @property
    def img(self):
        img = super(TabItem, self).img

        # draw an extra 1 pixel sized backboard so masking doesn't fail
        # (seems to be a bug if template is same size as image)
        y, x = img.shape
        img2 = numpy.zeros((y+1, x), dtype=numpy.uint8)
        img2[:y, :x] = img

        return img2

    def update(self):
        """
        Run standard click timeout updates, then check the templates to see
        if the tab is currently selected or not.
        """

        super(TabItem, self).update()

        # TODO: there is actually a third state where tabs are disabled (e.g.
        #  during cutscenes, on tutorial island etc.)

        cur_confidence = -float('inf')
        cur_x = cur_y = cur_h = cur_w = None
        cur_template_name = ''
        confidences = list()
        for template_name, template in self.templates.items():
            match = cv2.matchTemplate(
                self.img, template, cv2.TM_CCOEFF_NORMED,
                mask=self.masks.get('tab'),
            )
            _, confidence, _, (x, y) = cv2.minMaxLoc(match)

            # log confidence for later
            confidences.append(f'{template_name}: {confidence:.3f}')

            if confidence > cur_confidence:
                cur_confidence = confidence
                cur_x = x
                cur_y = y
                cur_h, cur_w = template.shape
                cur_template_name = template_name

        selected = cur_template_name.endswith('selected')

        self.logger.debug(
            f'{self.name}: '
            f'chosen: {cur_template_name}, '
            f'selected: {selected}, '
            f'confidence: {confidences}'
        )

        self.selected = selected

        # TODO: convert to base class method
        if f'{self.name}_bbox' in self.client.args.show and selected:
            cx1, cy1, _, _ = self.client.get_bbox()
            x1, y1, x2, y2 = self.get_bbox()
            if self.client.is_inside(x1, y1) and self.client.is_inside(x2, y2):
                # convert local to client image
                x1, y1, x2, y2 = self.client.localise(x1, y1, x2, y2)

                # draw a rect around entity on main screen
                cv2.rectangle(
                    self.client.original_img, (x1, y1), (x2, y2),
                    self.colour, 1)

        # recursively call the icons on the interface
        self.interface.update()
