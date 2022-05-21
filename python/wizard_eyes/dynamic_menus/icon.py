from abc import ABC

import cv2
import numpy

from ..game_objects.game_objects import GameObject


class AbstractIcon(GameObject, ABC):
    """
    Class to represent icons/buttons/items etc. dynamically generated in
    an instance of :class:`TabInterface`.
    """

    def as_string(self):
        return f'{self.__class__.__name__}<{self.name} {self.state}>'

    def __repr__(self):
        return self.as_string()

    def __str__(self):
        return self.as_string()

    def __init__(self, name, *args, threshold=None, type_=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.confidence = None
        self.state = None
        self.state_changed_at = None
        self.updated_at = None
        self.threshold = threshold
        self.type = type_
        self._img = None

    @property
    def img(self):

        # same update loop, no need to create a new image
        if self.updated_at == self.client.time:
            return self._img

        img = super().img

        # draw an extra 1 pixel sized backboard so masking doesn't fail
        # (seems to be a bug if template is same size as image)
        y, x = img.shape
        img2 = numpy.zeros((y+1, x), dtype=numpy.uint8)
        img2[:y, :x] = img

        self._img = img2
        return img2

    def draw(self):
        if f'{self.type}_bbox' in self.client.args.show:
            self.draw_bbox()

        if f'{self.type}_state' in self.client.args.show:
            cx1, cy1, _, _ = self.client.get_bbox()
            x1, y1, x2, y2 = self.get_bbox()
            if self.client.is_inside(x1, y1) and self.client.is_inside(x2, y2):
                # convert local to client image
                x1, y1, x2, y2 = self.client.localise(x1, y1, x2, y2)

                # draw the state just under the bbox
                cv2.putText(
                    self.client.original_img, str(self.state), (x1, y2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, self.colour, thickness=1
                )

    def update(self, threshold=0.99):
        """
        Run the standard click timer updates, then run template matching to
        determine the current state of the icon. Usually it will have a
        different appearance if activated/clicked.
        """
        super().update()

        cur_confidence = -float('inf')
        cur_state = None
        for state, template in self.templates.items():
            mask = self.masks.get(state)

            try:
                match = cv2.matchTemplate(
                    self.img, template, cv2.TM_CCOEFF_NORMED,
                    mask=mask,
                )
            except cv2.error:
                # if template sizes don't match the icon image size then it's
                # definitely not the right template
                continue

            _, confidence, _, _ = cv2.minMaxLoc(match)

            if confidence > cur_confidence and confidence > threshold:
                cur_state = state
                cur_confidence = confidence

        if cur_state != self.state:
            self.logger.debug(
                f'{self.name} state changed from {self.state} to {cur_state} '
                f'at {self.client.time:.3f}')
            self.state = cur_state
            self.state_changed_at = self.client.time

        self.confidence = cur_confidence
        self.updated_at = self.client.time
