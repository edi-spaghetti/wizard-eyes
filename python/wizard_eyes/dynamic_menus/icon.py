from abc import ABC

import cv2
import numpy

from ..game_objects.game_objects import GameObject


class AbstractIcon(GameObject, ABC):
    """
    Class to represent icons/buttons/items etc. dynamically generated in
    an instance of :class:`TabInterface`.
    """

    DETECT_ANYTHING = False

    PATH_TEMPLATE = '{root}/data/{container}/{widget}/{name}.npy'

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f'{self.__class__.__name__}<{self.name} {self.state}>'

    def __init__(self, name, *args, type_=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.previous_state = None
        self.state = None
        self.state_changed_at = None
        self.updated_at = None
        self.type = type_
        self._img = None

    def resolve_path(self, **kwargs):
        """Add extra keys to path template for resolving."""
        kwargs['container'] = self.parent.widget.parent.name
        kwargs['widget'] = self.parent.widget.name
        return super().resolve_path(**kwargs)

    def draw(self):
        bboxes = {'*bbox', f'{self.type}_bbox', f'{self.name}_bbox'}
        if self.client.args.show.intersection(bboxes):
            self.draw_bbox()

        states = {'*state', f'{self.type}_state', f'{self.name}_state'}
        if self.client.args.show.intersection(states):
            cx1, cy1, _, _ = self.client.get_bbox()
            x1, y1, x2, y2 = self.get_bbox()
            if self.client.is_inside(x1, y1) and self.client.is_inside(x2, y2):
                # convert local to client image
                x1, y1, x2, y2 = self.client.localise(x1, y1, x2, y2, draw=True)

                # draw the state just under the bbox
                cv2.putText(
                    self.client.original_img, str(self.state), (x1, y2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, self.colour, thickness=1
                )

    def update(self):
        """
        Run the standard click timer updates, then run template matching to
        determine the current state of the icon. Usually it will have a
        different appearance if activated/clicked.
        """
        super().update()
        if self.covered_by_right_click_menu():
            return

        cur_state = self.identify()

        if self.DETECT_ANYTHING and not cur_state:
            canny = cv2.Canny(self.img, threshold1=100, threshold2=200)
            if canny.any():
                cur_state = 'something'

        cur_state = cur_state or 'nothing'
        if cur_state.endswith('placeholder'):
            # invert colour for placeholders
            self.colour = tuple(int(c) for c in cv2.bitwise_not(
                numpy.uint8([[self.DEFAULT_COLOUR]])).reshape(4,))
        else:
            self.colour = self.DEFAULT_COLOUR

        if cur_state != self.state:
            self.logger.debug(
                f'{self.name} state changed from {self.state} to {cur_state} '
                f'at {self.client.time:.3f}')
            self.previous_state = self.state
            self.state = cur_state
            self.state_changed_at = self.client.time

