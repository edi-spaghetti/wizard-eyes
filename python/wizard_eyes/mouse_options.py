from typing import Iterable

import cv2
import numpy

from .game_objects.readable import OCRReadable
from .constants import REDA


class MouseOptions(OCRReadable):
    """Mouse options widget class.

    Mouse options is located in the top left of the game screen and is a
    base client feature that provides information about what a left click
    will do based on the position of the mouse.

    """

    PATH_TEMPLATE = '{root}/data/mouse/letters/{name}.npy'
    SYSTEM_PATH_TEMPLATE = '{root}/data/mouse/system/{name}.npy'
    SYSTEM_TEMPLATES = ['loading', 'waiting']

    DEFAULT_COLOUR = REDA

    WHITE_LIST = (
        'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-/()'
    )
    BLACK_LIST = '!?@#$%&*<>+=:;\'"'
    NUMERIC_MODE = '0'

    def __init__(self, client, *args, **kwargs):
        super().__init__(client, client, *args, config_path='mouse_options',
                         container_name='mouse_options', **kwargs)
        self._img = None
        self.updated_at = None
        self._thread = None
        self.new_thread = None
        self.state_changed_at = None
        self.confidence = None

        self.thresh_lower = 195
        self.thresh_upper = 255

        self.system_templates = self.load_system_templates()
        self.use_ocr = True

    @staticmethod
    def parse_names(names):
        """
        Convert lower case letters to _<letter>, because windows does not
        e.g. a.npy as different to A.npy
        """

        parsed = list()
        for name in names:
            # skip spaces and commas so templates can be loaded in a little
            # more readable format
            if name in {' ', ','}:
                continue

            if name == name.upper():
                parsed.append(name)
            else:
                parsed.append(f'_{name}')

        return parsed

    def load_system_templates(self):
        """
        Load the templates that represent system messages
        e.g. Loading, or connection lost.
        """

        temp = self.PATH_TEMPLATE
        self.PATH_TEMPLATE = self.SYSTEM_PATH_TEMPLATE
        templates = super().load_templates(self.SYSTEM_TEMPLATES, cache=False)
        self.PATH_TEMPLATE = temp
        return templates

    def load_masks(self, names: Iterable = None, cache: bool = True):
        names = names or list()
        return super().load_masks(self.parse_names(names), cache=cache)

    def load_templates(self, names: Iterable = None, cache: bool = True):
        names = names or list()
        return super().load_templates(self.parse_names(names), cache=cache)

    def template_method(self, img):
        """Use template matching on the image to find individual letters."""
        found = list()
        for letter, template in self.templates.items():

            mask = self.masks.get(letter)
            matches = cv2.matchTemplate(
                img, template, cv2.TM_CCOEFF_NORMED,
                mask=mask,
            )
            (my, mx) = numpy.where(matches >= self.match_threshold)
            for _, x in zip(my, mx):
                found.append((letter.replace('_', ''), x))

        letters = sorted(found, key=lambda lx: lx[1])
        state = ''.join([lx[0] for lx in letters])

        return state

    def update(self):
        """Update the state of the mouse options widget."""

        state = None
        # first check for system messages
        for message, template in self.system_templates.items():
            match = cv2.matchTemplate(self.img, template, cv2.TM_CCOEFF_NORMED)
            _, confidence, _, _ = cv2.minMaxLoc(match)

            if confidence > self.match_threshold:
                state = message
                break

        if state:
            self.set_state(state)
            return

        if self.client.ocr is None or not self.use_ocr:
            states = []
            for img in self.process_img(self.img):
                state = self.template_method(img)
                states.append(state)
            state = ' | '.join(states)
            self.set_state(state)
        else:
            super().update()

    def draw(self):
        super().draw()

        states = {'*state', 'mo_state'}
        if self.client.args.show.intersection(states):
            x1, y1, x2, y2 = self.get_bbox()
            x1, y1, x2, y2 = self.client.localise(x1, y1, x2, y2)

            # TODO: manage this as configuration if we need to add more
            y_display_offset = 14

            cv2.putText(
                self.client.original_img, str(self.state),
                # convert relative to client image so we can draw
                (x1, y2 + y_display_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                self.colour, thickness=1
            )
