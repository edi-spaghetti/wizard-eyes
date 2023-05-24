from typing import Iterable

import cv2
import numpy

from .game_objects.game_objects import GameObject


class MouseOptions(GameObject):

    PATH_TEMPLATE = '{root}/data/mouse/letters/{name}.npy'
    SYSTEM_PATH_TEMPLATE = '{root}/data/mouse/system/{name}.npy'
    SYSTEM_TEMPLATES = ['loading', 'waiting']

    def __init__(self, client, *args, **kwargs):
        super().__init__(client, client, *args, config_path='mouse_options',
                         container_name='mouse_options', **kwargs)
        self._img = None
        self.updated_at = None
        self._state = None
        self._thread = None
        self.new_thread = None
        self.state_changed_at = None
        self.confidence = None

        self.thresh_lower = 195
        self.thresh_upper = 255

        self.system_templates = self.load_system_templates()

    @property
    def state(self):
        return str(self._state)

    def process_img(self, img):
        _, img = cv2.threshold(
            img, self.thresh_lower, self.thresh_upper, cv2.THRESH_BINARY)
        return img

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

    def update_state(self):
        """
        Try to read the mouse options with template matching.
        The mouse left click options move their spacing around, so templates
        should be individual letters, that we will use to reconstruct the word.
        Assumes the templates have already been loaded.
        """

        threshold = 0.99
        found = list()
        img = self.process_img(self.img)
        state = None

        # first check for system messages
        for message, template in self.system_templates.items():
            match = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            _, confidence, _, _ = cv2.minMaxLoc(match)

            if confidence > threshold:
                state = message
                break

        if not state:
            for letter, template in self.templates.items():

                mask = self.masks.get(letter)
                matches = cv2.matchTemplate(
                    img, template, cv2.TM_CCOEFF_NORMED,
                    mask=mask,
                )
                (my, mx) = numpy.where(matches >= threshold)
                for _, x in zip(my, mx):
                    found.append((letter.replace('_', ''), x))

            letters = sorted(found, key=lambda lx: lx[1])
            state = ''.join([lx[0] for lx in letters])

        if state != self._state:
            self.logger.debug(
                f'mouse state changed from {self._state} to {state} '
                f'at {self.client.time:.3f}')
            self._state = state
            self.state_changed_at = self.client.time

        self.updated_at = self.client.time
