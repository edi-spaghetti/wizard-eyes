from abc import ABC, abstractmethod
from typing import Union

import cv2
import numpy

from ..game_objects.game_objects import GameObject
from .interface import AbstractInterface


class AbstractWidget(GameObject, ABC):

    def as_string(self):
        return f'{self.__class__.__name__}<{self.name}>'

    def __str__(self):
        return self.as_string()

    def __repr__(self):
        return self.as_string()

    def __init__(self, name, *args, selected=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.name: str = name
        self.selected: bool = selected
        self._img = None
        self.updated_at: Union[float, None] = None
        self.state: Union[str, None] = None
        self.state_changed_at: Union[float, None] = None

        args, kwargs = self.interface_init_params
        self.interface: Union[AbstractInterface, None] = self.interface_class(
            *args, **kwargs
        )

    @property
    @abstractmethod
    def interface_class(self):
        """"""

    @property
    @abstractmethod
    def interface_init_params(self):
        """"""

    @property
    def img(self):

        # TODO: abstract duplication with abstract icon

        # same update loop, no need to create a new image
        if self.updated_at == self.client.time:
            return self._img

        img = super().img

        # draw an extra 1 pixel sized backboard so masking doesn't fail
        # (seems to be a bug if template is same size as image)
        y, x = img.shape
        img2 = numpy.zeros((y+1, x), dtype=numpy.uint8)
        img2[:y, :x] = img

        return img2

    def get_mask(self, name):
        """"""
        return self.masks.get(name)

    def draw(self):
        if f'{self.name}_bbox' in self.client.args.show and self.selected:
            self.draw_bbox()

    def update(self):
        """
        Run standard click timeout updates, then check the templates to see
        if the tab is currently selected or not.
        """

        super().update()

        # TODO: there is actually a third state where tabs are disabled (e.g.
        #  during cutscenes, on tutorial island etc.)

        cur_confidence = -float('inf')
        cur_x = cur_y = cur_h = cur_w = None
        cur_template_name = ''
        confidences = list()
        for template_name, template in self.templates.items():
            match = cv2.matchTemplate(
                self.img, template, cv2.TM_CCOEFF_NORMED,
                mask=self.get_mask(template_name)
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
        if self.state != cur_template_name:
            self.state_changed_at = self.client.time
        self.state = cur_template_name

        # recursively call the icons on the interface
        self.interface.update(selected=self.selected)
