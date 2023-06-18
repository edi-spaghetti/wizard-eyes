from abc import ABC, abstractmethod
from typing import Union, Type

import cv2
import numpy

from ..game_objects.game_objects import GameObject
from .interface import AbstractInterface
from ..constants import REDA


class AbstractWidget(GameObject, ABC):

    PATH_TEMPLATE = '{root}/data/{container}/{name}.npy'
    DEFAULT_COLOUR = REDA

    METHOD_MAPPING = {}

    def __str__(self):
        return f'{self.__class__.__name__}<{self.name}>'

    def __repr__(self):
        return str(self)

    def __init__(
            self,
            name,
            client: 'wizard_eyes.client.Client',
            parent: Type['AbstractContainer'],  # noqa: 622
            *args, selected=False, **kwargs):
        super().__init__(client, parent, *args, **kwargs)
        self.name: str = name
        self.selected: bool = selected
        self._img = None
        self.located: bool = False
        self.auto_locate: bool = False  # enable as needed
        self.match_threshold: float = .99
        self.updated_at: Union[float, None] = None
        self.state: Union[str, None] = None
        self.state_changed_at: Union[float, None] = None

        args, kwargs = self.interface_init_params
        self.interface: Union[AbstractInterface, None] = self.interface_class(
            *args, **kwargs
        )

    def resolve_path(self, **kwargs):
        """Add extra keys to path template for resolving."""
        kwargs['container'] = self.parent.name
        return super().resolve_path(**kwargs)

    @property
    @abstractmethod
    def interface_class(self):
        """"""

    @property
    def interface_init_params(self):
        """Get default init params for interface."""
        return (self.client, self), dict()

    def draw(self):

        bboxes = {'*bbox', f'{self.name}_bbox'}
        if self.client.args.show.intersection(bboxes) and self.located:
            self.draw_bbox()

    def locate(self):
        """Attempt to find the widget within the client. This method can also
        be used to check if the widget is currently visible.

        :return: True if the widget was found, False otherwise
        """

        for name, template in self.templates.items():
            mask = self.masks.get(name)
            matches = cv2.matchTemplate(
                # must be client img, because we don't know where
                # the widget is yet
                self.client.img,
                template,
                self.METHOD_MAPPING.get(name, cv2.TM_CCOEFF_NORMED),
                mask=mask
            )

            # TODO: configurable threshold for widgets
            if self.METHOD_MAPPING.get(name) == cv2.TM_SQDIFF_NORMED:
                (my, mx) = numpy.where(matches <= 1 - self.match_threshold)
            else:
                (my, mx) = numpy.where(matches >= self.match_threshold)
            # assume widgets are unique and we only get one match
            for y, x in zip(my, mx):
                h, w = template.shape
                x1, y1, x2, y2 = self.client.globalise(
                    x, y, x + w - 1, y + h - 1)
                self.set_aoi(x1, y1, x2, y2)

                # TODO: this is a bit of a hack to avoid matches on areas of
                #       pure black which throw false positives.
                img = self.client.get_img_at((x1, y1, x2, y2))
                expected = numpy.sum(
                    cv2.bitwise_and(
                        template,
                        cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]
                    )
                )
                actual = numpy.sum(
                    cv2.bitwise_and(
                        img,
                        cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]
                    )
                )

                if expected * .9 < actual < expected * 1.1:
                    self.logger.debug(
                        f'{self.parent.name}.{self.name}: '
                        f'located at {x1}, {y1}, {x2}, {y2}'
                    )
                    return True
        return False

    def is_selected(self, name):
        """Check if the widget is selected. This is a default implementation
        that can be overridden by subclasses."""

        # can't be selected if the widget's not there
        if not name:
            return False

        return name.endswith('selected')

    def update_state(self):
        if self.covered_by_right_click_menu():
            return
        if not self.located:
            return

        state = self.identify()
        selected = self.is_selected(state)

        self.logger.debug(
            f'{self.name}: '
            f'chosen: {state}, '
            f'selected: {selected}, '
            f'confidence: {self.confidence:.1f}'
        )

        self.selected = selected
        if self.state != state:
            self.state_changed_at = self.client.time
        self.state = state

    def update(self):
        """
        Run standard click timeout updates, then checks to see if the widget
        has been located. Once located we can do regular state updates.
        """

        super().update()
        if self.covered_by_right_click_menu():
            return

        if not self.located:
            if self.auto_locate:
                if not self.locate():
                    return
                self.located = True
            else:
                return

        self.update_state()

        # recursively call the icons on the interface
        self.interface.update(selected=self.selected)
