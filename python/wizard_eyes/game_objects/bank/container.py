from typing import Optional

from .interface import BankInterface
from .widget import BankWidget
from ...dynamic_menus.container import AbstractContainer
from ..game_objects import GameObject

import cv2
import numpy


class BankButton(GameObject):
    """Object with state."""

    PATH_TEMPLATE = '{root}/data/bank/{name}.npy'

    def __str__(self):
        return f'{self.__class__.__name__}<{self.name} "{self.state}">'

    def __repr__(self):
        return str(self)

    def __init__(self, *args, name: str = "", **kwargs):
        self.parent: Optional[Bank] = None
        self.state: str = ''
        self.name: str = name
        super().__init__(*args, **kwargs)

    def update(self):
        super().update()

        if not self.parent.located or not self.parent.open:
            self.state = ''
            return

        state = self.identify()
        self.state = state


class Bank(AbstractContainer):

    PATH_TEMPLATE = '{root}/data/bank/{name}.npy'

    CLOSE_HOTKEY = 'esc'

    STATIC_TABS = ['tabINF', 'tab0', 'tab1', 'tab2', 'tab3']
    PERMUTATIONS = ['selected']

    def __init__(self, client):
        super().__init__(client, client)

        self.located: bool = False
        self.open: bool = False

        # for main bank interface bbox
        self.load_templates(
            [
                'top_left',
                'bottom_right',
                'withdraw_x',
                'withdraw_x_selected',
                'withdraw_all',
                'withdraw_all_selected',
            ]
        )
        self.load_masks(
            [
                'top_left',
                'bottom_right',
                'withdraw_x',
                'withdraw_x_selected',
                'withdraw_all',
                'withdraw_all_selected',
            ]
        )

        # this is the 'all items' tab we can assume is always present.
        self.tabINF = self.create_widget('tabINF')
        # bank tabs are fairly arbitrary, as they depend on the templates we
        # provide them. Can't really see a situation where we'd need more than
        # 4 tabs though.
        self.tab0 = self.create_widget('tab0')
        self.tab1 = self.create_widget('tab1')
        self.tab2 = self.create_widget('tab2')
        self.tab3 = self.create_widget('tab3')

        self.withdraw_x = BankButton(
            self.client,
            self,
            name='withdraw_x',
            template_names=['withdraw_x', 'withdraw_x_selected']
        )
        self.withdraw_all = BankButton(
            self.client,
            self,
            name='withdraw_all',
            template_names=['withdraw_all', 'withdraw_all_selected']
        )
        self.buttons = [self.withdraw_x, self.withdraw_all]

    @property
    def interface_class(self):
        return BankInterface

    @property
    def widget_class(self):
        return BankWidget

    def close(self):

        self.client.screen.press_key(self.CLOSE_HOTKEY)
        offset = self.client.screen.map_between(start=0.6, stop=1.2)
        self.add_timeout(offset)

        return self.CLOSE_HOTKEY

    def locate(self):

        # find the top left
        x1, y1, x2, y2 = self.client.get_bbox()
        img = self.client.get_img_at((x1, y1, x2, y2))
        # img = cv2.bitwise_not(img)

        matches = cv2.matchTemplate(
            img,
            self.templates['top_left'],
            cv2.TM_CCOEFF_NORMED,
            mask=self.masks['top_left'],
        )
        my, mx = numpy.where(matches >= .99)
        if len(my) > 1:
            self.logger.warning('More than one TL bounding box detected')

        for x01, y01 in zip(mx, my):

            # find bottom right relative to top left
            a = x1 + x01
            b = y1 + y01
            img2 = self.client.get_img_at(
                (a, b, x2, y2),
                mode=self.client.GRAY
            )

            matches2 = cv2.matchTemplate(
                img2,
                self.templates['bottom_right'],
                cv2.TM_CCOEFF_NORMED,
                mask=self.masks['bottom_right'],
            )
            my2, mx2 = numpy.where(matches2 >= .99)

            if len(my2) > 1:
                self.logger.warning('More than one BR bounding box detected')

            for x02, y02 in zip(mx2, my2):

                x1_final = a
                y1_final = b
                x2_final = a + x02 + self.templates['bottom_right'].shape[1] - 1
                y2_final = b + y02 + self.templates['bottom_right'].shape[0] - 1

                bbox = (x1_final, y1_final, x2_final, y2_final)

                self.set_aoi(*bbox)
                self.located = True
                break

    def update(self):

        prev_located = self.located
        if not self.located:
            self.locate()
        if self.located and not prev_located:
            for button in self.buttons:
                b_location = self.find(button.name)
                if not b_location:
                    b_location = self.find(f'{button.name}_selected')
                if b_location:
                    x1, y1 = b_location[0]
                    h, w = button.templates[button.name].shape
                    x2 = x1 + w - 1
                    y2 = y1 + h - 1
                    global_bbox = self.globalise(x1, y1, x2, y2)
                    button.set_aoi(*global_bbox)
                else:
                    raise RuntimeError(
                        f'Could not find button {button.name} '
                        f'in bank interface'
                    )

        self.open = self.find('top_left') == ((0, 0),)

        for button in self.buttons:
            button.update()
        super().update()
