from typing import Dict, List, Optional

from . import grid
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


class BankRowButton(GameObject):
    """A button in the bank's bottom row (see :data:`grid.BUTTONS`).

    State is ``<name>``, ``<name>_selected`` when it's switched on (red
    background) and either with ``_hover`` when the mouse is over it.
    Greyscale templates can't tell red from grey buttons apart, so the
    state comes from colour and the mouse position; the recorded templates
    (``data/bank/buttons/<name>[_selected][_hover].npy``) are for reference
    and identification.
    """

    PATH_TEMPLATE = '{root}/data/bank/buttons/{name}.npy'

    def __init__(self, client, parent, name: str):
        self.name = name
        self.state = ''
        super().__init__(client, parent)

    def __repr__(self):
        return f'BankRowButton<{self.name} "{self.state}">'

    @property
    def selected(self) -> bool:
        img = self.client.get_img_at(self.get_bbox(), mode=self.client.BGRA)
        if img is None or not img.size:
            return False
        band = numpy.concatenate([img[2:6].reshape(-1, 4),
                                  img[-6:-2].reshape(-1, 4)])
        b, g, r = numpy.median(band[:, :3], axis=0)
        return r > g + 40

    @property
    def hovered(self) -> bool:
        return self.is_inside(*self.client.screen.mouse_xy)

    def update(self):
        super().update()
        if not self.parent.open:
            self.state = ''
            return
        state = self.name
        if self.selected:
            state += '_selected'
        if self.hovered:
            state += '_hover'
        self.state = state


class BankTagTab(GameObject):
    """One tab in the Bank Tags plugin's tag column. Which tag it is comes
    from templates (``data/bank/tags/<tag name>.npy``, background masked
    out so selected and unselected tabs match the same template)."""

    PATH_TEMPLATE = '{root}/data/bank/tags/{name}.npy'

    def __init__(self, client, parent, index: int):
        self.index = index
        self.tag = ''
        """Name of the matching tag template, '' if none matched."""
        super().__init__(client, parent)
        self.match_threshold = 0.95

    def __repr__(self):
        sel = ' selected' if self.selected else ''
        return f'BankTagTab<{self.index} {self.tag!r}{sel}>'

    @property
    def selected(self) -> bool:
        img = self.client.get_img_at(self.get_bbox(), mode=self.client.BGRA)
        if img is None or not img.size:
            return False
        lo, hi = grid.TAG_SELECTED_BGR
        mask = cv2.inRange(img[:, :, :3], lo, hi)
        return mask.mean() > 255 * 0.15

    def update(self):
        super().update()
        if not self.parent.open:
            return
        self.tag = self.identify() if self.templates else ''


class Bank(AbstractContainer):

    PATH_TEMPLATE = '{root}/data/bank/{name}.npy'

    CLOSE_HOTKEY = 'esc'

    STATIC_TABS = ['tabINF', 'tab0', 'tab1', 'tab2', 'tab3']
    PERMUTATIONS = ['selected']

    RELOCATE_SECONDS = 1.0
    """While closed, look for the bank this often in case it moved or
    changed size (client window resized)."""

    def __init__(self, client):
        super().__init__(client, client)

        self.located: bool = False
        self.open: bool = False
        self._relocate_at: float = 0.0
        self.tag_templates: List[str] = []

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

        self.row_buttons: Dict[str, BankRowButton] = {
            name: BankRowButton(self.client, self, name)
            for name in grid.BUTTONS
        }
        self.tags: List[BankTagTab] = []
        self.buttons_top: Optional[int] = None
        """Top of the bottom button row, relative to the bank's top."""

    @property
    def interface_class(self):
        return BankInterface

    @property
    def widget_class(self):
        return BankWidget

    def button(self, name: str) -> BankRowButton:
        return self.row_buttons[name]

    def load_tag_templates(self, names: List[str]):
        """Tag templates (``data/bank/tags/<name>.npy``) to identify the
        tag tabs with, e.g. ``['potion_maker']``."""
        self.tag_templates = list(names)
        for tag in self.tags:
            self._load_tag(tag)

    def _load_tag(self, tag: BankTagTab):
        if self.tag_templates:
            tag.load_templates(self.tag_templates)
            tag.load_masks(self.tag_templates)

    @property
    def selected_tag(self) -> Optional[BankTagTab]:
        for tag in self.tags:
            if tag.selected:
                return tag
        return None

    def find_tag(self, name: str) -> Optional[BankTagTab]:
        for tag in self.tags:
            if tag.tag == name:
                return tag
        return None

    TAG_SELECT_SECONDS = 2.0

    def select_tag(self, name: str) -> Optional[bool]:
        """Make sure the named tag tab is open, one step per call (frame).

        Returns True once it's selected, None while working on it (clicked,
        waiting), and False if it can't be done: the tag isn't in the
        column, or it was clicked and didn't become selected in time (the
        caller should stop and look, not keep clicking).
        """
        tag = self.find_tag(name)
        if tag is None:
            self.logger.warning(f'Tag tab {name!r} not found in the tag '
                                f'column: {self.tags}')
            return False
        if tag.selected:
            tag.clear_timeout()
            self._tag_clicked_at = None
            return True
        clicked_at = getattr(self, '_tag_clicked_at', None)
        if clicked_at is None:
            tag.click(tmin=0.6, tmax=1.0, pause_before_click=True)
            self._tag_clicked_at = self.client.time
            self.logger.info(f'Clicked tag tab {name!r}')
            return None
        if self.client.time - clicked_at < self.TAG_SELECT_SECONDS:
            return None
        self.logger.warning(f'Clicked tag tab {name!r} but it is not '
                            f'selected after {self.TAG_SELECT_SECONDS}s')
        self._tag_clicked_at = None
        return False

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
        my, mx = numpy.where((1.01 > matches) & (matches >= .99))
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
            my2, mx2 = numpy.where((1.01 > matches2) & (matches2 >= .99))

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

    def locate_buttons(self):
        """Find the withdraw buttons by template, then place the bottom row
        buttons and the tag tabs relative to them."""
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

        bx1, by1, _, _ = self.get_bbox()
        ax1, ay1, _, _ = self.withdraw_all.get_bbox()
        for name, button in self.row_buttons.items():
            button.set_aoi(*grid.button_bbox(name, ax1, ay1))
        self.buttons_top = ay1 - by1

        self.tags = []
        for i in range(grid.tag_count(self.buttons_top)):
            x1, y1, x2, y2 = grid.tag_bbox(i)
            tag = BankTagTab(self.client, self, i)
            tag.set_aoi(bx1 + x1, by1 + y1, bx1 + x2, by1 + y2)
            self._load_tag(tag)
            self.tags.append(tag)

    def is_open(self) -> bool:
        """Top left corner or bottom right corner where we located them.
        Either can be covered: long mouse-over text runs across the top
        left corner (e.g. hovering a tag tab)."""
        if not self.located:
            return False
        if self.find('top_left') == ((0, 0),):
            return True
        _, _, x2, y2 = self.get_bbox()
        template = self.templates['bottom_right']
        h, w = template.shape
        # a couple of pixels of slack around where we located it
        img = self.client.get_img_at((x2 - w - 1, y2 - h - 1, x2 + 3, y2 + 3))
        if img is None or img.shape[0] < h or img.shape[1] < w:
            return False
        matches = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED,
                                    mask=self.masks['bottom_right'])
        return bool(((matches >= .99) & (matches < 1.01)).any())

    def reset_layout(self):
        """The bank moved or changed size: everything positioned from it
        must be found again."""
        self.logger.info(f'Bank moved / resized to {self.get_bbox()}, '
                         f're-locating its contents')
        for widget in self.widgets:
            widget.located = False
            widget.interface.located = False
            widget.interface.icons = dict()

    def update(self):

        prev_located = self.located
        prev_bbox = self.get_bbox() if self.located else None
        if not self.located:
            self.locate()
        elif not self.open and \
                self.client.time - self._relocate_at > self.RELOCATE_SECONDS:
            self._relocate_at = self.client.time
            self.locate()
        moved = (prev_bbox is not None and self.located
                 and self.get_bbox() != prev_bbox)
        if moved:
            self.reset_layout()
        if self.located and (not prev_located or moved):
            self.locate_buttons()

        self.open = self.is_open()

        for button in self.buttons:
            button.update()
        if self.open:
            for button in self.row_buttons.values():
                button.update()
            for tag in self.tags:
                tag.update()
        super().update()
