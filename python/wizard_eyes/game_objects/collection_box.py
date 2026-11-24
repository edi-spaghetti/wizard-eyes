"""The Grand Exchange collection box (bank booth / banker "Collect").

Found by its title bar (``data/collection_box/title.npy``); everything else
is placed relative to the box's top left corner. The box has 8 offer slots
(2 rows of 4), each with a progress bar and two collect boxes, and two
buttons to collect everything to the inventory or to the bank.

Only looked for while a bot expects it (``update`` does a full client
template match until it's found).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy

from .bank.quantity import read_quantity
from .game_objects import GameObject

BBox = Tuple[int, int, int, int]

SLOT_ORIGIN = (12, 44)
SLOT_PITCH = (110, 84)
SLOT_SIZE = (100, 74)
SLOT_COLUMNS = 4
SLOTS = 8
BAR = (8, 8, 47, 25)
"""Progress bar, relative to a slot."""
ITEMS = ((8, 30, 47, 65), (53, 30, 91, 65))
"""The two collect boxes, relative to a slot."""
OFFER_ICON = (72, 8, 91, 25)
"""Small icon of the offer's item, relative to a slot (next to the bar)."""
BAR_ROW = (9, 17, 46, 17)
"""A row of the progress bar, inside its frame."""
QUANTITY = (-2, -2, 39, 14)
"""Where a collect box's quantity can be, relative to the box's corner."""
INVENTORY_BUTTON = (138, 213, 221, 231)
BANK_BUTTON = (234, 213, 316, 231)
CLOSE_BUTTON = (428, 7, 451, 28)

ICON_STD = 5
"""The same for the offer's small icon (none is about 0.4, the darkest herb
11)."""
ITEM_STD = 20
"""A collect box whose pixels vary more than this holds an item (empty
ones are about 5, items 40+)."""


def _offset(origin: Tuple[int, int], box: BBox) -> BBox:
    x, y = origin
    return box[0] + x, box[1] + y, box[2] + x, box[3] + y


@dataclass
class OfferSlot:
    index: int
    status: str
    """'empty' (no offer), 'complete' (green), 'cancelled' (red) or
    'active' (orange, in progress)."""
    items: Tuple[bool, bool]
    """Whether each collect box holds something."""
    progress: float = 0.0
    """Share of the bar that's filled (about 3% steps)."""
    quantities: Tuple[Optional[int], Optional[int]] = (None, None)
    """What each collect box says it holds (None: no number)."""

    @property
    def collectable(self) -> bool:
        return any(self.items)


class CollectionBox(GameObject):

    PATH_TEMPLATE = '{root}/data/collection_box/{name}.npy'

    def __init__(self, client):
        super().__init__(client, client)
        self.load_templates(['title'])
        self.located = False
        self.open = False
        self.match_threshold = 0.99
        self.slots: List[OfferSlot] = []

    def __repr__(self):
        return (f'CollectionBox<open={self.open} '
                f'{[s.status for s in self.slots]}>')

    # geometry (global coordinates, from the located box)

    def _global(self, box: BBox) -> BBox:
        x1, y1, _, _ = self.get_bbox()
        return _offset((x1, y1), box)

    def slot_bbox(self, index: int) -> BBox:
        row, col = divmod(index, SLOT_COLUMNS)
        x = SLOT_ORIGIN[0] + col * SLOT_PITCH[0]
        y = SLOT_ORIGIN[1] + row * SLOT_PITCH[1]
        return self._global((x, y, x + SLOT_SIZE[0] - 1,
                             y + SLOT_SIZE[1] - 1))

    def _in_slot(self, index: int, box: BBox) -> BBox:
        sx, sy, _, _ = self.slot_bbox(index)
        return _offset((sx, sy), box)

    @property
    def bank_button(self) -> BBox:
        return self._global(BANK_BUTTON)

    @property
    def inventory_button(self) -> BBox:
        return self._global(INVENTORY_BUTTON)

    @property
    def close_button(self) -> BBox:
        return self._global(CLOSE_BUTTON)

    # detection

    def locate(self) -> bool:
        template = self.templates['title']
        img = self.client.img
        matches = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        ys, xs = numpy.where((matches >= self.match_threshold)
                             & (matches < 1.01))
        if not len(xs):
            return False
        h, w = template.shape
        cx1, cy1, _, _ = self.client.get_bbox()
        x1, y1 = int(xs[0]) + cx1, int(ys[0]) + cy1
        # the box is as wide as the title bar; its height is fixed
        self.set_aoi(x1, y1, x1 + w - 1, y1 + 245)
        self.located = True
        return True

    def is_open(self) -> bool:
        if not self.located:
            return False
        x1, y1, x2, _ = self.get_bbox()
        h, _ = self.templates['title'].shape
        # get_img_at includes the bottom right pixel
        img = self.client.get_img_at((x1, y1, x2, y1 + h - 1))
        if img is None or img.shape != self.templates['title'].shape:
            return False
        score = cv2.matchTemplate(img, self.templates['title'],
                                  cv2.TM_CCOEFF_NORMED).max()
        return self.match_threshold <= score < 1.01

    @staticmethod
    def _bar_status(bgr) -> str:
        b, g, r = (int(c) for c in bgr)
        if g > r + 40 and g > b + 40:
            return 'complete'
        if r > g + 40 and r > b + 40 and g < 60:
            return 'cancelled'
        if r > 120 and g > 60 and b < 60:
            return 'active'
        return 'empty'

    def read_slots(self) -> List[OfferSlot]:
        slots = []
        for i in range(SLOTS):
            bar = self.client.get_img_at(self._in_slot(i, BAR),
                                         mode=self.client.BGRA)
            colour = numpy.median(bar[:, :, :3].reshape(-1, 3), axis=0)
            items, quantities = [], []
            for box in ITEMS:
                img = self.client.get_img_at(self._in_slot(i, box))
                items.append(bool(img is not None and img.size
                                  and img.std() > ITEM_STD))
                x, y = box[:2]
                q = QUANTITY
                quantities.append(read_quantity(self.client.get_img_at(
                    self._in_slot(i, (x + q[0], y + q[1], x + q[2],
                                      y + q[3])), mode=self.client.BGRA))
                    if items[-1] else None)
            row = self.client.get_img_at(self._in_slot(i, BAR_ROW),
                                         mode=self.client.BGRA)[0]
            states = [self._bar_status(c[:3]) for c in row]
            status = self._bar_status(colour)
            progress = 1.0 if status in ('complete', 'cancelled') else \
                states.count('active') / len(states)
            icon = self.client.get_img_at(self._in_slot(i, OFFER_ICON))
            if status == 'empty' and ('active' in states or (
                    icon is not None and icon.size
                    and icon.std() > ICON_STD)):
                status = 'active'       # little or nothing filled yet
            slots.append(OfferSlot(i, status, tuple(items), progress,
                                   tuple(quantities)))
        return slots

    @property
    def collectable(self) -> bool:
        return any(s.collectable for s in self.slots)

    def update(self):
        super().update()
        if not self.located:
            self.locate()
        self.open = self.is_open()
        if not self.open and self.located:
            # it can open somewhere else next time (window resized)
            self.located = False
        self.slots = self.read_slots() if self.open else []
