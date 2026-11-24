"""Teleporting with a worn item's right-click option, e.g. a ring of wealth's
"Grand Exchange".

    tele = WornTeleport(app, method)
    ...  # every frame:
    tele.update()
    result = tele.step()   # None: working on it, True: landed, False: failed
    tele.worn              # 'uncharged' / 'empty' if that's why it failed

``method`` comes from ``config/teleports.json``: ``worn_option`` and ``slot``
(where it's worn). The equipment tab and its slots are placed relative to the
inventory tab.

Expectations: right-clicking the slot shows a menu with exactly one line
reading the option and one of the method's items (an empty slot's menu has no
such line); the position then jumps to near a landing point. Anything else
fails, never retried.

The item's charges aren't read from its name: OCR misreads the digits.
"""
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy

from .file_path_utils import ensure_local, get_root
from .game_objects.game_objects import GameObject
from .screen_text import colour_mask, menu_line, read
from .teleports import Method, name_matches

TAB_SECONDS = 2.0
MAX_TRIES = 2
MENU_SECONDS = 3.0
MENU_SETTLE = 0.5
LAND_SECONDS = 10.0
SETTLE_SECONDS = 1.2
JUMP_TILES = 20

EQUIPMENT_TAB = 33
"""Right of the inventory tab."""
SLOTS = {'ring': (267, -105)}
"""Middle of a worn slot, from the inventory tab's middle."""
SLOT_HALF = 10
STATES = {'ring': {'ring_empty': 'empty',
                   'ring_of_wealth_uncharged': 'uncharged',
                   'ring_of_wealth': 'charged'}}
"""Templates of what a slot can hold (``data/teleport_items/worn``)."""
STATE_AREA = 18
STATE_SCORE = 0.85
STATE_MARGIN = 0.03


def worn_state(img: Optional[numpy.ndarray], slot: str) -> Optional[str]:
    """What a grey image of a worn slot shows: 'charged', 'uncharged',
    'empty' or None (something else, or unclear)."""
    scored = []
    for name, state in STATES[slot].items():
        path = Path(get_root()) / 'data' / 'teleport_items' / 'worn' /             f'{name}.npy'
        ensure_local(str(path))
        tpl = numpy.load(str(path))
        if img is None or img.shape[0] < tpl.shape[0] or                 img.shape[1] < tpl.shape[1]:
            return None
        score = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED).max()
        scored.append((float(score), state))
    scored.sort(reverse=True)
    if scored[0][0] < STATE_SCORE or             scored[0][0] - scored[1][0] < STATE_MARGIN:
        return None
    return scored[0][1]


class WornTeleport:

    def __init__(self, app, method: Method):
        if not method.worn_option or method.slot not in SLOTS:
            raise ValueError(f'{method.key}: not a worn item with an option')
        self.app = app
        self.client = app.client
        self.method = method
        self.state = 'tab'
        self.since = time.time()
        self.tries = 0
        self.target: Optional[GameObject] = None
        self.start: Optional[tuple] = None
        self.left_at = 0.0
        self.worn: Optional[str] = None
        self.failed: Optional[str] = None

    @property
    def log(self):
        return self.client.logger

    def _position(self):
        x, y, z = self.client.gauges.minimap.gps.get_coordinates()
        return None if x == -1 else (int(x), int(y), int(z))

    def _fail(self, message: str) -> bool:
        self.failed = f'{self.method.key}: {message}'
        self.log.warning(f'Teleport {self.failed}')
        return False

    def _go(self, state: str):
        self.state, self.since = state, time.time()

    def update(self):
        self.client.tabs.update()

    def step(self) -> Optional[bool]:
        if self.failed is not None:
            return False
        if self.client.ocr is None:
            return self._fail('needs OCR')
        return getattr(self, f'_{self.state}')(time.time())

    def _tab(self, now) -> Optional[bool]:
        """From the inventory to the equipment tab. (Clicking an open tab
        closes it, so the inventory is opened first if it isn't.)"""
        inv = self.client.tabs.inventory
        if not inv.located or not (inv.state or '').endswith('selected'):
            if now - self.since > 3 * TAB_SECONDS:
                return self._fail('the inventory did not open')
            if inv.located and not inv.clicked:
                self.app._click_tab(inv)
            return None
        x1, y1, x2, y2 = inv.get_bbox()
        self.client.screen.click_aoi(
            x1 + EQUIPMENT_TAB + 6, y1 + 6, x2 + EQUIPMENT_TAB - 6, y2 - 6,
            pause_before_click=True)
        self._go('slot')
        return None

    def _slot(self, now) -> Optional[bool]:
        inv = self.client.tabs.inventory
        if (inv.state or '').endswith('selected'):
            if now - self.since > TAB_SECONDS:
                return self._fail('the equipment tab did not open')
            return None
        if now - self.since < MENU_SETTLE:
            return None
        x1, y1, x2, y2 = inv.get_bbox()
        dx, dy = SLOTS[self.method.slot]
        x, y = (x1 + x2) // 2 + dx, (y1 + y2) // 2 + dy
        self.worn = worn_state(self.client.get_img_at(
            (x - STATE_AREA, y - STATE_AREA, x + STATE_AREA,
             y + STATE_AREA)), self.method.slot)
        if self.worn in ('uncharged', 'empty'):
            return self._fail(f'the worn {self.method.slot} is {self.worn}')
        self.target = GameObject(self.client, self.client)
        self.target.set_aoi(x - SLOT_HALF, y - SLOT_HALF, x + SLOT_HALF,
                            y + SLOT_HALF)
        if self.app._right_click(self.target):
            self.tries += 1
            self._go('menu')
        return None

    def _menu(self, now) -> Optional[bool]:
        self.target.update_context_menu()
        menu = self.target.context_menu
        texts: List[str] = []
        if menu is not None and menu.items and \
                now - self.since > MENU_SETTLE:
            imgs = [self.client.get_img_at(i.get_bbox(),
                                           mode=self.client.BGRA)
                    for i in menu.items]
            texts = [read(self.client.ocr, colour_mask(i, 'white'))
                     for i in imgs]
            k = menu_line(texts, self.method.worn_option)
            if k is not None:
                name = read(self.client.ocr, colour_mask(imgs[k], 'orange'))
                if name_matches(name, self.method.item):
                    self.start = self._position()
                    menu.items[k].click(tmin=.6, tmax=.9,
                                        pause_before_click=True)
                    self.log.info(f'Teleport {self.method.key}: {texts[k]!r} '
                                  f'on {name!r} from {self.start}')
                    self._go('leaving')
                    return None
                texts.append(name)
        if now - self.since > MENU_SECONDS:
            if menu is not None:
                self.client.screen.mouse_away_object(menu)
            if self.tries < MAX_TRIES and not texts:
                self._go('slot')    # no menu at all: once more
                return None
            return self._fail(f'no {self.method.worn_option!r} on the worn '
                              f'{self.method.slot} (menu {texts})')
        return None

    def _leaving(self, now) -> Optional[bool]:
        pos, start = self._position(), self.start
        if pos is not None and start is not None and not self.left_at and (
                pos[2] != start[2] or max(abs(pos[0] - start[0]),
                                          abs(pos[1] - start[1]))
                > JUMP_TILES):
            self.left_at = now
        if self.left_at:
            if now - self.left_at < SETTLE_SECONDS:
                return None
            if pos is None or self.method.landing_point(pos) is None:
                return self._fail(f'landed at {pos}, not near '
                                  f'{self.method.lands.get("tiles")}')
            self.log.info(f'Teleport {self.method.key}: landed at {pos}')
            return True
        if now - self.since > LAND_SECONDS:
            return self._fail(f'no teleport within {LAND_SECONDS:.0f}s '
                              f'(at {pos})')
        return None
