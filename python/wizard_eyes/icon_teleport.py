"""Teleporting with an inventory item found by its picture, through its
right-click menu, with or without a submenu of destinations ("Teleport >").

    tele = IconTeleport(app, method)
    ...  # every frame:
    tele.update()
    result = tele.step()   # None: working on it, True: landed, False: failed

``method`` comes from ``config/teleports.json``: ``icon`` (a template in
``data/teleport_items/inventory``), ``option`` (the menu line) and ``choice``
(the submenu's line, if it has one).

Expectations: the inventory is open and holds the item; its menu has exactly
one line reading the option; hovering that shows a submenu with the choice;
the position then jumps to near a landing point within ``LAND_SECONDS``.
Anything else fails, never retried (a menu that didn't come up is, once).
"""
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy

from .file_path_utils import ensure_local, get_root
from .screen_text import colour_mask, menu_line, pick, read
from .teleports import Method

TAB_SECONDS = 6.0
ICON_DIFFERENCE = 12.0
"""Mean colour difference of a match (others are over 20)."""
ICON_SHIFT = 4
MENU_SECONDS = 3.0
MENU_SETTLE = 0.5
SUBMENU_SECONDS = 3.0
MAX_TRIES = 2
LAND_SECONDS = 12.0
SETTLE_SECONDS = 1.5
JUMP_TILES = 20

LINE = 15
SUBMENU_LINES = 12
MENU_BGR = (71, 84, 93)
"""Background of all menus."""
MAX_SUBMENU_WIDTH = 220

_cache: Dict[str, numpy.ndarray] = {}


def load_icon(name: str) -> numpy.ndarray:
    """BGR + mask."""
    if name not in _cache:
        path = Path(get_root()) / 'data' / 'teleport_items' / 'inventory' / \
            f'{name}.npy'
        ensure_local(str(path))
        _cache[name] = numpy.load(str(path))
    return _cache[name]


def icon_difference(bgr: numpy.ndarray, name: str) -> float:
    """Mean colour difference between an icon and the best placed part of an
    image (larger than the icon)."""
    icon = load_icon(name)
    want, mask = icon[..., :3].astype(int), icon[..., 3] > 0
    h, w = mask.shape
    best = float('inf')
    for y in range(bgr.shape[0] - h + 1):
        for x in range(bgr.shape[1] - w + 1):
            part = bgr[y:y + h, x:x + w, :3].astype(int)
            best = min(best, float(
                numpy.abs(part - want).max(axis=2)[mask].mean()))
    return best


def submenu_left(row: numpy.ndarray, right: int) -> Optional[int]:
    """Left edge of a submenu that ends left of column ``right``, from a row
    of pixels (BGR); None if there isn't one."""
    same = (row[:, :3] == MENU_BGR).all(axis=1)
    cols = numpy.where(same[max(0, right - MAX_SUBMENU_WIDTH):right])[0]
    if len(cols) < 20:
        return None
    return int(cols[0]) + max(0, right - MAX_SUBMENU_WIDTH)


class IconTeleport:

    def __init__(self, app, method: Method):
        if not method.icon:
            raise ValueError(f'{method.key}: no icon')
        self.app = app
        self.client = app.client
        self.method = method
        self.state = 'tab'
        self.since = time.time()
        self.tries = 0
        self.icon = None
        self.line_y = 0
        self.menu_left = 0
        self.start: Optional[tuple] = None
        self.left_at = 0.0
        self.landed: Optional[tuple] = None
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
        if self.state in ('tab', 'find'):
            self.client.tabs.update()

    def step(self) -> Optional[bool]:
        if self.failed is not None:
            return False
        if self.client.ocr is None:
            return self._fail('needs OCR')
        return getattr(self, f'_{self.state}')(time.time())

    def _tab(self, now) -> Optional[bool]:
        inv = self.client.tabs.inventory
        if (inv.state or '').endswith('selected') and inv.interface.located:
            self._go('find')
            return None
        if now - self.since > TAB_SECONDS:
            return self._fail('the inventory did not open')
        if inv.located and not inv.clicked and \
                not (inv.state or '').endswith('selected'):
            self.app._click_tab(inv)
        return None

    def find(self):
        """The inventory slot that holds the item, or None."""
        h, w = load_icon(self.method.icon).shape[:2]
        scored = []
        for icon in self.client.tabs.inventory.interface.icons.values():
            x1, y1, x2, y2 = icon.get_bbox()
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            img = self.client.get_img_at(
                (cx - w // 2 - ICON_SHIFT, cy - h // 2 - ICON_SHIFT,
                 cx + w // 2 + ICON_SHIFT, cy + h // 2 + ICON_SHIFT),
                mode=self.client.BGRA)
            if img is None or img.shape[0] < h or img.shape[1] < w:
                continue
            scored.append((icon_difference(img, self.method.icon),
                           len(scored), icon))
        if not scored or min(scored)[0] > ICON_DIFFERENCE:
            return None
        return min(scored)[2]

    def _find(self, now) -> Optional[bool]:
        self.icon = self.find()
        if self.icon is None:
            if now - self.since > MENU_SECONDS:
                return self._fail(f'no {self.method.icon} in the inventory')
            return None
        if self.app._right_click(self.icon):
            self.tries += 1
            self._go('menu')
        return None

    def _menu(self, now) -> Optional[bool]:
        self.icon.update_context_menu()
        menu = self.icon.context_menu
        texts: List[str] = []
        if menu is not None and menu.items and \
                now - self.since > MENU_SETTLE:
            texts = [read(self.client.ocr, colour_mask(
                self.client.get_img_at(i.get_bbox(), mode=self.client.BGRA),
                'white')) for i in menu.items]
            # (a submenu's ">" reads as a letter or two)
            k = menu_line([t.split(' >')[0] for t in texts],
                          self.method.option)
            if k is None:
                k = pick(texts, self.method.option, 0.8)
            if k is not None:
                return self._use(menu, k)
        if now - self.since > MENU_SECONDS:
            if menu is not None:
                self.client.screen.mouse_away_object(menu)
            if self.tries < MAX_TRIES and not texts:
                self._go('find')
                return None
            return self._fail(f'no {self.method.option!r} in the menu: '
                              f'{texts}')
        return None

    def _use(self, menu, k: int) -> None:
        item = menu.items[k]
        self.start = self._position()
        if not self.method.choice:
            item.click(tmin=.6, tmax=.9, pause_before_click=True)
            self.log.info(f'Teleport {self.method.key}: used '
                          f'{self.method.option!r} from {self.start}')
            self._go('leaving')
            return None
        x1, y1, x2, y2 = item.get_bbox()
        self.menu_left = menu.get_bbox()[0]
        self.line_y = (y1 + y2) // 2
        self.client.screen.mouse_to(x1 + 12, self.line_y)
        # (the submenu is outside the menu: the menu object would be reset)
        menu.reset()
        self._go('submenu')
        return None

    def read_submenu(self) -> Tuple[Optional[int], List[str]]:
        """(left edge, its lines) of the open submenu, client image
        coordinates. Its first line is level with the hovered one."""
        self.client.img     # make sure this frame's images are grabbed
        img = self.client.original_img
        ox, oy = self.client.get_bbox()[:2]
        y, right = self.line_y - oy, self.menu_left - ox
        # a row between two lines of text
        edge = submenu_left(img[y + LINE // 2 - 1, :], right)
        if edge is None:
            return None, []
        texts = []
        for i in range(SUBMENU_LINES):
            y1 = y - LINE // 2 + i * LINE
            part = img[y1:y1 + LINE, edge:right - 2]
            if not (part[..., :3] == MENU_BGR).all(axis=2).any():
                break
            texts.append(read(self.client.ocr, colour_mask(part, 'white')))
        return edge, texts

    def _submenu(self, now) -> Optional[bool]:
        if now - self.since < MENU_SETTLE:
            return None
        edge, texts = self.read_submenu()
        k = pick(texts, self.method.choice, 0.75) if texts else None
        if k is None:
            if now - self.since > SUBMENU_SECONDS:
                h, w = self.client.img.shape[:2]
                ox, oy = self.client.get_bbox()[:2]
                self.client.screen.mouse_to(ox + w // 2, oy + h // 3)
                return self._fail(f'no {self.method.choice!r} in the '
                                  f'submenu: {texts}')
            return None
        ox, oy = self.client.get_bbox()[:2]
        # sideways into the submenu first, so its parent line stays lit
        x = (ox + edge + self.menu_left) // 2
        y = self.line_y + k * LINE
        self.client.screen.mouse_to(x, self.line_y)
        self.client.screen.click_aoi(x - 10, y - 3, x + 10, y + 3,
                                     pause_before_click=True)
        self.log.info(f'Teleport {self.method.key}: chose {texts[k]!r} from '
                      f'{self.start}')
        self._go('leaving')
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
            self.landed = pos
            if self.method.lands.get('instance'):
                return True
            if pos is None or self.method.landing_point(pos) is None:
                return self._fail(f'landed at {pos}, not near '
                                  f'{self.method.lands.get("tiles")}')
            self.log.info(f'Teleport {self.method.key}: landed at {pos}')
            return True
        if now - self.since > LAND_SECONDS:
            return self._fail(f'no teleport within {LAND_SECONDS:.0f}s '
                              f'(at {pos})')
        return None
