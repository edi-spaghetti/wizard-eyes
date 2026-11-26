"""Minigame teleports: the spellbook's "Minigame Teleport" (second icon, next
to the home teleport, on every spellbook) opens a scrolling menu of them.

    tele = MinigameTeleport(app, method)
    ...  # every frame:
    tele.update()
    result = tele.step()   # None: working on it, True: landed, False: failed

``method`` comes from ``config/teleports.json`` (``minigame``: its emblem's
template, ``data/minigames/<minigame>.npy``). The menu marks the ones the
account can't use with red text: those aren't clicked.

All minigame teleports share a 20 minute cooldown, remembered per account
across restarts (``data/teleports/<account>.local.json``).

Some (TzHaar Fight Pit) first say which world the minigame is on, with
"Click here to continue" in the chat box; the cast only starts once that's
continued (space), so it's pressed when seen (``continue.npy``).

Expectations: the menu opens after the spell is clicked; scrolling shows the
emblem before the list ends; the position jumps within ``LAND_SECONDS`` (of
the click, or of the last continue), to near the method's landing point.
Anything else fails (never retried).
"""
import json
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy
import pyautogui

from .bank_pin import account_from_title
from .file_path_utils import ensure_local, get_root
from .teleports import Method

THRESHOLD = 0.9
TAB_SECONDS = 3.0
MAX_TAB_CLICKS = 2
MAGIC_TAB = 3
"""Tabs to the right of the inventory tab."""
MENU_SECONDS = 4.0
SCROLL_SECONDS = 0.7
MAX_SCROLLS = 25
SCROLL = -120
LAND_SECONDS = 25.0
"""The cast takes 11-15s."""
SETTLE_SECONDS = 1.5
JUMP_TILES = 20
MAX_CONTINUES = 2
CONTINUE_SECONDS = 2.0
"""Between presses, for the dialogue to go."""

# relative to the menu title's top left
LIST = (-208, 25, 268, 314)
"""The scrolling part of the menu (x1, y1, x2, y2)."""
TILE = (-7, -4, 218, 68)
"""A minigame's tile, relative to its emblem's top left."""
LOCKED_PIXELS = 15

_templates = {}


def load_template(name: str) -> numpy.ndarray:
    if name not in _templates:
        path = Path(get_root()) / 'data' / 'minigames' / f'{name}.npy'
        ensure_local(str(path))
        _templates[name] = numpy.load(str(path))
    return _templates[name]


def locate(img, name: str) -> Optional[Tuple[int, int]]:
    """Top left of a template in an image, or None."""
    tpl = load_template(name)
    if img is None or img.shape[0] < tpl.shape[0] or \
            img.shape[1] < tpl.shape[1]:
        return None
    scores = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
    _, score, _, (x, y) = cv2.minMaxLoc(scores)
    return (x, y) if score >= THRESHOLD else None


def is_locked(tile_bgra: numpy.ndarray) -> bool:
    """A tile whose requirements aren't met has them in red."""
    b, g, r = (tile_bgra[:, :, i].astype(int) for i in range(3))
    return int(((r > 180) & (g < 60) & (b < 60)).sum()) >= LOCKED_PIXELS


class Cooldowns:
    """When shared cooldowns end, per account."""

    def __init__(self, account: Optional[str], path=None):
        self.path = Path(path) if path else (
            Path(get_root()) / 'data' / 'teleports'
            / f'{account or "unknown"}.local.json')

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding='utf8'))
        except (OSError, ValueError):
            return {}

    def left(self, name: str) -> float:
        return max(0.0, self._read().get(name, 0.0) - time.time())

    def start(self, name: str, seconds: float):
        data = self._read()
        data[name] = time.time() + seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data), encoding='utf8')


class MinigameTeleport:

    def __init__(self, app, method: Method,
                 cooldowns: Optional[Cooldowns] = None):
        if not method.minigame:
            raise ValueError(f'{method.key}: not a minigame teleport')
        self.app = app
        self.client = app.client
        self.method = method
        self.cooldowns = cooldowns or Cooldowns(account_from_title(
            getattr(self.client, 'title', '')))
        self.state = 'tab'
        self.since = time.time()
        self.scrolls = 0
        self.tab_clicks = 0
        self.over_list = False
        self.unchanged = 0
        self.last_list: Optional[numpy.ndarray] = None
        self.start: Optional[tuple] = None
        self.left_at = 0.0
        self.continues = 0
        self.continued_at = 0.0
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

    def _click(self, x1, y1, x2, y2):
        """Click a box given in client image coordinates."""
        ox, oy = self.client.get_bbox()[:2]
        self.client.screen.click_aoi(ox + x1, oy + y1, ox + x2, oy + y2,
                                     pause_before_click=True)

    def update(self):
        self.client.tabs.update()

    def step(self) -> Optional[bool]:
        if self.failed is not None:
            return False
        return getattr(self, f'_{self.state}')(time.time())

    def _tab(self, now):
        left = self.cooldowns.left('minigame')
        if left:
            return self._fail(f'on cooldown for {left:.0f}s')
        self._go('spell')
        return None

    def _find_spell(self) -> Optional[Tuple[int, int]]:
        img = self.client.img
        if img is None:
            return None
        h, w = img.shape[:2]
        ox, oy = w // 2, h // 3  # the side panel is bottom right
        at = locate(img[oy:, ox:], 'spell')
        return None if at is None else (at[0] + ox, at[1] + oy)

    def _spell(self, now):
        """Click the spell; open the magic tab first if it isn't showing.
        The magic tab is found from the inventory tab, in the same row (its
        own icon looks different on each spellbook)."""
        at = self._find_spell()
        if at is not None:
            x, y = at
            self._click(x + 4, y + 4, x + 17, y + 17)
            self._go('menu')
            return None
        if now - self.since < (TAB_SECONDS if self.tab_clicks else 0):
            return None
        inv = self.client.tabs.inventory
        if not inv.located:
            if now - self.since > TAB_SECONDS:
                return self._fail('the inventory tab was not found')
            return None
        if self.tab_clicks >= MAX_TAB_CLICKS:
            return self._fail('no Minigame Teleport spell showing (not on '
                              'the magic tab, or on cooldown)')
        self.tab_clicks += 1
        x1, y1, x2, y2 = inv.get_bbox()
        dx = MAGIC_TAB * (x2 - x1 + 1)
        self.client.screen.click_aoi(x1 + dx + 6, y1 + 6, x2 + dx - 6, y2 - 6,
                                     pause_before_click=True)
        self._go('spell')
        return None

    def _menu(self, now):
        img = self.client.img
        title = locate(img, 'title')
        if title is None:
            if now - self.since > MENU_SECONDS:
                return self._fail('the menu did not open')
            return None
        if now - self.since < SCROLL_SECONDS:
            return None  # let it settle (opening / scrolling)
        tx, ty = title
        x1, y1, x2, y2 = (tx + LIST[0], ty + LIST[1],
                          tx + LIST[2], ty + LIST[3])
        listing = img[y1:y2, x1:x2]
        at = locate(listing, self.method.minigame)
        if at is not None:
            ex, ey = at[0] + x1, at[1] + y1
            tile = self.client.original_img[
                max(y1, ey + TILE[1]):min(y2, ey + TILE[3]),
                ex + TILE[0]:ex + TILE[2]]
            if is_locked(tile):
                self.client.screen.press_key('esc')
                return self._fail('not unlocked on this account')
            self.start = self._position()
            self._click(ex + 6, ey + 6, ex + TILE[2] - 12, ey + 56)
            self.log.info(f'Teleport {self.method.key}: casting from '
                          f'{self.start}')
            self._go('leaving')
            return None
        if not self.over_list:
            # the gap between the two columns: nothing highlights there. The
            # game needs a moment to notice the mouse before it'll scroll
            ox, oy = self.client.get_bbox()[:2]
            self.client.screen.mouse_to(ox + tx + 31, oy + ty + 150)
            self.over_list = True
            self.since = now
            return None
        same = self.last_list is not None and \
            self.last_list.shape == listing.shape and \
            numpy.array_equal(self.last_list, listing)
        self.unchanged = self.unchanged + 1 if same else 0
        if self.scrolls >= MAX_SCROLLS or self.unchanged >= 2:
            self.client.screen.press_key('esc')
            return self._fail(f'not in the menu ({self.scrolls} scrolls)')
        self.last_list = listing.copy()
        pyautogui.scroll(SCROLL)
        self.scrolls += 1
        self.since = now
        return None

    def _continue_showing(self) -> bool:
        """"Click here to continue" in the chat box (bottom left)."""
        img = self.client.img
        if img is None:
            return False
        h, w = img.shape[:2]
        return locate(img[h // 2:, :w // 2], 'continue') is not None

    def _leaving(self, now):
        pos, start = self._position(), self.start
        if pos is not None and start is not None and not self.left_at and (
                pos[2] != start[2] or max(abs(pos[0] - start[0]),
                                          abs(pos[1] - start[1]))
                > JUMP_TILES):
            self.left_at = now
            self.cooldowns.start('minigame', self.method.cooldown)
        if self.left_at:
            if now - self.left_at < SETTLE_SECONDS:
                return None
            if pos is None or self.method.landing_point(pos) is None:
                return self._fail(f'landed at {pos}, not near '
                                  f'{self.method.lands.get("tiles")}')
            self.log.info(f'Teleport {self.method.key}: landed at {pos}')
            return True
        if self.continues < MAX_CONTINUES and                 now - self.continued_at > CONTINUE_SECONDS and                 self._continue_showing():
            self.client.screen.press_key('space')
            self.continues += 1
            self.continued_at = self.since = now
            self.log.info(f'Teleport {self.method.key}: continued the '
                          f'dialogue ({self.continues})')
            return None
        if now - self.since > LAND_SECONDS:
            return self._fail(f'no teleport within {LAND_SECONDS:.0f}s '
                              f'(at {pos})')
        self.app.msg.append(f'casting {self.method.key}')
        return None
