"""Teleporting with an object's teleport menu, e.g. the house's ornate
jewellery box: use its menu option, wait for the menu, press the
destination's hotkey, check the landing.

    tele = ObjectMenuTeleport(app, box_entity, menu, method)
    ...  # every frame:
    result = tele.step()   # None: working on it, True: landed, False: failed

``menu`` and ``method`` come from ``config/teleports.json`` (see
:class:`wizard_eyes.teleports.Teleports`). Every step has an expectation:

1. the option is used (``UseOption``: left click or right-click menu);
   the menu title must show within ``MENU_SECONDS``, else try once more
2. the hotkey is pressed; the menu must close within ``KEY_SECONDS``
   (pressed once more if not), and the player must leave (the position
   jumps or leaves the house) within ``LAND_SECONDS``
3. the landing must be in range of one of the method's landing points

A failed expectation is logged and the teleport gives up (returns False)
after ``MAX_TRIES``; it never keeps pressing keys blindly.
"""

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy

from .file_path_utils import ensure_local, get_root
from .interactions import UseOption
from .teleports import Method

MENU_SECONDS = 3.0
KEY_SECONDS = 2.0
LAND_SECONDS = 8.0
SETTLE_SECONDS = 1.2
MAX_TRIES = 2
TITLE_THRESHOLD = 0.9

_templates = {}


def load_title(name: str) -> numpy.ndarray:
    """A menu title template (grayscale), e.g. 'house/jewellery_box_title'."""
    if name not in _templates:
        path = Path(get_root()) / 'data' / f'{name}.npy'
        ensure_local(str(path))
        _templates[name] = numpy.load(str(path))
    return _templates[name]


def title_showing(client, name: str) -> bool:
    """Whether the menu title is on screen. The menu is drawn in the
    middle of the game screen, so only the middle half of the client
    (either way) is searched; that follows the window size."""
    img = client.img
    if img is None:
        return False
    h, w = img.shape[:2]
    img = img[h // 8:h * 5 // 8, w // 4:w * 3 // 4]
    tpl = load_title(name)
    if img is None or img.shape[0] < tpl.shape[0] or \
            img.shape[1] < tpl.shape[1]:
        return False
    score = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED).max()
    return float(score) >= TITLE_THRESHOLD


class ObjectMenuTeleport:

    def __init__(self, app, target, menu: dict, method: Method):
        self.app = app
        self.client = app.client
        self.target = target
        self.menu = menu
        self.method = method
        self.key = method.hotkey
        if not self.key:
            raise ValueError(f'{method.key}: no hotkey')
        self.state = 'start'
        self.use: Optional[UseOption] = None
        self.since = 0.0
        self.tries = 0
        self.key_presses = 0
        self.start: Optional[tuple] = None
        self.left_at = 0.0

    @property
    def log(self):
        return self.client.logger

    def _position(self):
        x, y, z = self.client.gauges.minimap.gps.get_coordinates()
        return None if x == -1 else (int(x), int(y), int(z))

    def _fail(self, message: str) -> bool:
        self.log.warning(f'{self.method.key}: expected {message}')
        return False

    def showing(self) -> bool:
        return title_showing(self.client, self.menu['title_template'])

    def step(self) -> Optional[bool]:
        """The menu title is only looked for once at the start (in case
        it was left open), after the object was clicked, and after the
        hotkey was pressed."""
        now = time.time()

        if self.state == 'start':
            self.state = 'menu' if self.showing() else 'use'
            self.since = now
            return None

        if self.state == 'use':
            if self.use is None:
                if self.tries >= MAX_TRIES:
                    return self._fail(f'the {self.menu["object"]} menu to '
                                      f'open (tried {self.tries} times)')
                self.tries += 1
                self.use = UseOption(self.app, self.target,
                                     self.menu['option'],
                                     [self.menu['object']])
            result = self.use.step()
            if result is False:
                return self._fail(f'to use {self.menu["option"]!r} on the '
                                  f'{self.menu["object"]}')
            if result:
                self.state, self.since = 'opening', now
            return None

        if self.state == 'opening':
            if self.showing():
                self.state, self.since = 'menu', now
                return None
            if now - self.since > MENU_SECONDS:
                self.log.warning(f'{self.method.key}: no menu '
                                 f'{MENU_SECONDS:.0f}s after using '
                                 f'{self.menu["option"]!r}; trying again')
                self.use = None
                self.state = 'use'
            return None

        if self.state == 'menu':
            if self.key_presses >= MAX_TRIES:
                return self._fail(f'the menu to close after pressing '
                                  f'{self.key!r} {self.key_presses} times')
            self.start = self._position()
            self.client.screen.press_key(self.key)
            self.key_presses += 1
            self.log.info(f'{self.method.key}: pressed {self.key!r} '
                          f'({self.menu["hotkeys"].get(self.key)})')
            self.state, self.since = 'pressed', now
            return None

        if self.state == 'pressed':
            if not self.showing():
                self.state, self.since = 'leaving', now
                return None
            if now - self.since > KEY_SECONDS:
                self.state = 'menu'  # press once more
            return None

        if self.state == 'leaving':
            pos = self._position()
            moved = pos is not None and (
                self.start is None or pos[2] != self.start[2]
                or max(abs(pos[0] - self.start[0]),
                       abs(pos[1] - self.start[1])) > 20)
            if moved and not self.left_at:
                self.left_at = now
            if self.left_at and now - self.left_at >= SETTLE_SECONDS:
                point = self.method.landing_point(pos) if pos else None
                if point is None:
                    return self._fail(f'to land near '
                                      f'{self.method.lands.get("tiles")}, '
                                      f'landed at {pos}')
                self.log.info(f'{self.method.key}: landed at {pos}, near '
                              f'{point}')
                return True
            if now - self.since > LAND_SECONDS:
                return self._fail(f'a teleport within {LAND_SECONDS:.0f}s '
                                  f'of the menu closing (position {pos})')
            return None

        return None
