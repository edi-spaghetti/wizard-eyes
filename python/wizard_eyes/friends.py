"""Hopping to a friend's world with the friends list's right-click "Hop-to".

    hop = HopToFriend(app, 'Name')
    ...  # every frame (app.client.tabs updated):
    result = hop.step()   # None: working on it, True: on the friend's world,
                          # False: failed (``failed`` says why)

The list is placed relative to the inventory tab. The friend must be in the
part of it that shows (online friends come first) and is found by name with
OCR. A friend's world is green when it's ours, so that's also the proof that
the hop worked; nothing is clicked if it's green to begin with.

Expectations: the list shows within ``TAB_SECONDS`` of clicking its tab; the
menu has exactly one line reading "Hop-to" (the same menu has "Delete"); the
friend's world turns green within ``HOP_SECONDS``. Tried twice at most.
"""
import time
from dataclasses import dataclass
from typing import List, Optional

from .game_objects.game_objects import GameObject
from .screen_text import colour_mask, menu_line, pick, read

TAB_SECONDS = 3.0
MAX_TAB_CLICKS = 2
MENU_SECONDS = 3.0
MENU_SETTLE = 0.4
HOP_SECONDS = 25.0
SETTLE_SECONDS = 4.0
"""Loading the new world: nothing is read or clicked meanwhile."""
MAX_TRIES = 2

# relative to the inventory tab's centre
FRIENDS_TAB = 131
TITLE = (116, -285, 288, -266)
FIRST_ROW = (124, -248, 283, -233)
ROW_PITCH = 15
ROWS = 12
NAME_WIDTH = 100
WORLD_PIXELS = 60


@dataclass
class Friend:
    row: int
    name: str           # as OCR reads it
    online: bool
    same_world: bool


class HopToFriend:

    def __init__(self, app, friend: str):
        self.app = app
        self.client = app.client
        self.friend = friend
        self.state = 'open'
        self.since = time.time()
        self.tab_clicks = 0
        self.tries = 0
        self.clicked_at = 0.0
        self.target: Optional[GameObject] = None
        self.failed: Optional[str] = None

    @property
    def log(self):
        return self.client.logger

    def _go(self, state: str):
        self.state, self.since = state, time.time()

    def _fail(self, why: str) -> bool:
        self.failed = why
        self.log.warning(f'Hop to friend: {why}')
        return False

    # reading

    def _box(self, rel):
        x1, y1, x2, y2 = self.client.tabs.inventory.get_bbox()
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        return rel[0] + cx, rel[1] + cy, rel[2] + cx, rel[3] + cy

    def row_box(self, i: int):
        x1, y1, x2, y2 = self._box(FIRST_ROW)
        return (x1, y1 + i * ROW_PITCH, x2, y2 + i * ROW_PITCH - 1)

    def _image(self, box):
        return self.client.get_img_at(box, mode=self.client.BGRA)

    def showing(self) -> bool:
        """Whether the friends list is open."""
        img = self._image(self._box(TITLE))
        if img is None or not img.size:
            return False
        return read(self.client.ocr, colour_mask(img, 'orange')
                    ).lower().startswith('friend')

    def rows(self) -> List[Friend]:
        out = []
        for i in range(ROWS):
            img = self._image(self.row_box(i))
            if img is None or not img.size:
                break
            name = read(self.client.ocr,
                        colour_mask(img[:, :NAME_WIDTH], 'white'))
            green = colour_mask(img[:, NAME_WIDTH - 5:], 'green').sum()
            yellow = colour_mask(img[:, NAME_WIDTH - 5:], 'yellow').sum()
            same = green >= WORLD_PIXELS
            out.append(Friend(i, name, same or yellow >= WORLD_PIXELS, same))
        return out

    def find(self) -> Optional[Friend]:
        rows = self.rows()
        i = pick([r.name for r in rows], self.friend)
        return None if i is None else rows[i]

    # steps

    def step(self) -> Optional[bool]:
        if self.client.ocr is None:
            return self._fail('needs OCR')
        if not self.client.tabs.inventory.located:
            if time.time() - self.since > TAB_SECONDS:
                return self._fail('the inventory tab was not found')
            return None
        return getattr(self, f'_{self.state}')(time.time())

    def _open(self, now) -> Optional[bool]:
        if self.showing():
            self._go('look')
            return None
        if self.tab_clicks and now - self.since < TAB_SECONDS:
            return None
        if self.tab_clicks >= MAX_TAB_CLICKS:
            return self._fail('the friends list did not open')
        self.tab_clicks += 1
        x1, y1, x2, y2 = self.client.tabs.inventory.get_bbox()
        self.client.screen.click_aoi(
            x1 + FRIENDS_TAB + 6, y1 + 6, x2 + FRIENDS_TAB - 6, y2 - 6,
            pause_before_click=True)
        self._go('open')
        return None

    def _look(self, now) -> Optional[bool]:
        friend = self.find()
        if friend is None:
            if now - self.since > TAB_SECONDS:
                return self._fail(f'{self.friend!r} is not in the list '
                                  f'({[r.name for r in self.rows()]})')
            return None
        if friend.same_world:
            self.log.info(f'Hop to friend: on the same world as '
                          f'{friend.name!r}')
            return True
        if not friend.online:
            return self._fail('offline')
        if self.tries >= MAX_TRIES:
            return self._fail('not on the same world after hopping')
        self.tries += 1
        x1, y1, x2, y2 = self.row_box(friend.row)
        self.target = GameObject(self.client, self.client)
        self.target.set_aoi(x1 + 4, y1 + 3, x1 + 50, y2 - 3)
        if self.app._right_click(self.target):
            self._go('menu')
        return None

    def _menu(self, now) -> Optional[bool]:
        self.target.update_context_menu()
        menu = self.target.context_menu
        texts = []
        if menu is not None and menu.items and \
                now - self.since > MENU_SETTLE:
            imgs = [self._image(i.get_bbox()) for i in menu.items]
            texts = [read(self.client.ocr, colour_mask(i, 'white'))
                     for i in imgs]
            k = menu_line(texts, 'Hop-to')
            if k is not None:
                name = read(self.client.ocr, colour_mask(imgs[k], 'orange'))
                if pick([name], self.friend) is None:
                    self.client.screen.mouse_away_object(menu)
                    return self._fail(f'the menu is for {name!r}')
                menu.items[k].click(tmin=.6, tmax=.9,
                                    pause_before_click=True)
                self.log.info(f'Hop to friend: hopping to {name!r}')
                self._go('hopping')
                return None
        if now - self.since > MENU_SECONDS:
            if menu is not None:
                self.client.screen.mouse_away_object(menu)
            return self._fail(f'no "Hop-to" in the menu: {texts}')
        return None

    def _hopping(self, now) -> Optional[bool]:
        if now - self.since < SETTLE_SECONDS:
            return None
        # the world switcher may have taken the list's place
        if not self.showing():
            if now - self.since > HOP_SECONDS:
                return self._fail('the friends list did not come back')
            if now - self.clicked_at > TAB_SECONDS + 1:
                # (clicking an open tab would close it)
                self.clicked_at = now
                x1, y1, x2, y2 = self.client.tabs.inventory.get_bbox()
                self.client.screen.click_aoi(
                    x1 + FRIENDS_TAB + 6, y1 + 6, x2 + FRIENDS_TAB - 6,
                    y2 - 6, pause_before_click=True)
            return None
        friend = self.find()
        if friend is not None and friend.same_world:
            self.log.info('Hop to friend: there')
            return True
        if now - self.since > HOP_SECONDS:
            self._go('look')
        return None
