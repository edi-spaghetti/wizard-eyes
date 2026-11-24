"""Trading between two of our own bots, which tell each other where they are
through the coordination service.

    signal = TradeSignal(Coordinator(url, 'cleaner'), 'some-id', 'mixer')
    trade = PlayerTrade(app, 'PartnerName', signal, ask=True,
                        give=[Give('Cadantine', 100, icon)])
    ...  # every frame:
    trade.update()
    result = trade.step()   # None: working on it, True: traded, False: failed

Both bots run one, with the same trade id; one of them asks (``ask``), the
other answers the request in its chat box (which must be open). They agreed
on the items beforehand, so nothing is checked on the trade screens: a bot
signals "offered" once its items are up, and each accepts both screens once
the other has. Being the right player is settled by the request: it's made
from, and answered on, the partner's name.

Stages: ``here`` (with the tile), ``offered``, ``done`` / ``failed``.
"""
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import keyboard
import numpy

from .coordination import CoordinatorUnavailable
from .file_path_utils import ensure_local, get_root
from .screen_text import alike, colour_mask, lines, menu_line, pick, read

Box = Tuple[int, int, int, int]

MATCH = 0.8
PROMPT_MATCH = 0.8
POLL_SECONDS = 1.0
WAIT_SECONDS = 150.0
"""For the partner, at each stage."""
REQUEST_SECONDS = 25.0
"""From asking to the screen opening (walking there included); then it's
asked again."""
STEP_SECONDS = 4.0
SETTLE_SECONDS = 0.6
CLOSED_SECONDS = 2.0
"""No trade screen for this long: it's closed, not changing."""
TOTAL_SECONDS = 300.0
MAX_TRIES = 3

# relative to the first screen's Decline text (its template's top left)
FIRST_ACCEPT = (-12, -78, 54, -52)
FIRST_DECLINE = (-12, -4, 50, 18)
# relative to the second screen's
SECOND_ACCEPT = (-74, -2, -24, 16)
BESIDE = (-74, 0, -34, 12)
"""Left of Decline: on the second screen that's the Accept button (dark), on
the first one Accept is above it."""
BUTTON_DARK = 45

_cache: Dict[str, numpy.ndarray] = {}


def load(name: str) -> numpy.ndarray:
    if name not in _cache:
        path = Path(get_root()) / 'data' / f'{name}.npy'
        ensure_local(str(path))
        _cache[name] = numpy.load(str(path))
    return _cache[name]


def locate(img: Optional[numpy.ndarray], name: str, least: float = MATCH
           ) -> Optional[Tuple[int, int]]:
    tpl = load(name)
    if img is None or img.shape[0] < tpl.shape[0] or \
            img.shape[1] < tpl.shape[1]:
        return None
    m = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
    _, score, _, at = cv2.minMaxLoc(m)
    return at if score >= least else None


@dataclass
class Give:
    item: str
    quantity: int
    icon: object        # the inventory icon of the (noted) stack


class TradeSignal:
    """What the two bots tell each other about one trade."""

    def __init__(self, coordinator, trade_id: str, partner_role: str,
                 clock=time.time):
        self.coordinator = coordinator
        self.trade_id = trade_id
        self.partner_role = partner_role
        self.clock = clock
        self.stage: Optional[str] = None
        self.seen: dict = {}
        self.polled = -float('inf')
        self.error: Optional[str] = None

    def post(self, stage: str, **info):
        """Tell the partner."""
        self.stage = stage
        try:
            self.coordinator.set_shared(
                f'trade-{self.coordinator.bot}',
                {'id': self.trade_id, 'stage': stage, **info})
            self.error = None
        except CoordinatorUnavailable as e:
            self.error = str(e)

    def partner(self) -> dict:
        """What the partner said last ({}: nothing about this trade yet)."""
        now = self.clock()
        if now - self.polled >= POLL_SECONDS:
            self.polled = now
            try:
                entry = self.coordinator.shared().get(
                    f'trade-{self.partner_role}') or {}
                value = entry.get('value') or {}
                if value.get('id') == self.trade_id:
                    self.seen = value
                self.error = None
            except CoordinatorUnavailable as e:
                self.error = str(e)
        return self.seen


class PlayerTrade:

    def __init__(self, app, partner: str, signal: TradeSignal,
                 ask: bool = False, give: List[Give] = ()):
        self.app = app
        self.client = app.client
        self.partner = partner
        self.signal = signal
        self.ask = ask
        self.todo = list(give)
        self.state = 'request'
        self.started = self.since = time.time()
        self.entity = None
        self.tries = 0
        self.posted_at = 0.0
        self.closed_at: Optional[float] = None
        self.menu_target = None
        self.before: Optional[numpy.ndarray] = None
        self.failed: Optional[str] = None

    @property
    def log(self):
        return self.client.logger

    # plumbing

    def _go(self, state: str, pause: bool = True):
        self.state = state
        self.since = time.time() + (random.uniform(.6, 1.5) if pause else 0)

    def _waited(self) -> float:
        return time.time() - self.since

    def _click(self, at, rel: Box, pad: int = 4):
        x, y = at
        x1, y1, x2, y2 = self.client.globalise(
            x + rel[0], y + rel[1], x + rel[2], y + rel[3])
        self.client.screen.click_aoi(x1 + pad, y1 + pad, x2 - pad, y2 - pad,
                                     pause_before_click=True)

    def screen(self) -> Tuple[Optional[str], Optional[Tuple[int, int]]]:
        """('first' / 'second' / None, where its Decline is)."""
        img = self.client.img
        at = locate(img, 'trade/first_decline')
        if at is None:
            return None, None
        x, y = at
        x1, y1, x2, y2 = BESIDE
        beside = img[max(0, y + y1):y + y2, max(0, x + x1):max(0, x + x2)]
        second = beside.size and numpy.median(beside) < BUTTON_DARK
        return ('second' if second else 'first'), at

    def _fail(self, why: str) -> bool:
        name, at = self.screen()
        if at is not None:
            self._click(at, FIRST_DECLINE)
        self.failed = why
        self.signal.post('failed', why=why)
        self.log.warning(f'Trade with {self.partner!r}: {why}')
        return False

    def _inventory(self) -> Optional[numpy.ndarray]:
        img = self.client.get_img_at(
            self.client.tabs.inventory.interface.get_bbox())
        return None if img is None else img.copy()

    def _position(self) -> Optional[List[int]]:
        pos = [int(i) for i in
               self.client.gauges.minimap.gps.get_coordinates()]
        return None if pos[0] == -1 else pos

    def _menu_texts(self, menu) -> List[str]:
        return [read(self.client.ocr, colour_mask(
            self.client.get_img_at(i.get_bbox(), mode=self.client.BGRA),
            'white')) for i in menu.items]

    def update(self):
        if self.entity is not None and self.state in ('request', 'menu'):
            self.app._update_game_entities(self.entity)

    def step(self) -> Optional[bool]:
        if self.failed is not None:
            return False
        now = time.time()
        if now < self.since:
            return None
        if self.client.ocr is None:
            return self._fail('needs OCR')
        if now - self.started > TOTAL_SECONDS:
            return self._fail('took too long')
        if self.state != 'accept' and \
                self.signal.partner().get('stage') == 'failed':
            return self._fail('the partner failed: '
                              f'{self.signal.partner().get("why")}')
        return getattr(self, f'_{self.state}')()

    # getting the trade screen open

    def _opened(self) -> bool:
        if self.screen()[0] != 'first':
            return False
        self.before = self._inventory()
        self._go('offer')
        return True

    def _request(self) -> Optional[bool]:
        if self._opened():
            return None
        if self._waited() > WAIT_SECONDS:
            return self._fail(f'no trade screen ({self.signal.error or ""})')
        if time.time() - self.posted_at > POLL_SECONDS:
            self.posted_at = time.time()
            self.signal.post('here', tile=self._position())
        return self._ask() if self.ask else self._answer()

    def _ask(self) -> Optional[bool]:
        tile = self.signal.partner().get('tile')
        if not tile:
            return None
        if self.tries >= MAX_TRIES:
            return self._fail('asked, no trade screen')
        if self.entity is None or \
                tuple(self.entity.get_global_coordinates()) != tuple(tile):
            self.entity = self.client.game_screen.create_game_entity(
                self.partner, self.partner, (0, 0), self.client, self.client)
            self.entity.set_global_coordinates(*tile)
            return None     # placed next frame
        if self.app._right_click(self.entity):
            self.tries += 1
            self._go('menu', pause=False)
        return None

    def _menu(self) -> Optional[bool]:
        menu = self.entity.context_menu
        if menu is not None and menu.items and \
                self._waited() > SETTLE_SECONDS:
            texts = self._menu_texts(menu)
            k = pick(texts, f'Trade with {self.partner}', 0.65)
            if k is not None and alike(texts[k][:10], 'Trade with') > 0.7:
                menu.items[k].click(tmin=.6, tmax=.9,
                                    pause_before_click=True)
                self.log.info(f'Trade: asked ({texts[k]!r})')
                self._go('asked')
                return None
        if self._waited() > STEP_SECONDS:
            if menu is not None:
                self.client.screen.mouse_away_object(menu)
            self._go('request')
        return None

    def _asked(self) -> Optional[bool]:
        if not self._opened() and self._waited() > REQUEST_SECONDS:
            self._go('request')
        return None

    def request_line(self) -> Optional[Box]:
        """Where the chat box has the partner's request (the last one)."""
        img = self.client.original_img
        h, w = img.shape[:2]
        y0 = h * 3 // 4
        mask = colour_mask(img[y0:, :min(w // 2, 520)], 'purple')
        for a, b in reversed(lines(mask)):
            text = read(self.client.ocr, mask[max(0, a - 2):b + 2])
            if alike(text, f'{self.partner} wishes to trade with you') > 0.7:
                cols = numpy.where(mask[a:b].any(axis=0))[0]
                return int(cols[0]), y0 + a, int(cols[-1]), y0 + b
        return None

    def _answer(self) -> Optional[bool]:
        self.client.img     # make sure this frame's images are grabbed
        line = self.request_line()
        if line is not None:
            self._click((0, 0), line, pad=2)
            self._go('answered')
        return None

    def _answered(self) -> Optional[bool]:
        # (an old request's line may have been clicked: look again)
        if not self._opened() and self._waited() > STEP_SECONDS:
            self._go('request')
        return None

    # offering

    def _offer(self) -> Optional[bool]:
        if self.screen()[0] is None:
            return self._closed()
        self.closed_at = None
        if not self.todo:
            self.signal.post('offered')
            self._go('accept')
            return None
        self.menu_target = self.todo[0].icon
        if self.app._right_click(self.menu_target):
            self._go('offer_menu', pause=False)
        return None

    def _offer_menu(self) -> Optional[bool]:
        self.menu_target.update_context_menu()
        menu = self.menu_target.context_menu
        texts: List[str] = []
        if menu is not None and menu.items and \
                self._waited() > SETTLE_SECONDS:
            texts = self._menu_texts(menu)
            k = menu_line([t.replace('%', 'X') for t in texts], 'Offer-X')
            if k is not None:
                menu.items[k].click(tmin=.6, tmax=.9,
                                    pause_before_click=True)
                self._go('amount', pause=False)
                return None
        if self._waited() > STEP_SECONDS:
            if menu is not None:
                self.client.screen.mouse_away_object(menu)
            self.tries += 1
            if self.tries > 2 * MAX_TRIES:
                return self._fail(f'no "Offer-X" in the menu: {texts}')
            self._go('offer')
        return None

    def _amount(self) -> Optional[bool]:
        img = self.client.img
        h, w = img.shape[:2]
        if locate(img[h // 2:, :w // 2], 'bank/withdraw_x/prompt_amount',
                  PROMPT_MATCH) is None:
            if self._waited() > STEP_SECONDS:
                return self._fail('the amount prompt did not open')
            return None
        g = self.todo.pop(0)
        for ch in str(int(g.quantity)):
            keyboard.write(ch)
            time.sleep(random.uniform(.12, .35))
        time.sleep(random.uniform(.3, .8))
        keyboard.press_and_release('enter')
        self.log.info(f'Trade: offered {g.quantity} x {g.item!r}')
        self._go('offer')
        return None

    # accepting

    def _accept(self) -> Optional[bool]:
        name, at = self.screen()
        if name is None:
            return self._closed()
        self.closed_at = None
        if name == 'first' and self.signal.partner().get('stage') not in (
                'offered', 'done'):
            if self._waited() > WAIT_SECONDS:
                return self._fail('the partner did not offer '
                                  f'({self.signal.error or ""})')
            return None
        # (still greyed, or waiting for the partner: clicked again later)
        self._click(at, FIRST_ACCEPT if name == 'first' else SECOND_ACCEPT)
        self.since = time.time() + random.uniform(2.5, 4.5)
        return None

    def _closed(self) -> Optional[bool]:
        """No trade screen: traded if the inventory is different now."""
        now = time.time()
        if self.closed_at is None:
            self.closed_at = now
        if now - self.closed_at < CLOSED_SECONDS:
            return None
        after = self._inventory()
        if self.state == 'accept' and after is not None and \
                self.before is not None and (
                    after.shape != self.before.shape or
                    numpy.abs(after.astype(int) - self.before).mean() > .05):
            self.signal.post('done')
            self.log.info(f'Trade with {self.partner!r}: done')
            return True
        if now - self.closed_at < STEP_SECONDS:
            return None
        return self._fail('the trade screen closed, nothing traded')
