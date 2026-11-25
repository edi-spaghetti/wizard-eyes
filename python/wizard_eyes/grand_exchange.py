"""The Grand Exchange interface: its two screens (the offer slots, and
setting up an offer), reading it, and placing an offer step by step.

Everything is placed relative to the screen's top bar (History button to
the close X), found by template; the bar differs between the two screens,
so it also tells which one is open. Measured on the resizable modern
layout (2026-11-22).

Numbers (quantity, price) are read with glyph templates of the game's
font (``data/ge/digits``), not OCR: tesseract misreads this font's 4s.
Text (offer type, item name, search results) is read with OCR.

Quantity and price are typed into a chat-box prompt. Typing is only done
once that prompt's question is seen (template); otherwise digits would go
to the chat input. Missing prompt templates make the offer stop there.

    offer = PlaceOffer(app, 'buy', 'Grimy ranarr weed', 100, 4_500)
    ...  # every frame:
    result = offer.step()  # None: working on it, True: done, False: failed

Without ``confirm=True`` it sets the offer up, checks every field, and goes
back without confirming (a dry run).
"""

import difflib
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import keyboard
import numpy
from PIL import Image

from .file_path_utils import ensure_local, get_root

Box = Tuple[int, int, int, int]

DATA = 'ge'
BAR_THRESHOLD = 0.95
PROMPT_THRESHOLD = 0.9
SLOT_THRESHOLD = 0.95

# offer slots, relative to the main screen's bar
SLOTS = 8
SLOT_ORIGIN = (-44, 60)
SLOT_PITCH = (117, 118)
SLOT_SIZE = (112, 107)
SLOT_COLUMNS = 4
# relative to a slot's top left
SLOT_PARTS: Dict[str, Box] = {
    'header': (0, 0, 112, 22),
    'buy': (5, 42, 51, 87),
    'sell': (62, 42, 107, 87),
    # a slot with an offer in it
    'quantity': (3, 28, 44, 48),
    'name': (44, 28, 111, 60),
    'bar': (4, 80, 108, 81),
    'price': (2, 87, 111, 105),
}
MAIN_PARTS: Dict[str, Box] = {
    'close': (401, 0, 422, 21),
    'collect': (340, 36, 417, 49),
}

# the offer status screen, relative to its bar (type and name are where the
# set-up screen has them)
STATUS_PARTS: Dict[str, Box] = {
    'type': (-7, 39, 63, 53),
    'name': (114, 37, 373, 55),
    'back': (-37, 249, -7, 274),
    'abort': (277, 241, 297, 260),     # (the icon left of it: modify)
    'bar': (7, 279, 295, 280),
}

# the offer set-up screen, relative to its bar
SETUP_PARTS: Dict[str, Box] = {
    'type': (-7, 39, 63, 53),          # "Buy offer" / "Sell offer"
    'name': (114, 37, 373, 55),        # item name
    'info': (117, 70, 393, 86),        # buy limit / actively traded price
    'item': (10, 64, 49, 99),          # item icon (opens the search)
    'quantity': (-15, 151, 137, 169),
    'price': (197, 151, 390, 169),
    'total': (-17, 204, 393, 224),
    'confirm': (113, 242, 263, 281),
    'back': (-37, 249, -7, 274),
    'quantity_+1': (-37, 174, -3, 197),
    'quantity_+10': (3, 174, 37, 197),
    'quantity_+100': (44, 174, 78, 197),
    'quantity_+1k': (85, 174, 118, 197),
    'quantity_all': (85, 174, 118, 197),    # the same one, selling
    'quantity_custom': (126, 174, 160, 197),
    'price_-99%': (174, 174, 208, 197),
    'price_-5%': (215, 174, 249, 197),
    'price_guide': (256, 174, 290, 197),
    'price_custom': (297, 174, 331, 197),
    'price_+5%': (338, 174, 372, 197),
    'price_+99%': (379, 174, 413, 197),
}

TITLE = (5, 19, 136, 239)
"""Rows and columns of "Grand Exchange" in the main screen's bar template."""
TITLE_CENTRE = 187
"""The title is centred here; RuneLite can add the offers' value to it
("Grand Exchange (334K)"), which moves the words."""

PAUSE = (0.9, 2.6)
DITHER = (3.0, 6.0)
DITHER_CHANCE = 0.12
KEY_DELAY = (0.12, 0.38)


def pause_seconds() -> float:
    """Breathing space before an action: an unhurried person, who sometimes
    stops to look."""
    return random.uniform(*(DITHER if random.random() < DITHER_CHANCE
                            else PAUSE))


def short_amount(n: int) -> str:
    """How to type an amount: the game takes k and m."""
    n = int(n)
    for letter, size in (('m', 1_000_000), ('k', 1_000)):
        if n and n % size == 0:
            return f'{n // size}{letter}'
    return str(n)


def type_slowly(text: str):
    for ch in text:
        keyboard.write(ch)
        time.sleep(random.uniform(*KEY_DELAY))


PROMPTS = {
    ('buy', 'search'): 'prompt_buy_search',
    ('buy', 'quantity'): 'prompt_buy_quantity',
    ('buy', 'price'): 'prompt_buy_price',
    ('sell', 'quantity'): 'prompt_sell_quantity',
    ('sell', 'price'): 'prompt_buy_price',      # the same question
}

_cache: Dict[str, Optional[numpy.ndarray]] = {}


def template(name: str) -> Optional[numpy.ndarray]:
    """A GE template (``data/ge/<name>.npy``), or None if there isn't
    one (yet)."""
    if name not in _cache:
        path = Path(get_root()) / 'data' / DATA / f'{name}.npy'
        ensure_local(str(path))
        _cache[name] = numpy.load(str(path)) if path.exists() else None
    return _cache[name]


def glyphs() -> Dict[str, numpy.ndarray]:
    out = {}
    for ch in '0123456789':
        g = template(f'digits/{ch}')
        if g is not None:
            out[ch] = g.astype(bool)
    comma = template('digits/comma')
    if comma is not None:
        out[','] = comma.astype(bool)
    return out


def columns(mask: numpy.ndarray) -> List[Tuple[int, int]]:
    """Runs of columns with any set pixel (one per glyph in this font)."""
    cols = mask.any(axis=0)
    runs, i = [], 0
    while i < len(cols):
        if cols[i]:
            j = i
            while j < len(cols) and cols[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def read_digits(mask: numpy.ndarray,
                font: Optional[Dict[str, numpy.ndarray]] = None
                ) -> Optional[int]:
    """The number at the start of a field ("4,700 coins" -> 4700), from a
    text mask the height of the glyph templates. None if the first glyph
    isn't a digit or one before the first non-digit doesn't match."""
    font = font if font is not None else glyphs()
    text = ''
    for a, b in columns(mask):
        seg = mask[:, a:b]
        match = None
        for ch, g in font.items():
            if g.shape == seg.shape and numpy.array_equal(g, seg):
                match = ch
                break
        if match is None:
            break  # "coins", or something unreadable
        text += match
    digits = text.replace(',', '')
    if not digits.isdigit():
        return None
    # commas must group by three
    if ',' in text and not re.fullmatch(r'\d{1,3}(,\d{3})*', text):
        return None
    return int(digits)


def find_main_title(region: numpy.ndarray, bar: numpy.ndarray
                    ) -> Optional[Tuple[int, int]]:
    """Top left of the main screen's bar in a grey image, from its title,
    with or without a value after it. None for the GE's other screens
    ("Grand Exchange: ...")."""
    r1, r2, c1, c2 = TITLE
    word = bar[r1:r2, c1:c2]
    if region.shape[0] < word.shape[0] or region.shape[1] < word.shape[1]:
        return None
    m = cv2.matchTemplate(region, word, cv2.TM_CCOEFF_NORMED)
    _, score, _, (x, y) = cv2.minMaxLoc(m)
    if score < BAR_THRESHOLD:
        return None
    row = (region[y:y + r2 - r1, x:x + 300] > 130).any(axis=0)
    end = last = c2 - c1 - 1
    for i in range(last + 1, len(row)):
        if row[i]:
            if i - end > 12:
                break
            if end == last and i - end < 5:
                return None     # a colon: another screen
            end = i
    return x - (TITLE_CENTRE - (end + 1) // 2), y - r1


@dataclass
class OfferView:
    """An offer as the main screen shows it."""
    index: int
    side: Optional[str]         # 'buy' / 'sell'
    item: Optional[str]
    quantity: Optional[int]     # rounded down from 100,000 ("135K")
    price: Optional[int]
    status: str                 # 'active', 'complete' or 'cancelled'
    progress: float             # of the bar, 0-1 (about 1% steps)

    @property
    def readable(self) -> bool:
        return None not in (self.side, self.item, self.quantity, self.price)


def bar_progress(colours: numpy.ndarray) -> Tuple[str, float]:
    """(status, share filled) from a row of the progress bar (BGR)."""
    from .game_objects.collection_box import CollectionBox
    states = [CollectionBox._bar_status(c[:3]) for c in colours]
    n = len(states)
    for status in ('cancelled', 'complete'):
        if states.count(status) > n / 2:
            return status, 1.0
    return 'active', states.count('active') / n


MISREADS = (('Fu', 'K'), ('Fi', 'A'), ('fi', 'A'))
"""How OCR reads some first letters of names."""


def best_name(text: str, items: Iterable[str], least: float = 0.8
              ) -> Optional[str]:
    """The item a misread name most likely is; None if none is close, or
    two are as close."""
    texts = [text] + [new + text[len(old):] for old, new in MISREADS
                      if (text or '').startswith(old)]
    scored = sorted(((max(similar(t, i) for t in texts), i) for i in items),
                    reverse=True)
    if not scored or scored[0][0] < least:
        return None
    if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.04:
        return None
    return scored[0][1]


def text_mask(hsv: numpy.ndarray) -> numpy.ndarray:
    """Pixels of the GE's bright text (orange / white / yellow, V 255) on
    the dark background."""
    return hsv[..., 2] > 130


def similar(a: str, b: str) -> float:
    a = re.sub(r'\W', '', a or '').lower()
    b = re.sub(r'\W', '', b or '').lower()
    return difflib.SequenceMatcher(None, a, b).ratio()


class GrandExchange:
    """Reads the GE interface from the client image (call :meth:`update`
    once per frame while it matters; nothing is read otherwise)."""

    def __init__(self, client):
        self.client = client
        self.screen: Optional[str] = None
        """'main' (offer slots), 'setup' (setting up an offer), 'status'
        (one offer) or None."""
        self.origin: Optional[Tuple[int, int]] = None
        """Top left of the bar, client image coordinates."""

    @property
    def log(self):
        return self.client.logger

    def update(self) -> Optional[str]:
        """Find which screen is open (if any)."""
        img = self.client.img
        self.screen, self.origin = None, None
        if img is None:
            return None
        h, w = img.shape[:2]
        # the interface is drawn in the middle of the game screen
        y0, x0 = h // 8, w // 6
        region = img[y0:h * 7 // 8, x0:w * 5 // 6]
        for screen in ('main', 'setup', 'status'):
            bar = template(f'{screen}_bar')
            if bar is None:
                continue
            m = cv2.matchTemplate(region, bar, cv2.TM_CCOEFF_NORMED)
            _, score, _, (x, y) = cv2.minMaxLoc(m)
            if score >= BAR_THRESHOLD:
                self.screen, self.origin = screen, (x0 + x, y0 + y)
                return screen
        bar = template('main_bar')
        found = find_main_title(region, bar) if bar is not None else None
        if found is not None:
            self.screen = 'main'
            self.origin = x0 + found[0], y0 + found[1]
        return self.screen

    # geometry (client image coordinates; inclusive boxes like get_img_at)

    def local_box(self, rel: Box) -> Box:
        bx, by = self.origin
        x1, y1, x2, y2 = rel
        return bx + x1, by + y1, bx + x2 - 1, by + y2 - 1

    def slot_box(self, i: int, part: Optional[str] = None) -> Box:
        r, c = divmod(i, SLOT_COLUMNS)
        sx = SLOT_ORIGIN[0] + SLOT_PITCH[0] * c
        sy = SLOT_ORIGIN[1] + SLOT_PITCH[1] * r
        if part is None:
            rel = (sx, sy, sx + SLOT_SIZE[0], sy + SLOT_SIZE[1])
        else:
            x1, y1, x2, y2 = SLOT_PARTS[part]
            rel = (sx + x1, sy + y1, sx + x2, sy + y2)
        return self.local_box(rel)

    def setup_box(self, part: str) -> Box:
        return self.local_box(SETUP_PARTS[part])

    def status_box(self, part: str) -> Box:
        return self.local_box(STATUS_PARTS[part])

    def global_box(self, box: Box) -> Box:
        return self.client.globalise(*box)

    def click(self, box: Box, pad: int = 4):
        x1, y1, x2, y2 = self.global_box(box)
        self.client.screen.click_aoi(x1 + pad, y1 + pad, x2 - pad, y2 - pad,
                                     pause_before_click=True)

    # reading

    def _crop(self, box: Box, hsv: bool = False) -> numpy.ndarray:
        self.client.img  # make sure this frame's images are grabbed
        img = self.client.hsv_img if hsv else self.client.img
        x1, y1, x2, y2 = box
        return img[y1:y2 + 1, x1:x2 + 1]

    def slot_empty(self, i: int) -> bool:
        tpl = template('slot_empty')
        # the origin can be a pixel out (see find_main_title)
        x1, y1, x2, y2 = self.slot_box(i, 'header')
        crop = self._crop((x1 - 3, y1, x2 + 3, y2))
        if tpl is None or crop.shape[0] < tpl.shape[0] or \
                crop.shape[1] < tpl.shape[1]:
            return False
        score = cv2.matchTemplate(crop, tpl, cv2.TM_CCOEFF_NORMED).max()
        return float(score) >= SLOT_THRESHOLD

    def empty_slots(self) -> List[int]:
        return [i for i in range(SLOTS) if self.slot_empty(i)]

    def read_slot(self, i: int, items: Iterable[str]
                  ) -> Optional[OfferView]:
        """The offer in a slot (main screen), None if it's empty. The name
        is matched to one of ``items``; what can't be read is None."""
        if self.slot_empty(i):
            return None
        from .game_objects.bank.quantity import (GREEN, WHITE, YELLOW,
                                                 read_quantity)
        header = self._ocr(self._crop(self.slot_box(i, 'header'), hsv=True))
        side = best_name(header, ('buy', 'sell'), 0.6)
        name = self._ocr(self._crop(self.slot_box(i, 'name'), hsv=True),
                         mode=6)
        x1, y1, x2, y2 = self.slot_box(i, 'quantity')
        area = self.client.original_img[y1:y2 + 1, x1:x2 + 1]
        quantity = read_quantity(area)
        if quantity is None and not any(
                (area[..., :3] == colour).all(axis=2).any()
                for colour in (YELLOW, WHITE, GREEN)):
            quantity = 1    # no number: a single one
        price = read_digits(text_mask(
            self._crop(self.slot_box(i, 'price'), hsv=True)))
        x1, y1, x2, _ = self.slot_box(i, 'bar')
        status, progress = bar_progress(
            self.client.original_img[y1, x1:x2 + 1])
        return OfferView(i, side, best_name(name, items), quantity, price,
                         status, progress)

    def read_slots(self, items: Iterable[str]) -> List[OfferView]:
        """All offers. The mouse must not be over the slots: the tooltip it
        brings up covers others' prices."""
        items = list(items)
        views = (self.read_slot(i, items) for i in range(SLOTS))
        return [v for v in views if v is not None]

    def _ocr(self, hsv: numpy.ndarray, mode: int = 7) -> str:
        ocr = self.client.ocr
        if ocr is None:
            return ''
        mask = text_mask(hsv)
        big = cv2.resize(numpy.where(mask, 0, 255).astype(numpy.uint8),
                         None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        big = cv2.copyMakeBorder(big, 10, 10, 10, 10, cv2.BORDER_CONSTANT,
                                 value=255)
        ocr.SetImage(Image.fromarray(big))
        ocr.SetPageSegMode(mode)
        return ' '.join(ocr.GetUTF8Text().split())

    def main_box(self, part: str) -> Box:
        return self.local_box(MAIN_PARTS[part])

    def read_number(self, part: str) -> Optional[int]:
        """quantity / price (the set-up screen's small orange font)."""
        mask = text_mask(self._crop(self.setup_box(part), hsv=True))
        return read_digits(mask)

    def read_text(self, part: str) -> str:
        """OCR of a set-up screen part (type, name, info)."""
        return self._ocr(self._crop(self.setup_box(part), hsv=True))

    def read_status_text(self, part: str) -> str:
        return self._ocr(self._crop(self.status_box(part), hsv=True))

    def status_progress(self) -> Tuple[str, float]:
        """(status, share filled) of the offer on the status screen."""
        x1, y1, x2, _ = self.status_box('bar')
        self.client.img  # make sure this frame's images are grabbed
        return bar_progress(self.client.original_img[y1, x1:x2 + 1])

    def abort_button(self) -> bool:
        """Whether the status screen shows its abort button."""
        tpl = template('abort_button')
        if tpl is None or self.screen != 'status':
            return False
        x1, y1, x2, y2 = self.status_box('abort')
        crop = self._crop((x1 - 2, y1 - 2, x2 + 2, y2 + 2))
        if crop.shape[0] < tpl.shape[0] or crop.shape[1] < tpl.shape[1]:
            return False
        score = cv2.matchTemplate(crop, tpl, cv2.TM_CCOEFF_NORMED).max()
        return float(score) >= SLOT_THRESHOLD

    def collect_button(self) -> bool:
        """Whether the main screen offers to collect anything."""
        text = self._ocr(self._crop(self.main_box('collect'), hsv=True))
        return similar(text, 'Collect') >= 0.7

    def find_prompt(self, kind: str, what: str
                    ) -> Optional[Tuple[int, int]]:
        """Where a chat-box prompt's question is (client image
        coordinates), or None if it isn't showing or there's no template
        for it yet."""
        tpl = template(PROMPTS[(kind, what)])
        img = self.client.img
        if tpl is None or img is None:
            return None
        h, w = img.shape[:2]
        # the chat box: bottom left
        y0 = h // 2
        region = img[y0:, :w // 2]
        m = cv2.matchTemplate(region, tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, (x, y) = cv2.minMaxLoc(m)
        if score < PROMPT_THRESHOLD:
            return None
        return x, y0 + y

    def search_results(self, prompt: Tuple[int, int]
                       ) -> List[Tuple[str, Box]]:
        """Item search results under the search prompt: (OCR text, row
        box) per row. Rows are dark text on the chat box's parchment."""
        ocr = self.client.ocr
        px, py = prompt
        img = self.client.img
        # (the question is centred, so it moves as text is typed)
        x1, x2 = max(0, px - 150), min(img.shape[1] - 1, px + 400)
        y1, y2 = py + 26, min(img.shape[0] - 1, py + 130)
        area = img[y1:y2, x1:x2]
        dark = area < 60
        # the chat box's border and scroll bar are dark in nearly every
        # row; without them, rows of text separate
        dark[:, dark.mean(axis=0) > 0.5] = False
        rows = dark.any(axis=1)
        out = []
        i = 0
        while i < len(rows):
            if rows[i]:
                j = i
                while j < len(rows) and rows[j]:
                    j += 1
                if j - i >= 6 and ocr is not None:
                    crop = numpy.where(dark[max(0, i - 2):j + 2], 0, 255)
                    big = cv2.resize(crop.astype(numpy.uint8), None, fx=3,
                                     fy=3, interpolation=cv2.INTER_NEAREST)
                    ocr.SetImage(Image.fromarray(big))
                    ocr.SetPageSegMode(7)
                    text = ocr.GetUTF8Text().strip()
                    # click on the name itself: the widest cluster of
                    # dark columns in the row (not the chat box's border)
                    cols = numpy.where(dark[i:j].any(axis=0))[0]
                    groups, start, prev = [], cols[0], cols[0]
                    for c in cols[1:]:
                        if c > prev + 8:
                            groups.append((start, prev))
                            start = c
                        prev = c
                    groups.append((start, prev))
                    # several results can share a row: each on its own too
                    for a, b in groups:
                        if len(groups) > 1 and b - a >= 20:
                            part = cv2.copyMakeBorder(
                                crop[:, a:b + 1].astype(numpy.uint8), 4, 4,
                                6, 6, cv2.BORDER_CONSTANT, value=255)
                            ocr.SetImage(Image.fromarray(cv2.resize(
                                part, None, fx=3, fy=3,
                                interpolation=cv2.INTER_NEAREST)))
                            ocr.SetPageSegMode(7)
                            out.append((ocr.GetUTF8Text().strip(),
                                        (x1 + a, y1 + i - 1, x1 + b,
                                         y1 + j)))
                    a, b = max(groups, key=lambda g: g[1] - g[0])
                    out.append((text, (x1 + a, y1 + i - 1,
                                       x1 + b, y1 + j)))
                i = j
            else:
                i += 1
        return out


STEP_SECONDS = 2.5
TYPE_SETTLE = 1.2
TYPE_FIRST = 3
"""Letters of an item's last words typed before looking for it in the
suggestions: "grimy can"."""
LOOK_SETTLE = 0.7
FOUND, CLEAR = 0.88, 0.08
"""It's found once a suggestion is this similar, and this much more than any
other one."""
MAX_TRIES = 2


class PlaceOffer:
    """Set up (and with ``confirm`` place) a buy or sell offer, checking
    each step. For a sell offer pass the inventory icon of the item
    (clicking it with the GE open starts a sell offer for it).

    ``quantity`` None (sell only): all of it, at most ``most``.

    :meth:`sell_by_name` sells whatever an inventory icon turns out to be:
    the name on the set-up screen picks the order."""

    def __init__(self, app, kind: str, item: Optional[str],
                 quantity: Optional[int], price: Optional[int],
                 confirm: bool = False, slot: Optional[int] = None,
                 inventory_icon=None, most: Optional[int] = None):
        assert kind in ('buy', 'sell')
        if kind == 'sell' and inventory_icon is None:
            raise ValueError('a sell offer needs the inventory icon')
        if kind == 'buy' and None in (item, quantity, price):
            raise ValueError('a buy offer needs item, quantity and price')
        self.app = app
        self.client = app.client
        self.ge = GrandExchange(self.client)
        self.kind = kind
        self.item = item
        self.quantity = None if quantity is None else int(quantity)
        self.price = None if price is None else int(price)
        self.most = most
        self.orders: Optional[Dict[str, Tuple[Optional[int], int]]] = None
        self.names: List[str] = []
        self.last_name: Optional[str] = None
        self.last_quantity: Optional[int] = None
        self.skipped = False
        """Sell by name: there's no order for the item, went back."""
        self.confirm = confirm
        self.slot = slot
        self.icon = inventory_icon
        self.state = 'start'
        self.since = time.time()
        self.tries: Dict[str, int] = {}
        self.typed = 0

    @classmethod
    def sell_by_name(cls, app, inventory_icon,
                     orders: Dict[str, Tuple[Optional[int], int]],
                     names: Iterable[str], confirm: bool = False,
                     most: Optional[int] = None) -> 'PlaceOffer':
        """Sell the item behind ``inventory_icon``. ``orders``: item name
        -> (quantity or None for all, price). ``names``: every name it
        could be, ordered or not (so a near miss isn't taken for another
        item). Without an order it goes back: ``skipped`` is then set and
        :meth:`step` returns True."""
        offer = cls(app, 'sell', None, None, None, confirm=confirm,
                    inventory_icon=inventory_icon, most=most)
        offer.orders = dict(orders)
        offer.names = sorted(set(names) | set(orders))
        return offer

    @property
    def log(self):
        return self.client.logger

    def _next(self, state: str):
        # breathing space before the next action; timeouts run from its end
        self.state, self.since = state, time.time() + pause_seconds()

    def _waited(self) -> float:
        return time.time() - self.since

    def _try(self, what: str) -> bool:
        """Count an attempt; False once ``MAX_TRIES`` are used up."""
        n = self.tries.get(what, 0) + 1
        self.tries[what] = n
        return n <= MAX_TRIES

    def _fail(self, message: str) -> bool:
        self.log.warning(f'GE {self.kind} {self.item!r}: expected {message}')
        return False

    def _pick_order(self, text: str) -> Optional[bool]:
        """Sell by name: the order for the name on the screen, once it
        has read the same twice."""
        name = best_name(text, self.names, 0.85)
        again, self.last_name = name == self.last_name, name
        if name is None or not again:
            if self._waited() > STEP_SECONDS:
                return self._fail(f'one of {self.names}, the screen says '
                                  f'{text!r}')
            return None
        if name not in self.orders:
            self.log.info(f'GE: no order to sell {name!r}')
            self.skipped = True
            self.ge.click(self.ge.setup_box('back'))
            self._next('backing_out')
            return None
        self.item = name
        self.quantity, self.price = self.orders[name]
        self.log.info(f'GE selling {name!r} ({text!r}): {self.quantity} '
                      f'at {self.price}')
        self._next('quantity')
        return None

    def _all_of_it(self, current: Optional[int]) -> Optional[bool]:
        """Quantity None: all of it, which is what a sell offer starts with
        (user), once the number reads the same twice; ``most`` if that's
        less."""
        again, self.last_quantity = (current == self.last_quantity,
                                     current)
        if not current or not again:
            if self._waited() > STEP_SECONDS:
                return self._fail(f'a quantity, the screen says {current}')
            return None
        if self.most is not None and current > self.most:
            self.quantity = self.most
            self._next('quantity')
            return None
        self.quantity = current
        self._next('price')
        return None

    def step(self) -> Optional[bool]:
        if time.time() < self.since:
            return None
        ge = self.ge
        ge.update()
        s = self.state

        if s == 'start':
            if ge.screen == 'setup':
                return self._fail('the offer slots, not an offer already '
                                  'being set up')
            if ge.screen != 'main':
                return self._fail('the Grand Exchange to be open')
            empty = ge.empty_slots()
            if self.slot is None:
                if not empty:
                    return self._fail('an empty offer slot')
                self.slot = empty[0]
            elif self.slot not in empty:
                return self._fail(f'slot {self.slot} to be empty')
            if not self._try('open'):
                return self._fail('the offer set-up to open')
            if self.kind == 'buy':
                ge.click(ge.slot_box(self.slot, 'buy'))
            else:
                x1, y1, x2, y2 = self.icon.get_bbox()
                self.client.screen.click_aoi(x1 + 4, y1 + 4, x2 - 4, y2 - 4,
                                             pause_before_click=True)
            self._next('opening')
            return None

        if s == 'opening':
            if ge.screen == 'setup':
                kind = ge.read_text('type')
                if similar(kind, f'{self.kind} offer') < 0.8:
                    return self._fail(f'a {self.kind} offer, the screen '
                                      f'says {kind!r}')
                self._next('search' if self.kind == 'buy' else 'check_item')
            elif self._waited() > STEP_SECONDS:
                self._next('start')
            return None

        if s == 'search':
            prompt = ge.find_prompt('buy', 'search')
            if prompt is None:
                if self._waited() > STEP_SECONDS:
                    return self._fail('the item search prompt')
                return None
            # no more than it takes to find it in the suggestions
            text = self.item.lower()
            start = text.find(' ') + 1 if ' ' in text else 0
            self.typed = min(len(text), start + TYPE_FIRST)
            type_slowly(text[:self.typed])
            self.state, self.since = 'typing', time.time()
            return None

        if s == 'typing':
            if self._waited() < LOOK_SETTLE:
                return None
            prompt = ge.find_prompt('buy', 'search')
            if prompt is not None:
                scored = sorted(((similar(t, self.item), t, box) for t, box
                                 in ge.search_results(prompt)), reverse=True)
                if scored and scored[0][0] >= FOUND and (
                        len(scored) == 1 or
                        scored[0][0] - scored[1][0] >= CLEAR):
                    score, text, box = scored[0]
                    self.log.info(f'GE search result {text!r} ({score:.2f}) '
                                  f'after {self.item[:self.typed]!r}')
                    ge.click(box, pad=3)
                    self._next('check_item')
                    return None
            if self.typed >= len(self.item):
                self._next('results')
                return None
            type_slowly(self.item.lower()[self.typed])
            self.typed += 1
            self.since = time.time()
            return None

        if s == 'results':
            if self._waited() < TYPE_SETTLE:
                return None
            prompt = ge.find_prompt('buy', 'search')
            if prompt is None:
                return self._fail('the item search prompt to stay open')
            rows = ge.search_results(prompt)
            scored = sorted(((similar(t, self.item), t, box)
                             for t, box in rows), reverse=True)
            # OCR reads K as "Fu"; the set-up screen's name is checked next
            if not scored or scored[0][0] < 0.75:
                if self._waited() > STEP_SECONDS + TYPE_SETTLE:
                    return self._fail(f'{self.item!r} in the search '
                                      f'results, got '
                                      f'{[t for t, _ in rows]}')
                return None
            score, text, box = scored[0]
            self.log.info(f'GE search result {text!r} ({score:.2f})')
            ge.click(box, pad=3)
            self._next('check_item')
            return None

        if s == 'check_item':
            name = ge.read_text('name')
            if self.item is None:
                return self._pick_order(name)
            if similar(name, self.item) >= 0.85:
                self.log.info(f'GE item {name!r}')
                self._next('quantity')
            elif self._waited() > STEP_SECONDS:
                return self._fail(f'the item to be {self.item!r}, '
                                  f'the screen says {name!r}')
            return None

        if s in ('quantity', 'price'):
            current = ge.read_number(s)
            if s == 'quantity' and self.quantity is None:
                return self._all_of_it(current)
            want = self.quantity if s == 'quantity' else self.price
            if current == want:
                self._next('price' if s == 'quantity' else 'ready')
                return None
            if not self._try(s):
                return self._fail(f'{s} {want}, the screen says {current}')
            ge.click(ge.setup_box(f'{s}_custom'))
            self._next(f'{s}_prompt')
            return None

        if s in ('quantity_prompt', 'price_prompt'):
            what = s.split('_')[0]
            if ge.find_prompt(self.kind, what) is None:
                if self._waited() > STEP_SECONDS:
                    return self._fail(f'the {what} prompt (template '
                                      f'{PROMPTS[(self.kind, what)]})')
                return None
            value = self.quantity if what == 'quantity' else self.price
            type_slowly(short_amount(value))
            time.sleep(random.uniform(0.4, 1.1))
            keyboard.press_and_release('enter')
            self._next(f'{what}_typed')
            return None

        if s in ('quantity_typed', 'price_typed'):
            what = s.split('_')[0]
            want = self.quantity if what == 'quantity' else self.price
            if ge.read_number(what) == want:
                self._next('price' if what == 'quantity' else 'ready')
            elif self._waited() > STEP_SECONDS:
                self._next(what)  # read again / one more try
            return None

        if s == 'ready':
            self.log.info(f'GE {self.kind} offer set up: {self.item!r} x '
                          f'{self.quantity} at {self.price} (slot '
                          f'{self.slot})')
            if not self.confirm:
                ge.click(ge.setup_box('back'))
                self._next('backing_out')
                return None
            ge.click(ge.setup_box('confirm'))
            self._next('confirming')
            return None

        if s == 'backing_out':
            if ge.screen == 'main':
                self.log.info('GE: went back without confirming')
                return True
            if self._waited() > STEP_SECONDS:
                return self._fail('the offer slots after going back')
            return None

        if s == 'confirming':
            if ge.screen == 'main' and not ge.slot_empty(self.slot):
                self.log.info(f'GE {self.kind} offer placed in slot '
                              f'{self.slot}')
                return True
            if self._waited() > STEP_SECONDS + 1:
                return self._fail(f'the offer in slot {self.slot} after '
                                  f'confirming (screen {ge.screen})')
            return None

        return None


class _Steps:
    """What the steps on existing offers share."""

    what = ''

    def __init__(self, app):
        self.client = app.client
        self.ge = GrandExchange(self.client)
        self.state = 'start'
        self.since = time.time()
        self.clicks = 0

    @property
    def log(self):
        return self.client.logger

    def _next(self, state: str):
        self.state, self.since = state, time.time() + pause_seconds()

    def _waited(self) -> float:
        return time.time() - self.since

    def _fail(self, message: str) -> bool:
        self.log.warning(f'GE {self.what}: expected {message}')
        return False


class AbortOffer(_Steps):
    """Abort the offer in a slot, checking it's the expected one (``side`` and
    ``item``) on the main screen and again on its own screen, and that its bar
    turns red. Ends on the main screen; what's left is to collect.

    ``status`` is then 'cancelled', or 'complete' if it filled first.
    """

    def __init__(self, app, slot: int, side: str, item: str,
                 names: Iterable[str] = ()):
        super().__init__(app)
        self.slot, self.side, self.item = slot, side, item
        self.names = sorted(set(names) | {item})
        self.status: Optional[str] = None
        self.what = f'abort {side} {item!r}'

    def step(self) -> Optional[bool]:
        if time.time() < self.since:
            return None
        ge, s = self.ge, self.state
        ge.update()

        if s == 'start':
            if ge.screen != 'main':
                if self._waited() > STEP_SECONDS:
                    return self._fail(f'the offer slots (screen {ge.screen})')
                return None
            view = ge.read_slot(self.slot, self.names)
            if view is None or (view.side, view.item) != (self.side,
                                                          self.item):
                if self._waited() > STEP_SECONDS:
                    return self._fail(f'that offer in slot {self.slot}, '
                                      f'got {view}')
                return None
            if view.status != 'active':
                self.status = view.status
                self.log.info(f'GE {self.what}: already {view.status}')
                return True
            if self.clicks >= MAX_TRIES:
                return self._fail('its status screen')
            self.clicks += 1
            ge.click(ge.slot_box(self.slot, 'name'))
            self._next('opening')
            return None

        if s == 'opening':
            if ge.screen == 'status':
                side = ge.read_status_text('type')
                name = ge.read_status_text('name')
                if similar(side, f'{self.side} offer') < 0.8 or \
                        best_name(name, self.names, 0.85) != self.item:
                    return self._fail(f'that offer, the screen says '
                                      f'{side!r} {name!r}')
                self.clicks = 0
                self._next('abort')
            elif self._waited() > STEP_SECONDS:
                self._next('start')
            return None

        if s in ('abort', 'aborting'):
            status = ge.status_progress()[0] if ge.screen == 'status' \
                else None
            if status in ('cancelled', 'complete'):
                self.status = status
                ge.click(ge.status_box('back'))
                self._next('back')
                return None
            if s == 'abort':
                if not ge.abort_button():
                    if self._waited() > STEP_SECONDS:
                        return self._fail('the abort button')
                    return None
                if self.clicks >= MAX_TRIES:
                    return self._fail('the offer to be cancelled')
                self.clicks += 1
                ge.click(ge.status_box('abort'))
                self._next('aborting')
            elif self._waited() > 2 * STEP_SECONDS:
                self._next('abort')
            return None

        if s == 'back':
            if ge.screen == 'main':
                self.log.info(f'GE {self.what}: {self.status}')
                return True
            if self._waited() > STEP_SECONDS:
                return self._fail('the offer slots after going back')
        return None


class CollectOffers(_Steps):
    """Collect everything there is to the inventory (bought items come as
    notes). Expectation: the Collect button goes; if it stays (inventory
    full?) it's tried once more. Nothing to collect is a success."""

    what = 'collect'

    def step(self) -> Optional[bool]:
        if time.time() < self.since:
            return None
        ge = self.ge
        if ge.update() != 'main':
            if self._waited() > STEP_SECONDS:
                return self._fail(f'the offer slots (screen {ge.screen})')
            return None
        button = ge.collect_button()
        if self.state == 'start':
            if not button:
                if self._waited() < TYPE_SETTLE:
                    return None
                self.log.info('GE collect: '
                              + ('done' if self.clicks else 'nothing'))
                return True
            if self.clicks >= MAX_TRIES:
                return self._fail('the Collect button to go (inventory '
                                  'full?)')
            self.clicks += 1
            ge.click(ge.main_box('collect'))
            self._next('collecting')
        elif not button or self._waited() > STEP_SECONDS:
            self._next('start')
        return None
