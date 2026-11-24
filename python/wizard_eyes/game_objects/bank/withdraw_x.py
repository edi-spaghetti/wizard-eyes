"""Check, before withdrawing, that the bank's X amount is the expected one.

Depositing a custom amount ("Deposit-X") changes what the "X" quantity button
withdraws too. The amount shows in the mouse-over text of a bank item while
the X button is selected ("Withdraw-14 Vial of water"). OCR reads the word but
misreads this font's 4s ("1k"), and the letters jiggle (their spacing changes
about every 20 s). So the number alone is read in two ways that must agree:
OCR limited to digits, and digit by digit with templates, wherever they are
(``data/bank/withdraw_x/<digit>``; only the digits of the expected amounts are
needed: any other is just "different").

    x = WithdrawX(app, 14)
    ...  # with the bank open and the X button selected, every frame:
    result = x.step(icon)   # True: hovering icon, it says Withdraw-14;
                            # None: working on it; False: failed

A wrong amount is set with the icon's right-click "Withdraw-X", which also
withdraws that many of it. Tried twice at most.
"""
import random
import re
import time
from pathlib import Path
from typing import List, Optional

import cv2
import keyboard
import numpy
from PIL import Image

from ...file_path_utils import ensure_local, get_root

WHITE = 150
"""The darkest the text gets: some of its pixels are shaded."""
MAX_TINT = 40
GLYPH_SCORE = 0.85
PROMPT_THRESHOLD = 0.9
GAP = 9
"""More blank columns than this end the number."""
READ_SECONDS = 70.0
"""The letters move about every 20 s: an unreadable text is given three
moves."""
WRONG_READS = 3
"""Different that often before it's changed."""
OCR_WIDTH = 40
"""Columns after the hyphen the number is in."""
TEXT_WIDTH = 125
OCR_SCALE = 2
"""Larger is read worse (a 14 as 1)."""
SETTLE_SECONDS = 0.35
STEP_SECONDS = 3.0
MAX_TRIES = 2

_cache = {}


def load(name: str) -> numpy.ndarray:
    if name not in _cache:
        path = Path(get_root()) / 'data' / 'bank' / 'withdraw_x' / \
            f'{name}.npy'
        ensure_local(str(path))
        _cache[name] = numpy.load(str(path))
    return _cache[name]


def white_mask(bgr: numpy.ndarray) -> numpy.ndarray:
    c = bgr[:, :, :3].astype(int)
    low, high = c.min(axis=2), c.max(axis=2)
    return (low > WHITE) & (high - low < MAX_TINT)


def trim(mask: numpy.ndarray) -> numpy.ndarray:
    rows = numpy.where(mask.any(axis=1))[0]
    cols = numpy.where(mask.any(axis=0))[0]
    if not len(rows):
        return mask[:0, :0]
    return mask[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def same_glyph(a: numpy.ndarray, b: numpy.ndarray) -> bool:
    """Whether two trimmed glyphs are the same, give or take a pixel."""
    if not a.size or abs(a.shape[0] - b.shape[0]) > 1 or \
            abs(a.shape[1] - b.shape[1]) > 1:
        return False
    h, w = max(a.shape[0], b.shape[0]), max(a.shape[1], b.shape[1])
    pads = [numpy.pad(i, ((0, h - i.shape[0]), (0, w - i.shape[1])))
            for i in (a, b)]
    both, either = (pads[0] & pads[1]).sum(), (pads[0] | pads[1]).sum()
    return both / either >= GLYPH_SCORE


def after_hyphen(mask: numpy.ndarray) -> Optional[numpy.ndarray]:
    """The part of a text's mask right of its first hyphen, as high as the
    letters left of it."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(numpy.uint8), connectivity=8)
    boxes = sorted((tuple(int(v) for v in stats[i][:4])
                    for i in range(1, n) if stats[i][4] >= 4))
    tall = [b for b in boxes if b[3] >= 5]
    for x, y, w, h in boxes:
        # (the letters jiggle up and down, too)
        left = [b for b in tall if 0 < x - (b[0] + b[2]) < 8]
        if h <= 3 and 4 <= w <= 8 and any(
                b[1] < y and y + h <= b[1] + b[3] for b in left):
            top = min(b[1] for b in left) - 5
            return mask[max(0, top):y + 8, x + w:]
    return None


def read_amount(mask: numpy.ndarray) -> Optional[str]:
    """The number after the hyphen in the mouse-over text's mask, "?" for
    every other glyph; None if there's no hyphen or nothing after it."""
    area = after_hyphen(mask)
    if area is None:
        return None
    digits = {d: load(d) for d in '14'}
    out, at, end = '', 0, None
    width = area.shape[1]
    while True:
        cols = numpy.where(area[:, at:].any(axis=0))[0]
        if not len(cols):
            break
        at += int(cols[0])
        if end is not None and at - end > GAP or end is None and at > GAP:
            break
        for d, glyph in digits.items():
            w = glyph.shape[1]
            # letters can touch: exactly the glyph's columns
            if at + w <= width and \
                    same_glyph(trim(area[:, at:at + w]), glyph):
                break
        else:
            d, w = '?', 1
            while at + w < width and area[:, at + w].any():
                w += 1
        out += d
        at += w
        end = at
    return out or None


def ocr_amount(ocr, mask: numpy.ndarray) -> Optional[str]:
    """The same read with OCR, limited to digits meanwhile. It makes a number
    out of anything, so it says little by itself."""
    area = after_hyphen(mask)
    if area is None:
        return None
    return _ocr_only(ocr, '0123456789', area[:, :OCR_WIDTH]) or None


def is_withdraw(ocr, mask: numpy.ndarray) -> bool:
    """Whether the mouse-over text starts with "Withdraw" ("Deposit" also has
    a hyphen and a number)."""
    text = _ocr_only(ocr, 'WithdrawDepos-0123456789', mask[:, :TEXT_WIDTH])
    return text.lower().startswith('withd')


def _ocr_only(ocr, letters: str, mask: numpy.ndarray) -> str:
    """OCR limited to ``letters`` (set back afterwards)."""
    name = 'tessedit_char_whitelist'
    before, mode = ocr.GetVariableAsString(name), ocr.GetPageSegMode()
    try:
        ocr.SetVariable(name, letters)
        ocr.SetPageSegMode(7)
        return _ocr_mask(ocr, mask)
    finally:
        ocr.SetVariable(name, before or '')
        ocr.SetPageSegMode(mode)


def _ocr_mask(ocr, mask: numpy.ndarray) -> str:
    black = numpy.where(mask, 0, 255).astype(numpy.uint8)
    big = cv2.resize(black, None, fx=OCR_SCALE, fy=OCR_SCALE,
                     interpolation=cv2.INTER_NEAREST)
    big = cv2.copyMakeBorder(big, 12, 12, 12, 12, cv2.BORDER_CONSTANT,
                             value=255)
    ocr.SetImage(Image.fromarray(big))
    return ocr.GetUTF8Text().strip()


def judge(amount: int, letters: Optional[str], digits: Optional[str]
          ) -> Optional[bool]:
    """Whether two reads say it's ``amount``; None if they disagree."""
    want = str(amount)
    if letters is None or digits is None:
        return None
    if letters == want and digits == want:
        return True
    if letters != want and digits != want:
        return False
    return None


def x_line(lines: List[str]) -> Optional[int]:
    """Index of the Withdraw-X line in an OCR'd menu: it reads like it, and
    is the one above Withdraw-All."""
    for i, line in enumerate(lines[:-1]):
        if re.match(r'(?i)withdraw.?[x%]\W*$', line) and \
                re.match(r'(?i)withdraw.?a[l1i]+\W*$', lines[i + 1]):
            return i
    return None


class WithdrawX:

    def __init__(self, app, amount: int):
        self.app = app
        self.client = app.client
        self.amount = amount
        self.reset()

    def reset(self):
        self.state = 'look'
        self.since: Optional[float] = None
        self.wrong = 0
        self.tries = 0
        self.hover_at = 0.0
        self.icon = None
        self.reads = (None, None)
        self.lines: List[str] = []

    @property
    def log(self):
        return self.client.logger

    def _go(self, state: str):
        self.state, self.since = state, time.time()
        self.wrong = 0

    def _fail(self, message: str) -> bool:
        self.log.error(f'Withdraw-{self.amount}: {message}')
        self.reset()
        return False

    def _says(self) -> Optional[bool]:
        """Whether the text says the amount, None if it can't be read."""
        img = self.client.get_img_at(self.client.mouse_options.get_bbox(),
                                     mode=self.client.BGRA)
        if img is None:
            return None
        mask = white_mask(img)
        if not is_withdraw(self.client.ocr, mask):
            return None
        self.reads = (read_amount(mask), ocr_amount(self.client.ocr, mask))
        return judge(self.amount, *self.reads)

    def step(self, icon) -> Optional[bool]:
        if self.client.ocr is None:
            return self._fail('needs OCR')
        if self.since is None:
            self._go('look')
        if self.state == 'look':
            # (later states keep the icon they started with: it's covered
            # by the menu meanwhile)
            self.icon = icon
        return getattr(self, f'_{self.state}')(time.time())

    def _look(self, now) -> Optional[bool]:
        if not self.icon.is_inside(*self.client.screen.mouse_xy):
            self.client.screen.mouse_to_object(self.icon)
            self.hover_at = now
            return None
        if now - self.hover_at < SETTLE_SECONDS:
            return None
        says = self._says()
        if says:
            if self.tries:
                self.log.info(f'Withdraw-X is now {self.amount}')
            self.reset()
            return True
        if says is False:
            self.wrong += 1
        if self.wrong < WRONG_READS:
            if now - self.since > READ_SECONDS:
                return self._fail(f'not readable (read {self.reads})')
            self.hover_at = now     # read again a moment later
            return None
        if self.tries >= MAX_TRIES:
            return self._fail('still a different amount')
        self.tries += 1
        self.log.info(f'Withdraw-X is {self.reads}, not {self.amount}: '
                      f'setting it')
        if self.app._right_click(self.icon):
            self._go('menu')
        return None

    def _read(self, item) -> str:
        img = self.client.get_img_at(item.get_bbox(), mode=self.client.BGRA)
        return _ocr_mask(self.client.ocr, white_mask(img))

    def _menu(self, now) -> Optional[bool]:
        menu = self.icon.context_menu
        if menu is not None and menu.items and \
                now - self.since > SETTLE_SECONDS:
            self.lines = [self._read(i) for i in menu.items]
            i = x_line(self.lines)
            if i is not None:
                menu.items[i].click(tmin=.6, tmax=.9, pause_before_click=True)
                self._go('prompt')
                return None
        if now - self.since > STEP_SECONDS:
            if menu is not None:
                self.client.screen.mouse_away_object(menu)
            return self._fail(f'no Withdraw-X in the menu: {self.lines}')
        return None

    def _prompt(self, now) -> Optional[bool]:
        tpl = load('prompt_amount')
        img = self.client.img
        h, w = img.shape[:2]
        m = cv2.matchTemplate(img[h // 2:, :w // 2], tpl,
                              cv2.TM_CCOEFF_NORMED)
        if m.max() < PROMPT_THRESHOLD:
            if now - self.since > STEP_SECONDS:
                return self._fail('the amount prompt did not open')
            return None
        for ch in str(self.amount):
            keyboard.write(ch)
            time.sleep(random.uniform(.12, .35))
        time.sleep(random.uniform(.2, .5))
        keyboard.press_and_release('enter')
        self._go('typed')
        return None

    def _typed(self, now) -> Optional[bool]:
        # the withdrawal goes through, then look again
        if now - self.since > 1.5:
            self._go('look')
        return None
