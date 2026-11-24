"""Read an item's quantity from its bank slot (the number in the slot's top
left corner).

The text's colour gives its unit: yellow is exact, white is thousands
("135K"), green is millions ("992M"); those two are rounded down by the
game. A placeholder shows a dimmed "0". The digits are the same font as the
Grand Exchange's (``data/ge/digits``), matched glyph by glyph. Anything that
doesn't match completely (e.g. partly covered) isn't read at all.
"""
from typing import Dict, Optional

import numpy

from ...grand_exchange import columns, glyphs

TEXT_BOX = (0, -2, 46, 10)
"""Where the text can be, relative to the slot's top left (inclusive)."""
GLYPH_ROWS = slice(5, 13)
"""The rows of the GE digit templates the digits are in."""
HEIGHT = 8

YELLOW, WHITE, GREEN = (0, 255, 255), (255, 255, 255), (128, 255, 0)   # BGR
DIM_YELLOW = ((10, 150, 150), (40, 180, 180))
"""A placeholder's text, seen as about (24, 165, 169)."""

_K = ('#...', '#..#', '#.#.', '##..', '##..', '#.#.', '#..#', '#...#')
_M = ('#....#', '##..##', '#.##.#') + ('#....#',) * 5


def _glyph(rows) -> numpy.ndarray:
    width = max(map(len, rows))
    return numpy.array([[c == '#' for c in r.ljust(width, '.')]
                        for r in rows])


K, M = _glyph(_K), _glyph(_M)
_digits: Dict[str, numpy.ndarray] = {}


def digits() -> Dict[str, numpy.ndarray]:
    if not _digits:
        _digits.update({ch: g[GLYPH_ROWS] for ch, g in glyphs().items()
                        if ch.isdigit()})
    return _digits


def read_text(mask: numpy.ndarray, font: Dict[str, numpy.ndarray]
              ) -> Optional[str]:
    """All of the text in a mask, None unless every glyph matches."""
    rows = numpy.where(mask.any(axis=1))[0]
    if not len(rows) or rows[-1] - rows[0] + 1 != HEIGHT:
        return None
    mask = mask[rows[0]:rows[-1] + 1]
    text, end = '', None
    for a, b in columns(mask):
        if end is not None and a - end > 4:
            return None     # a gap: something in the middle is covered
        end = b
        seg = mask[:, a:b]
        for ch, g in font.items():
            if g.shape == seg.shape and numpy.array_equal(g, seg):
                text += ch
                break
        else:
            return None
    return text


def read_quantity(bgr: numpy.ndarray,
                  font: Optional[Dict[str, numpy.ndarray]] = None
                  ) -> Optional[int]:
    """The quantity in a slot's :data:`TEXT_BOX` image (BGR or BGRA). None:
    no text (a single item, or nothing) or not readable."""
    font = dict(font if font is not None else digits())
    bgr = bgr[:, :, :3]
    found = []
    for colour, unit, suffix in ((YELLOW, 1, None), (WHITE, 1_000, K),
                                 (GREEN, 1_000_000, M)):
        mask = (bgr == colour).all(axis=2)
        if mask.sum() < 4:
            continue
        glyph_set = dict(font)
        if suffix is not None:
            glyph_set['x'] = suffix
        text = read_text(mask, glyph_set)
        if text is None:
            return None
        if suffix is not None:
            if not text.endswith('x'):
                return None
            text = text[:-1]
        if not text.isdigit():
            return None
        found.append(int(text) * unit)
    if len(found) == 1:
        return found[0]
    if found:
        return None
    lo, hi = DIM_YELLOW
    dim = ((bgr >= lo) & (bgr <= hi)).all(axis=2)
    # dimmed text blends with what's behind it, but stays a "0"
    zero = font.get('0')
    if zero is not None and int(dim.sum()) == int(zero.sum()) and \
            read_text(dim, {'0': zero}) == '0':
        return 0
    return None
