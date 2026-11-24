"""Reading coloured interface text: pixels of one text colour, then OCR.

The game's fonts are small: OCR works best on the bare text pixels, doubled
in size. ``letters`` limits what it may read (set back afterwards)."""
import difflib
import re
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy
from PIL import Image

SCALE = 2
MARGIN = 12
DIGITS = '0123456789,'


def colour_mask(bgr: numpy.ndarray, colour: str) -> numpy.ndarray:
    """Pixels of the interface's white / yellow / orange / cyan / purple /
    green / red text."""
    c = bgr[..., :3].astype(int)
    b, g, r = c[..., 0], c[..., 1], c[..., 2]
    if colour == 'white':
        return c.min(axis=2) > 170
    if colour == 'yellow':
        return (r > 170) & (g > 170) & (b < 90)
    if colour == 'orange':
        return (r > 170) & (g > 90) & (g < 175) & (b < 120)
    if colour == 'cyan':
        return (g > 170) & (b > 170) & (r < 90)
    if colour == 'purple':
        return (r > 90) & (b > 90) & (g < 60)
    if colour == 'green':
        return (g > 120) & (r < 80) & (b < 80)
    if colour == 'red':
        return (r > 120) & (g < 60) & (b < 60)
    raise ValueError(colour)


def read(ocr, mask: numpy.ndarray, letters: str = '') -> str:
    """One line of text from its mask ('' if there's nothing in it)."""
    if not mask.any():
        return ''
    name = 'tessedit_char_whitelist'
    before, mode = ocr.GetVariableAsString(name), ocr.GetPageSegMode()
    try:
        ocr.SetVariable(name, letters)
        ocr.SetPageSegMode(7)
        img = numpy.where(mask, 0, 255).astype(numpy.uint8)
        img = cv2.resize(img, None, fx=SCALE, fy=SCALE,
                         interpolation=cv2.INTER_NEAREST)
        img = cv2.copyMakeBorder(img, MARGIN, MARGIN, MARGIN, MARGIN,
                                 cv2.BORDER_CONSTANT, value=255)
        ocr.SetImage(Image.fromarray(img))
        return ' '.join(ocr.GetUTF8Text().split())
    finally:
        ocr.SetVariable(name, before or '')
        ocr.SetPageSegMode(mode)


def lines(mask: numpy.ndarray, least: int = 5) -> List[Tuple[int, int]]:
    """(first, last + 1) rows of each line of text in a mask. Rows up to two
    apart (a dot, a tail) belong together."""
    rows = numpy.where(mask.any(axis=1))[0]
    out: List[List[int]] = []
    for r in rows:
        if out and r - out[-1][1] <= 2:
            out[-1][1] = int(r)
        else:
            out.append([int(r), int(r)])
    return [(a, b + 1) for a, b in out if b + 1 - a >= least]


def alike(a: str, b: str) -> float:
    a = re.sub(r'\W', '', a or '').lower()
    b = re.sub(r'\W', '', b or '').lower()
    return difflib.SequenceMatcher(None, a, b).ratio()


def pick(texts: Iterable[str], wanted: str, least: float = 0.7,
         margin: float = 0.1) -> Optional[int]:
    """Index of the text that is ``wanted``, misread or not; None if none
    is, or two could be."""
    scored = sorted(((alike(t, wanted), i) for i, t in enumerate(texts)),
                    reverse=True)
    if not scored or scored[0][0] < least:
        return None
    if len(scored) > 1 and scored[0][0] - scored[1][0] < margin:
        return None
    return scored[0][1]


def menu_line(texts: List[str], option: str) -> Optional[int]:
    """Index of the only right-click menu line starting with ``option``
    (letters only, so "Hop-to" matches "Hop.to")."""
    want = re.sub(r'\W', '', option).lower()
    found = [i for i, t in enumerate(texts)
             if re.sub(r'\W', '', t).lower() == want]
    return found[0] if len(found) == 1 else None
