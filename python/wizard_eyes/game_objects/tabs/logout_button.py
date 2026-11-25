"""Telling whether the logout tab (the X by the minimap) is open.

Open, the X is drawn solid; closed, it's blended with the ground behind it.
So with the solid X as the reference, the line through (reference, seen) of
the X's pixels has a slope equal to the X's opacity, whatever the ground: the
ground mostly moves the line up or down (per colour channel: coloured ground,
e.g. lava, moves them differently).
"""
from typing import Tuple

import numpy

OPEN_SLOPE = 0.85
CLOSED_SLOPE = (0.45, 0.75)
MIN_CORRELATION = 0.8


def opacity(img: numpy.ndarray, reference: numpy.ndarray
            ) -> Tuple[float, float]:
    """(slope, correlation) of the X in an image.

    :param img: BGR or BGRA, the size of the reference.
    :param reference: BGRA of the open tab, alpha 255 on the X's pixels.
    """
    mask = reference[..., 3] > 0
    if img.shape[:2] != mask.shape:
        return 0.0, 0.0
    x = reference[..., :3][mask].astype(numpy.float64)
    y = img[..., :3][mask].astype(numpy.float64)
    x -= x.mean(axis=0)
    y -= y.mean(axis=0)
    xx, yy, xy = (x * x).sum(), (y * y).sum(), (x * y).sum()
    if not xx or not yy:
        return 0.0, 0.0
    return float(xy / xx), float(xy / numpy.sqrt(xx * yy))


def state(img: numpy.ndarray, reference: numpy.ndarray) -> str:
    """'logout', 'logout_selected', or '' (no X: e.g. the bank is open)."""
    slope, correlation = opacity(img, reference)
    if correlation < MIN_CORRELATION:
        return ''
    if slope > OPEN_SLOPE:
        return 'logout_selected'
    if CLOSED_SLOPE[0] < slope < CLOSED_SLOPE[1]:
        return 'logout'
    return ''
