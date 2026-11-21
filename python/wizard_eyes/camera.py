"""Check and fix the camera before a bot starts.

The engine assumes the camera is facing north, fully top-down and at the
default zoom. After logging in the camera is usually angled, so:

- **North**: the compass by the minimap must show N at the top. If it
  doesn't, the compass is clicked (which snaps the camera north). The compass
  is found by its north-facing template, or, when rotated, at a fixed offset
  from the XP button next to it.
- **Top-down**: the player's (cyan) current-tile highlight is a perfect
  square when seen from straight above, and a wide trapezoid otherwise. If
  it isn't square, the Up arrow is held until it is (RuneLite's camera plugin
  allows a fully vertical pitch).
- **Zoom**: at default zoom the tile is a known size for the window height.
  The zoom can't be set from here, so a mismatch is only warned about.

Nothing is clicked or pressed if the camera is already right.
"""
import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import keyboard
import numpy
import win32gui

from . import login
from .login import click_box, find, grab_window
from .window import focus_window

logger = logging.getLogger('wizard_eyes.camera')

login.TEMPLATES.update({
    'compass_north': 'gauges/compass/north.npy',
    'compass_xp_button': 'gauges/compass/xp_button.npy',
})

# Resizable - Modern: compass centre relative to the XP button template's
# top-left corner.
COMPASS_FROM_XP = (45, -9)
COMPASS_RADIUS = 9

# Player current-tile highlight colour (BGRA), see Player.TILE_COLOURA.
TILE_LOWER = (250, 250, 0, 0)
TILE_UPPER = (255, 255, 10, 255)

# Tile side (px) at default zoom, fitted against window height:
# 81px @1168, 61px @900, 51px @760.
ZOOM_FIT = (0.0735, -4.9)
ZOOM_TOLERANCE = 0.025


@dataclass
class Tile:
    x: int
    y: int
    w: int
    h: int
    top_width: int
    bottom_width: int

    @property
    def top_down(self) -> bool:
        # Fully vertical is exactly square (e.g. 81x81, equal top/bottom
        # edges); a nearly-vertical camera is a couple of pixels off.
        square = abs(self.w - self.h) <= 1
        no_perspective = abs(self.top_width - self.bottom_width) <= 1
        return square and no_perspective


def find_player_tile(bgra: numpy.ndarray) -> Optional[Tile]:
    """The cyan current-tile outline nearest the middle of the window."""
    mask = cv2.inRange(bgra, TILE_LOWER, TILE_UPPER)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    best, best_dist = None, float('inf')
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw < 15 or ch < 8:
            continue
        # the highlight is a thin outline; skip solid areas of the same
        # colour (e.g. a cyan skybox when the camera is tilted low)
        if mask[y:y + ch, x:x + cw].mean() / 255 > 0.3:
            continue
        dist = abs(x + cw / 2 - w / 2) + abs(y + ch / 2 - h / 2)
        if dist < best_dist:
            best, best_dist = (x, y, cw, ch), dist
    if best is None:
        return None
    x, y, cw, ch = best
    top = numpy.where(mask[y + 1, x:x + cw])[0]
    bottom = numpy.where(mask[y + ch - 2, x:x + cw])[0]
    span = lambda a: int(a.max() - a.min()) if len(a) else 0
    return Tile(x, y, cw, ch, span(top), span(bottom))


def expected_tile_size(window_height: int) -> float:
    a, b = ZOOM_FIT
    return a * window_height + b


def compass_centre(gray: numpy.ndarray) -> Tuple[Optional[Tuple[int, int]], bool]:
    """(centre of the compass, whether it's showing north)."""
    north = find(gray, 'compass_north', (0.5, 0.0, 1.0, 0.3))
    if north:
        return (north.x + north.w // 2, north.y + north.h // 2), True
    xp = find(gray, 'compass_xp_button', (0.5, 0.0, 1.0, 0.3))
    if xp:
        return (xp.x + COMPASS_FROM_XP[0], xp.y + COMPASS_FROM_XP[1]), False
    return None, False


def _grab(handle):
    from .screen_tools import Screen
    x1, y1, x2, y2 = win32gui.GetWindowRect(handle)
    bgra = Screen(None).grab_screen(x1, y1, x2 - 1, y2 - 1)
    return bgra, cv2.cvtColor(bgra, cv2.COLOR_BGRA2GRAY), (x1, y1)


@dataclass
class CameraState:
    north: Optional[bool]
    top_down: Optional[bool]
    zoom_ok: Optional[bool]
    tile: Optional[Tile]
    expected_tile: float

    def summary(self) -> str:
        fmt = lambda v: 'unknown' if v is None else ('yes' if v else 'NO')
        size = f'{self.tile.w}x{self.tile.h}px' if self.tile else 'not found'
        return (f'north: {fmt(self.north)}, top-down: {fmt(self.top_down)}, '
                f'default zoom: {fmt(self.zoom_ok)} (tile {size}, '
                f'expected ~{self.expected_tile:.0f}px)')


def camera_state(handle: int) -> CameraState:
    bgra, gray, _ = _grab(handle)
    _, north = compass_centre(gray)
    compass_found = compass_centre(gray)[0] is not None
    tile = find_player_tile(bgra)
    expected = expected_tile_size(bgra.shape[0])
    top_down = tile.top_down if tile else None
    zoom_ok = None
    if tile and tile.top_down:
        zoom_ok = abs(tile.w - expected) <= ZOOM_TOLERANCE * expected
    return CameraState(north if compass_found else None, top_down, zoom_ok,
                       tile, expected)


def ensure_camera(handle: int, log: Optional[logging.Logger] = None,
                  tilt_timeout: float = 6.0) -> CameraState:
    """Face north and tilt fully top-down if needed; warn about zoom."""
    log = log or logger
    focus_window(handle)

    # 1. north
    for attempt in range(3):
        bgra, gray, origin = _grab(handle)
        centre, north = compass_centre(gray)
        if centre is None:
            log.warning('Camera: compass not found (not in game, or layout '
                        'not Resizable - Modern); skipping camera checks')
            return camera_state(handle)
        if north:
            break
        log.info('Camera: not facing north, clicking the compass')
        cx, cy = centre[0] + origin[0], centre[1] + origin[1]
        r = COMPASS_RADIUS // 2
        click_box(cx - r, cy - r, cx + r, cy + r)
        time.sleep(1.0 + 0.5 * attempt)
    else:
        log.warning('Camera: still not facing north after clicking the compass')

    # 2. top-down. A very slight tilt can still look square but a little too
    # big, so also tilt if the size is off; if it's already at maximum
    # pitch, the hold ends as soon as the size stops changing, and whatever
    # difference is left is down to zoom.
    bgra = _grab(handle)[0]
    tile = find_player_tile(bgra)
    expected = expected_tile_size(bgra.shape[0])
    if tile is None:
        log.warning('Camera: player tile highlight not found; check the Tile '
                    'Indicators plugin highlights the current tile in cyan')
    elif not tile.top_down or abs(tile.w - expected) > 1.5:
        log.info(f'Camera: not top-down (tile {tile.w}x{tile.h}), '
                 'holding Up to tilt')
        # Pitch stops at its maximum, so keep holding until the tile looks
        # square AND has stopped changing size - "nearly" vertical can look
        # square while still a few pixels off.
        end = time.time() + tilt_timeout
        keyboard.press('up')
        try:
            last, stable = None, 0
            while time.time() < end:
                time.sleep(0.15)
                tile = find_player_tile(_grab(handle)[0])
                size = (tile.w, tile.h) if tile else None
                stable = stable + 1 if size and size == last else 0
                last = size
                if tile and tile.top_down and stable >= 3:
                    break
        finally:
            keyboard.release('up')
        time.sleep(0.3)  # let the camera settle after releasing

    state = camera_state(handle)
    if state.top_down is False:
        log.warning('Camera: could not tilt to fully top-down; check '
                    'RuneLite Camera plugin "Vertical camera" is enabled')
    if state.zoom_ok is False:
        log.warning(f'Camera: zoom looks wrong (tile {state.tile.w}px, '
                    f'expected ~{state.expected_tile:.0f}px). Reset to '
                    'default zoom in game settings.')
    log.info(f'Camera: {state.summary()}')
    return state
