"""Check and fix the camera before a bot starts.

The engine assumes the camera is facing north, fully top-down and at the
default zoom. After logging in the camera is usually angled, so:

- **North**: the compass by the minimap must show N at the top. If it
  doesn't, the compass is clicked (which snaps the camera north). The compass
  is found by its north-facing template, or, when rotated, at a fixed offset
  from the XP button next to it.
- **Top-down**: the player's (cyan) current-tile highlight is a square seen
  from straight above, and a wide trapezoid when angled. Uneven terrain can
  skew it slightly even when vertical, so only a perfect square at the
  expected size skips the fix; otherwise the Up arrow is held until the tile
  stops changing (RuneLite's camera plugin allows a fully vertical pitch,
  and pitch stops at its maximum, so this is a short tap if already there).
- **Zoom**: at default zoom the tile is a known size for the window height.
  Only judged on a clean square (terrain changes the size). The zoom can't
  be set from here, so a mismatch is only warned about.

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
# Exact colour only: a near-cyan skybox (#00FFFA) must not count.
TILE_LOWER = (253, 253, 0, 0)
TILE_UPPER = (255, 255, 2, 255)

# Tile side (px) at default zoom, fitted against window height:
# 51-52px @760, 61-62px @900, 81px @1168, 97px @1408 (residuals < 0.6px).
ZOOM_FIT = (0.07048, -1.89)
ZOOM_TOLERANCE = 0.025
TOP_DOWN_TOLERANCE = 0.15


@dataclass
class Tile:
    x: int
    y: int
    w: int
    h: int
    top_width: int
    bottom_width: int

    @property
    def perfect(self) -> bool:
        """Exactly square: fully vertical camera on flat ground."""
        return (abs(self.w - self.h) <= 1
                and abs(self.top_width - self.bottom_width) <= 1)

    @property
    def distortion(self) -> float:
        """How far from square, relative (0 = perfect square)."""
        edges = max(self.top_width, self.bottom_width, 1)
        return max(abs(self.w - self.h) / max(self.w, self.h, 1),
                   abs(self.top_width - self.bottom_width) / edges)

    @property
    def top_down(self) -> bool:
        """Roughly square. Uneven terrain skews the tile a little even when
        the camera is straight down, so this is lenient; an angled camera
        is far off (e.g. 185x78 after logging in)."""
        return self.distortion <= TOP_DOWN_TOLERANCE


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
    # only judge zoom on a clean square; on uneven ground the tile's size
    # shifts with terrain height, so leave it unknown
    if tile and tile.perfect:
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

    # 2. top-down. Only a perfect square at the expected size proves the
    # camera is fully vertical with nothing to do. Anything else (clearly
    # angled, very slightly tilted, or skewed by uneven terrain) gets Up
    # held until the tile stops changing: pitch stops at its maximum, so if
    # the camera was already vertical that's just a short tap, and whatever
    # skew is left is down to the terrain.
    bgra = _grab(handle)[0]
    tile = find_player_tile(bgra)
    expected = expected_tile_size(bgra.shape[0])
    if tile is None:
        log.warning('Camera: player tile highlight not found; check the Tile '
                    'Indicators plugin highlights the current tile in cyan')
    elif not tile.perfect or abs(tile.w - expected) > 1.5:
        log.info(f'Camera: tile {tile.w}x{tile.h} '
                 f'({"angled" if not tile.top_down else "not a clean square"})'
                 ', holding Up until it stops changing')
        end = time.time() + tilt_timeout
        keyboard.press('up')
        try:
            last, stable = None, 0
            while time.time() < end:
                time.sleep(0.15)
                tile = find_player_tile(_grab(handle)[0])
                shape = ((tile.w, tile.h, tile.top_width, tile.bottom_width)
                         if tile else None)
                stable = stable + 1 if shape and shape == last else 0
                last = shape
                if stable >= 3:
                    break
        finally:
            keyboard.release('up')
        time.sleep(0.3)  # let the camera settle after releasing
        tile = find_player_tile(_grab(handle)[0])
        if tile and tile.top_down and not tile.perfect:
            log.info(f'Camera: at maximum pitch; tile {tile.w}x{tile.h} is '
                     'skewed, probably uneven ground')

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
