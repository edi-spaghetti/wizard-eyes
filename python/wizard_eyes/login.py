"""Launch OSRS through the Jagex Launcher and log in, for a given character.

Flow for :func:`ensure_logged_in`:

1. If a RuneLite window for the character is already open, use it.
2. Otherwise make sure the Jagex Launcher is open on the Old School page with
   the requested character selected, and click Play.
3. Wait for the new RuneLite process and confirm the launcher handed it the
   right character (``JX_DISPLAY_NAME``).
4. Wait for the client to load, then click through the login screen
   ("Play Now") and lobby ("Click here to play") until the game is showing.

Which character the launcher has selected is checked cheaply where possible:
the launcher stores the last-played character's id in its settings.json, and
once a username has been seen with an id it is cached in
``config/accounts.local.json``. If the cache can't vouch for the selection,
the character name is read from the launcher window with OCR.

Only ``JX_DISPLAY_NAME`` and ``JX_CHARACTER_ID`` are ever read from the
RuneLite process; its session tokens are never touched.
"""
import json
import logging
import os
import random
import subprocess
import time
from dataclasses import dataclass
from os.path import exists, join
from typing import Dict, Optional, Tuple

import cv2
import numpy
import psutil
import pyautogui
import win32gui

from .file_path_utils import ensure_local, get_root
from .window import ClientWindow, find_client_windows, focus_window

logger = logging.getLogger('wizard_eyes.login')

LAUNCHER_EXE_DEFAULT = r'C:\Program Files (x86)\Jagex Launcher\JagexLauncher.exe'
LAUNCHER_TITLE = 'Jagex Launcher'
MATCH_THRESHOLD = 0.95

TEMPLATES = {
    'launcher_play': 'launcher/play.npy',
    'launcher_character': 'launcher/character.npy',
    'launcher_osrs_logo': 'launcher/osrs_logo.npy',
    'login_play_now': 'login/play_now.npy',
    'login_welcome': 'login/welcome.npy',
    'login_connecting': 'login/connecting.npy',
    'lobby_click_to_play': 'login/click_to_play.npy',
    # existing engine assets, only visible once in game
    'ingame_logout_tab': 'tabs/logout.npy',
    'ingame_inventory_tab': 'tabs/inventory.npy',
}

# Character name box, relative to the top-left of the "Character" label.
NAME_BOX_OFFSET = (15, 42, 220, 77)


class LoginError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

_template_cache: Dict[str, numpy.ndarray] = {}


def template(name: str) -> numpy.ndarray:
    if name not in _template_cache:
        path = join(get_root(), 'data', *TEMPLATES[name].split('/'))
        ensure_local(path)
        if not exists(path):
            raise LoginError(f'Missing template {TEMPLATES[name]}; '
                             'check remote assets are configured')
        _template_cache[name] = numpy.load(path)
    return _template_cache[name]


def grab_window(handle: int) -> Tuple[numpy.ndarray, Tuple[int, int]]:
    """Grayscale image of a window, plus its top-left screen position."""
    from .screen_tools import Screen  # sets DPI awareness on import
    x1, y1, x2, y2 = win32gui.GetWindowRect(handle)
    img = Screen(None).grab_screen(x1, y1, x2 - 1, y2 - 1)
    return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY), (x1, y1)


@dataclass
class Match:
    score: float
    x: int  # top-left, image coordinates
    y: int
    w: int
    h: int

    def screen_box(self, origin):
        ox, oy = origin
        return (ox + self.x, oy + self.y,
                ox + self.x + self.w - 1, oy + self.y + self.h - 1)


def find(img: numpy.ndarray, name: str) -> Optional[Match]:
    t = template(name)
    if img.shape[0] < t.shape[0] or img.shape[1] < t.shape[1]:
        return None
    res = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED)
    _, score, _, (x, y) = cv2.minMaxLoc(res)
    if score < MATCH_THRESHOLD:
        return None
    return Match(score, x, y, t.shape[1], t.shape[0])


def click_box(x1, y1, x2, y2):
    """Click somewhere near the middle of a screen-space box."""
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x = int(min(max(random.gauss(cx, (x2 - x1) / 8), x1), x2))
    y = int(min(max(random.gauss(cy, (y2 - y1) / 8), y1), y2))
    pyautogui.moveTo(x, y, duration=random.uniform(0.15, 0.4))
    time.sleep(random.uniform(0.05, 0.15))
    pyautogui.mouseDown()
    time.sleep(random.uniform(0.06, 0.14))
    pyautogui.mouseUp()


# --------------------------------------------------------------------------
# launcher state
# --------------------------------------------------------------------------

def launcher_settings_path() -> str:
    return join(os.environ.get('LOCALAPPDATA', ''), 'Jagex Launcher',
                'settings.json')


def launcher_selected_character_id() -> Optional[str]:
    """The character id the launcher last played OSRS with, if known."""
    try:
        with open(launcher_settings_path(), 'r', encoding='utf-8') as f:
            prefs = json.load(f).get('gamePreferences', {})
        return prefs.get('lastGameAccountByGameId', {}).get('osrs')
    except (OSError, ValueError):
        return None


def accounts_cache_path() -> str:
    return join(get_root(), 'config', 'accounts.local.json')


def load_account_cache() -> Dict[str, str]:
    try:
        with open(accounts_cache_path(), 'r', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def remember_account(username: str, character_id: str):
    cache = load_account_cache()
    if cache.get(username.lower()) == character_id:
        return
    cache[username.lower()] = character_id
    with open(accounts_cache_path(), 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2)


def launcher_exe() -> str:
    try:
        import winreg
        key = r'Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\Jagex Launcher'
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as k:
            loc = winreg.QueryValueEx(k, 'InstallLocation')[0]
            path = join(loc, 'JagexLauncher.exe')
            if exists(path):
                return path
    except OSError:
        pass
    return LAUNCHER_EXE_DEFAULT


def launcher_window() -> Optional[int]:
    found = []

    def visit(h, _):
        if (win32gui.IsWindowVisible(h) and not win32gui.GetParent(h)
                and win32gui.GetWindowText(h) == LAUNCHER_TITLE):
            found.append(h)
        return True

    win32gui.EnumWindows(visit, None)
    return found[0] if found else None


def open_launcher(timeout: float) -> int:
    """Start (or surface) the Jagex Launcher and wait for its window."""
    handle = launcher_window()
    if handle is None:
        exe = launcher_exe()
        if not exists(exe):
            raise LoginError(f'Jagex Launcher not found at {exe}; run setup.ps1')
        logger.info('Starting the Jagex Launcher')
        # starting it again also brings a tray-minimised launcher back
        subprocess.Popen([exe], close_fds=True)
        end = time.time() + timeout
        while handle is None and time.time() < end:
            time.sleep(1)
            handle = launcher_window()
        if handle is None:
            raise LoginError('Jagex Launcher window did not appear')
    focus_window(handle)
    return handle


def read_launcher_character(img: numpy.ndarray) -> Optional[str]:
    """OCR the selected character name from a launcher screenshot."""
    label = find(img, 'launcher_character')
    if label is None:
        return None
    dx1, dy1, dx2, dy2 = NAME_BOX_OFFSET
    crop = img[label.y + dy1:label.y + dy2, label.x + dx1:label.x + dx2]
    crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    if crop.mean() < 128:
        crop = 255 - crop
    import tesserocr
    from PIL import Image
    tessdata = join(get_root(), '..', 'tessdata')
    with tesserocr.PyTessBaseAPI(path=tessdata,
                                 psm=tesserocr.PSM.SINGLE_LINE) as api:
        api.SetImage(Image.fromarray(crop))
        return api.GetUTF8Text().strip() or None


# --------------------------------------------------------------------------
# RuneLite processes
# --------------------------------------------------------------------------

def _jx(pid: int) -> Tuple[Optional[str], Optional[str]]:
    """(display name, character id) the launcher gave a RuneLite process."""
    try:
        env = psutil.Process(pid).environ()
    except (psutil.Error, OSError):
        return None, None
    return env.get('JX_DISPLAY_NAME'), env.get('JX_CHARACTER_ID')


def find_runelite_for(username: str) -> Optional[ClientWindow]:
    for w in find_client_windows():
        name, _ = _jx(w.pid)
        if (name or '').lower() == username.lower():
            return w
        # fall back on the window title, which RuneLite sets to the name
        if name is None and username.lower() in w.title.lower():
            return w
    return None


def runelite_pids() -> set:
    return {p.pid for p in psutil.process_iter(['name'])
            if (p.info['name'] or '').lower() == 'runelite.exe'}


# --------------------------------------------------------------------------
# the flow
# --------------------------------------------------------------------------

def launch_from_launcher(username: str, timeout: float = 120) -> ClientWindow:
    handle = open_launcher(timeout=60)

    # wait for the launcher's page to render
    end = time.time() + 60
    img, origin, play = None, None, None
    while time.time() < end:
        img, origin = grab_window(handle)
        play = find(img, 'launcher_play')
        if play and find(img, 'launcher_osrs_logo'):
            break
        time.sleep(1)
    else:
        raise LoginError('Jagex Launcher is not showing the Old School page '
                         'with a Play button; select Old School in the '
                         'launcher and try again')

    # check the selected character
    cached = load_account_cache().get(username.lower())
    selected_id = launcher_selected_character_id()
    if cached and selected_id == cached:
        logger.info(f'Launcher character matches cached id for {username}')
    else:
        shown = read_launcher_character(img)
        logger.info(f'Launcher shows character: {shown!r}')
        if not shown or shown.lower() != username.lower():
            raise LoginError(
                f'Jagex Launcher has {shown!r} selected, not {username!r}. '
                'Select the right character in the launcher and try again.')

    before = runelite_pids()
    logger.info('Clicking Play in the Jagex Launcher')
    focus_window(handle)
    click_box(*play.screen_box(origin))

    # wait for the new RuneLite process and its window
    end = time.time() + timeout
    while time.time() < end:
        time.sleep(2)
        for pid in runelite_pids() - before:
            name, char_id = _jx(pid)
            if name and name.lower() != username.lower():
                raise LoginError(f'Launcher started RuneLite for {name!r}, '
                                 f'not {username!r}')
            windows = [w for w in find_client_windows() if w.pid == pid]
            if name and windows:
                if char_id:
                    remember_account(username, char_id)
                return windows[0]
    raise LoginError('RuneLite did not start from the Jagex Launcher')


def classify(img: numpy.ndarray) -> Tuple[str, Optional[Match]]:
    """Work out which login state a RuneLite screenshot shows."""
    for state, name in (('in_game', 'ingame_logout_tab'),
                        ('in_game', 'ingame_inventory_tab'),
                        ('lobby', 'lobby_click_to_play'),
                        ('login', 'login_play_now'),
                        ('connecting', 'login_connecting')):
        m = find(img, name)
        if m:
            return state, m
    return 'loading', None


def log_in(window: ClientWindow, timeout: float = 300,
           settle: float = 3.0) -> None:
    """Click through the login screen and lobby until in game.

    Polls the client window; nothing is clicked until a known screen is
    recognised, so a slow first launch just means more polling. A login
    button must stay visible for ``settle`` seconds before it's clicked.
    """
    end = time.time() + timeout
    seen_at: Dict[str, float] = {}
    last_click = 0.0
    last_state = None
    while time.time() < end:
        focus_window(window.handle)
        img, origin = grab_window(window.handle)
        state, match = classify(img)
        if state != last_state:
            logger.info(f'Client state: {state}')
            last_state = state
        if state == 'in_game':
            return
        if state in ('login', 'lobby'):
            first = seen_at.setdefault(state, time.time())
            ready = time.time() - first >= settle
            cooled = time.time() - last_click > 5
            if ready and cooled:
                logger.info(f'Clicking {"Play Now" if state == "login" else "Click here to play"}')
                click_box(*match.screen_box(origin))
                last_click = time.time()
        else:
            seen_at.clear()
        time.sleep(1)
    raise LoginError(f'Not logged in after {timeout:.0f}s (last state: {last_state})')


def ensure_logged_in(username: str, timeout: float = 300) -> ClientWindow:
    """Make sure a RuneLite client for ``username`` is open and in game."""
    window = find_runelite_for(username)
    if window:
        logger.info(f'Found RuneLite for {username}: {window.title!r}')
    else:
        window = launch_from_launcher(username)
        logger.info(f'RuneLite started for {username}')
    log_in(window, timeout=timeout)
    logger.info(f'{username} is logged in')
    return window
