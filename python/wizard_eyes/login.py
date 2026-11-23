"""Launch OSRS through the Jagex Launcher and log in, for a given character.

Flow for :func:`ensure_logged_in`:

1. If a RuneLite window for the character is already open, use it.
2. Otherwise make sure the Jagex Launcher is set to RuneLite and open on the
   Old School page with the requested character selected, and click Play.
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
import win32con
import win32gui

from .file_path_utils import ensure_local, get_root
from .window import ClientWindow, find_client_windows, focus_window

logger = logging.getLogger('wizard_eyes.login')

LAUNCHER_EXE_DEFAULT = r'C:\Program Files (x86)\Jagex Launcher\JagexLauncher.exe'
LAUNCHER_TITLE = 'Jagex Launcher'
MATCH_THRESHOLD = 0.95
LAUNCHER_PAGE_TIMEOUT = 75
LAUNCHER_ATTEMPTS = 3
# the launcher's id for RuneLite (the official client is 'osrs_ehc')
RUNELITE_CLIENT_ID = 'osrs_runelite'
# process names of the official Old School client
OFFICIAL_CLIENTS = ('osclient.exe',)

TEMPLATES = {
    'launcher_play': 'launcher/play.npy',
    'launcher_character': 'launcher/character.npy',
    'launcher_osrs_logo': 'launcher/osrs_logo.npy',
    'login_play_now': 'login/play_now.npy',
    'login_welcome': 'login/welcome.npy',
    'login_connecting': 'login/connecting.npy',
    'lobby_click_to_play': 'login/click_to_play.npy',
    # any login-screen dialog with an Ok button (e.g. "You were
    # disconnected from the server."); clicking Ok returns to Play Now
    'login_ok': 'login/ok.npy',
    'login_disconnected': 'login/disconnected.npy',
    # existing engine assets, only visible once in game
    'ingame_logout_tab': 'tabs/logout.npy',
    'ingame_combat_tab': 'tabs/combat.npy',
    'ingame_combat_tab_selected': 'tabs/combat_selected.npy',
    'ingame_inventory_tab': 'tabs/inventory.npy',
    'ingame_inventory_tab_selected': 'tabs/inventory_selected.npy',
    'ingame_equipment_tab': 'tabs/equipment.npy',
    'ingame_prayer_tab': 'tabs/prayer.npy',
    'ingame_emotes_tab': 'tabs/emotes.npy',
    'ingame_music_tab': 'tabs/music.npy',
    'ingame_chat_all': 'chat/all.npy',
    'ingame_chat_all_selected': 'chat/all_selected.npy',
    'ingame_chat_all_hover': 'chat/all_hover.npy',
    'ingame_chat_all_hover_selected': 'chat/all_hover_selected.npy',
}

# In-game indicators and where to look for them (fractions of the window:
# x1, y1, x2, y2). Several are used because some disappear in some
# situations (e.g. the logout X while the bank is open), and searching only
# the strip they live in keeps each check cheap.
BOTTOM_STRIP = (0.0, 0.88, 1.0, 1.0)
INGAME_INDICATORS = (
    ('ingame_chat_all_selected', BOTTOM_STRIP),
    ('ingame_chat_all', BOTTOM_STRIP),
    ('ingame_chat_all_hover_selected', BOTTOM_STRIP),
    ('ingame_chat_all_hover', BOTTOM_STRIP),
    ('ingame_inventory_tab', BOTTOM_STRIP),
    ('ingame_inventory_tab_selected', BOTTOM_STRIP),
    ('ingame_combat_tab', BOTTOM_STRIP),
    ('ingame_logout_tab', (0.6, 0.0, 1.0, 0.25)),
)

# If none of the fast strip searches find anything, these are searched across
# the whole window (fixed layout puts the chat box mid-window, and RuneLite
# plugins or window sizes can move things).
INGAME_FALLBACK = (
    'ingame_chat_all_selected', 'ingame_chat_all',
    'ingame_chat_all_hover_selected', 'ingame_chat_all_hover',
    'ingame_inventory_tab', 'ingame_inventory_tab_selected',
    'ingame_emotes_tab',
)

# Layout is worked out from where tabs sit *relative to each other*, never
# from absolute positions:
#   modern            - upper-row tabs (inventory...) share a row with the
#                       lower-row tabs (emotes, music), all in one strip
#   classic (either)  - the lower-row tabs are stacked well below the upper
#                       row, with the side panel between them
#   fixed vs resizable classic - fixed is a rigid frame, so the chat "All"
#                       button is always the same offset from the inventory
#                       tab; in resizable classic that gap grows with the window
UPPER_TABS = ('ingame_inventory_tab', 'ingame_inventory_tab_selected')
UPPER_TABS_EXTRA = ('ingame_prayer_tab', 'ingame_equipment_tab')
LOWER_TABS = ('ingame_emotes_tab', 'ingame_music_tab')
CHAT_ALL = ('ingame_chat_all_selected', 'ingame_chat_all',
            'ingame_chat_all_hover_selected', 'ingame_chat_all_hover')
SAME_ROW_TOLERANCE = 4
STACKED_RANGE = (150, 450)            # lower row below upper row, px
FIXED_CHAT_OFFSET = (-621, 312)       # chat All relative to inventory tab
FIXED_TOLERANCE = 4

MODERN = 'resizable_modern'
RESIZABLE_CLASSIC = 'resizable_classic'
FIXED_CLASSIC = 'fixed_classic'
UNKNOWN = 'unknown'
LAYOUT_NAMES = {
    MODERN: 'Resizable - Modern',
    RESIZABLE_CLASSIC: 'Resizable - Classic',
    FIXED_CLASSIC: 'Fixed - Classic',
    UNKNOWN: 'unknown',
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


def find(img: numpy.ndarray, name: str,
          region: Optional[Tuple[float, float, float, float]] = None
          ) -> Optional[Match]:
    """Find a template anywhere in ``img`` (or in a fractional region).

    Matching scans every position, so a screen that has shifted since the
    template was captured is still found; the returned coordinates are
    always relative to the full image.
    """
    t = template(name)
    ox = oy = 0
    if region:
        h, w = img.shape[:2]
        ox, oy = int(region[0] * w), int(region[1] * h)
        img = img[oy:int(region[3] * h), ox:int(region[2] * w)]
    if name.startswith(SCALED_PREFIXES):
        found = _find_scaled(img, t)
        if found is None:
            return None
        score, x, y, w, h = found
        return Match(score, ox + x, oy + y, w, h)
    m = _match(img, t)
    if m is None or m[0] < MATCH_THRESHOLD:
        return None
    score, x, y = m
    return Match(score, ox + x, oy + y, t.shape[1], t.shape[0])


SCALED_PREFIXES = ('login_', 'lobby_')
"""The login screen and lobby are stretched to fit the client window (2x
on a 1936x1168 window), so their templates are searched at any scale."""
SCALE_RANGE = (0.8, 3.5)
_login_scale: Optional[float] = None
"""Last scale the login screen matched at; tried first."""


def _match(img, t) -> Optional[Tuple[float, int, int]]:
    if img.shape[0] < t.shape[0] or img.shape[1] < t.shape[1]:
        return None
    res = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED)
    _, score, _, (x, y) = cv2.minMaxLoc(res)
    return score, x, y


def _scaled(t, scale: float) -> numpy.ndarray:
    if scale == 1.0:
        return t
    return cv2.resize(t, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_AREA if scale < 1
                      else cv2.INTER_LINEAR)


def _find_scaled(img, t) -> Optional[Tuple[float, int, int, int, int]]:
    """Best match of a template at any scale: the last scale that worked
    and 1:1 first, then a coarse search refined around the best candidate.
    Returns (score, x, y, w, h) or None."""
    global _login_scale

    def at(scale):
        tt = _scaled(t, scale)
        m = _match(img, tt)
        return None if m is None else (m[0], m[1], m[2], tt.shape[1],
                                       tt.shape[0])

    for scale in dict.fromkeys(s for s in (_login_scale, 1.0) if s):
        m = at(scale)
        if m and m[0] >= MATCH_THRESHOLD:
            return m
    best, best_scale = None, None
    for scale in numpy.arange(SCALE_RANGE[0], SCALE_RANGE[1] + 1e-9, 0.1):
        m = at(float(scale))
        if m and (best is None or m[0] > best[0]):
            best, best_scale = m, float(scale)
    if best is None or best[0] < 0.75:
        return None
    for scale in numpy.arange(best_scale - 0.06, best_scale + 0.0601, 0.01):
        m = at(round(float(scale), 2))
        if m and m[0] > best[0]:
            best, best_scale = m, round(float(scale), 2)
    if best[0] < MATCH_THRESHOLD:
        return None
    if best_scale != _login_scale:
        logger.info(f'Login screen scale: {best_scale:.2f}')
        _login_scale = best_scale
    return best


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


def launcher_selected_client() -> Optional[str]:
    """The client the launcher last used for OSRS, if known."""
    try:
        with open(launcher_settings_path(), 'r', encoding='utf-8') as f:
            prefs = json.load(f).get('gamePreferences', {})
        return prefs.get('lastGameClientByGameId', {}).get('osrs')
    except (OSError, ValueError, AttributeError):
        return None


def set_launcher_client(client_id: str = RUNELITE_CLIENT_ID):
    """Write the OSRS client choice; only while the launcher is closed."""
    path = launcher_settings_path()
    with open(path, 'r', encoding='utf-8') as f:
        settings = json.load(f)
    prefs = settings.setdefault('gamePreferences', {})
    prefs.setdefault('lastGameClientByGameId', {})['osrs'] = client_id
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(settings, f)
    os.replace(tmp, path)


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


def launcher_processes() -> list:
    return [p for p in psutil.process_iter(['name'])
            if (p.info['name'] or '').lower() == 'jagexlauncher.exe']


def close_launcher(timeout: float = 30, ask_for: float = 6):
    """Close the launcher (running games are separate processes).

    Closing its window may ask what closing should mean, or only minimise it
    to the tray, so anything left after ``ask_for`` seconds is ended.
    """
    handle = launcher_window()
    if handle:
        win32gui.PostMessage(handle, win32con.WM_CLOSE, 0, 0)
    start = time.time()
    while launcher_processes():
        if time.time() - start > timeout:
            raise LoginError('The Jagex Launcher did not close; exit it and '
                             'try again')
        if time.time() - start > ask_for:
            for p in launcher_processes():
                try:
                    p.terminate()
                except psutil.Error:
                    pass
        time.sleep(1)


def ensure_runelite_selected():
    """Make sure the launcher will start RuneLite, not another client.

    The launcher reads its choice from settings.json when it starts, so a
    wrong one is fixed by closing it and changing the setting. It is only
    touched when it's wrong.
    """
    client = launcher_selected_client()
    if client == RUNELITE_CLIENT_ID:
        logger.info('Launcher client is RuneLite')
        return
    if not exists(launcher_settings_path()):
        # never been run: nothing to check yet (see the check after Play)
        logger.warning('No launcher settings yet, cannot check the client')
        return
    logger.info(f'Launcher client is {client!r}, switching to RuneLite')
    close_launcher()
    set_launcher_client()
    if launcher_selected_client() != RUNELITE_CLIENT_ID:
        raise LoginError('Could not select RuneLite in the Jagex Launcher')


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


def official_pids() -> set:
    return {p.pid for p in psutil.process_iter(['name'])
            if (p.info['name'] or '').lower() in OFFICIAL_CLIENTS}


# --------------------------------------------------------------------------
# the flow
# --------------------------------------------------------------------------

def wait_for_launcher_page(handle: int, timeout: float = LAUNCHER_PAGE_TIMEOUT):
    """(image, origin, Play button) once the Old School page shows; Play is
    None if it doesn't (update checks can take a while)."""
    end = time.time() + timeout
    while True:
        img, origin = grab_window(handle)
        play = find(img, 'launcher_play')
        if play and find(img, 'launcher_osrs_logo'):
            return img, origin, play
        if time.time() > end:
            return img, origin, None
        time.sleep(1)


def launch_from_launcher(username: str, timeout: float = 120) -> ClientWindow:
    ensure_runelite_selected()
    # in a VM the launcher sometimes never draws its page: reopen it
    for attempt in range(LAUNCHER_ATTEMPTS):
        handle = open_launcher(timeout=60)
        img, origin, play = wait_for_launcher_page(handle)
        if play:
            break
        if attempt < LAUNCHER_ATTEMPTS - 1:
            logger.info('Launcher page not showing, reopening the launcher')
            close_launcher()
            time.sleep(5)
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
    official_before = official_pids()
    logger.info('Clicking Play in the Jagex Launcher')
    focus_window(handle)
    click_box(*play.screen_box(origin))

    # wait for the new RuneLite process and its window
    end = time.time() + timeout
    while time.time() < end:
        time.sleep(2)
        if official_pids() - official_before:
            raise LoginError('The Jagex Launcher started the official client; '
                             'select RuneLite in the launcher and try again')
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
    """Work out which login state a RuneLite screenshot shows.

    In-game is checked first (any one indicator is enough), then the lobby,
    login and connecting screens, each searched across the whole window.
    """
    for name, region in INGAME_INDICATORS:
        m = find(img, name, region)
        if m:
            return 'in_game', m
    for state, name in (('lobby', 'lobby_click_to_play'),
                        ('login', 'login_play_now'),
                        ('dialog', 'login_ok'),
                        ('connecting', 'login_connecting')):
        m = find(img, name)
        if m:
            return state, m
    for name in INGAME_FALLBACK:
        m = find(img, name)
        if m:
            return 'in_game', m
    return 'loading', None


def _first(img, names) -> Optional[Match]:
    for name in names:
        m = find(img, name)
        if m:
            return m
    return None


def detect_layout(img: numpy.ndarray) -> str:
    """Work out the in-game layout from a client screenshot.

    Uses only the positions of tabs relative to each other (and to the chat
    "All" button), so it doesn't matter where the UI is on screen. Returns
    one of ``resizable_modern``, ``resizable_classic``, ``fixed_classic`` or
    ``unknown`` (e.g. not in game, or the tabs are hidden).
    """
    upper = _first(img, UPPER_TABS) or _first(img, UPPER_TABS_EXTRA)
    lower = _first(img, LOWER_TABS)
    if upper is None or lower is None:
        return UNKNOWN

    dy = lower.y - upper.y
    if abs(dy) <= SAME_ROW_TOLERANCE:
        return MODERN
    if not STACKED_RANGE[0] <= dy <= STACKED_RANGE[1]:
        return UNKNOWN

    inventory = _first(img, UPPER_TABS)
    chat = _first(img, CHAT_ALL)
    if inventory and chat:
        ox, oy = chat.x - inventory.x, chat.y - inventory.y
        if (abs(ox - FIXED_CHAT_OFFSET[0]) <= FIXED_TOLERANCE
                and abs(oy - FIXED_CHAT_OFFSET[1]) <= FIXED_TOLERANCE):
            return FIXED_CLASSIC
    return RESIZABLE_CLASSIC


def check_layout(handle: int, log: Optional[logging.Logger] = None) -> str:
    """Grab the client once and warn if it isn't in Resizable - Modern."""
    log = log or logger
    try:
        img, _ = grab_window(handle)
        layout = detect_layout(img)
    except Exception as e:  # never stop a bot over this
        log.warning(f'Could not check game layout: {e}')
        return 'unknown'
    if layout == UNKNOWN:
        log.info('Could not verify the game layout (side panel tabs not '
                 'visible, e.g. not logged in)')
    elif layout != MODERN:
        log.warning(f'Game layout is "{LAYOUT_NAMES[layout]}", but the engine '
                    'expects "Resizable - Modern" (Settings > Display > Game '
                    'client layout).')
    else:
        log.info('Game layout: Resizable - Modern')
    return layout


def current_window(window: ClientWindow) -> Optional[ClientWindow]:
    """The client's main window right now.

    RuneLite shows a loading splash first and then opens the real client
    window, so the handle seen at launch can go away. Re-resolve by process
    each time, taking that process's largest visible client window.
    """
    candidates = [w for w in find_client_windows() if w.pid == window.pid]
    if not candidates:
        return None

    def area(w):
        try:
            x1, y1, x2, y2 = win32gui.GetWindowRect(w.handle)
            return (x2 - x1) * (y2 - y1)
        except win32gui.error:
            return 0

    return max(candidates, key=area)


def log_in(window: ClientWindow, timeout: float = 300,
           settle: float = 3.0, username: Optional[str] = None
           ) -> ClientWindow:
    """Click through the login screen and lobby until in game.

    Polls the client window; nothing is clicked until a known screen is
    recognised, so a slow first launch just means more polling. A login
    button must stay visible for ``settle`` seconds before it's clicked.
    """
    end = time.time() + timeout
    seen_at: Dict[str, float] = {}
    last_click = 0.0
    last_state = None
    last_logged_dialog = None
    gone_at = None
    while time.time() < end:
        current = current_window(window)
        if current is None and username:
            # RuneLite's launcher (the splash) starts the real client as
            # another process and may then exit
            current = find_runelite_for(username)
        if current is None:
            if not psutil.pid_exists(window.pid):
                gone_at = gone_at or time.time()
                if time.time() - gone_at > 30:
                    raise LoginError('RuneLite closed while logging in')
            time.sleep(1)
            continue
        gone_at = None
        if current.handle != window.handle:
            logger.info(f'Client window is now {current.title!r}')
            window = current
        try:
            focus_window(window.handle)
            img, origin = grab_window(window.handle)
        except win32gui.error:
            time.sleep(1)  # window went away between checks; re-resolve
            continue
        state, match = classify(img)
        if state != last_state:
            logger.info(f'Client state: {state}')
            last_state = state
        if state == 'in_game':
            return window
        if state == 'dialog' and state != last_logged_dialog:
            if find(img, 'login_disconnected'):
                logger.info('Login screen says: You were disconnected from '
                            'the server.')
        last_logged_dialog = state
        if state in ('login', 'lobby', 'dialog'):
            first = seen_at.setdefault(state, time.time())
            ready = time.time() - first >= settle
            cooled = time.time() - last_click > 5
            if ready and cooled:
                label = {'login': 'Play Now', 'lobby': 'Click here to play',
                         'dialog': 'Ok'}[state]
                logger.info(f'Clicking {label}')
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
    window = log_in(window, timeout=timeout, username=username)
    logger.info(f'{username} is logged in')
    return window
