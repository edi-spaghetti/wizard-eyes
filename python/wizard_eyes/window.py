"""Find and focus the game client window.

Windows are identified by the process that owns them (RuneLite.exe, or the
official client's osclient.exe), not just by title, so a browser tab or
folder with "RuneLite" in its name can't be mistaken for the client.
"""
import ctypes
import logging
import time
from ctypes import wintypes
from dataclasses import dataclass
from os.path import basename
from typing import List, Optional

import win32con
import win32gui
import win32process

logger = logging.getLogger('wizard_eyes.window')

CLIENT_EXECUTABLES = {'runelite.exe', 'osclient.exe'}
CLIENT_TITLES = ('RuneLite', 'Old School RuneScape')

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
_user32 = ctypes.WinDLL('user32', use_last_error=True)


@dataclass
class ClientWindow:
    handle: int
    title: str
    pid: int
    exe: str


def process_path(pid: int) -> str:
    """Full path of a process's executable, or '' if it can't be read."""
    h = _kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return ''
    try:
        size = wintypes.DWORD(1024)
        buf = ctypes.create_unicode_buffer(size.value)
        if _kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return buf.value
        return ''
    finally:
        _kernel32.CloseHandle(h)


def find_client_windows(title_contains: Optional[str] = None
                        ) -> List[ClientWindow]:
    """List visible top-level game client windows, ordered by process id.

    :param title_contains: Optionally only return windows whose title contains
        this text (case-insensitive), e.g. the account name RuneLite shows in
        its title, to pick between several clients.
    """
    found = []

    def visit(handle, _):
        if not win32gui.IsWindowVisible(handle):
            return True
        if win32gui.GetParent(handle):
            return True
        title = win32gui.GetWindowText(handle)
        if not title.startswith(CLIENT_TITLES):
            return True
        _, pid = win32process.GetWindowThreadProcessId(handle)
        exe = basename(process_path(pid)).lower()
        if exe not in CLIENT_EXECUTABLES:
            return True
        if title_contains and title_contains.lower() not in title.lower():
            return True
        found.append(ClientWindow(handle, title, pid, exe))
        return True

    win32gui.EnumWindows(visit, None)
    found.sort(key=lambda w: (w.pid, w.handle))
    return found


def select_client_window(title_contains: Optional[str] = None,
                         index: int = 1) -> Optional[ClientWindow]:
    """Pick one client window.

    :param title_contains: see :func:`find_client_windows`.
    :param index: 1-based position among matching windows (ordered by process
        id, so roughly launch order) when there are several.
    """
    windows = find_client_windows(title_contains)
    if not windows:
        return None
    if len(windows) > 1 and not title_contains:
        logger.warning(
            f'{len(windows)} client windows found, using #{index}. '
            'Use --window-title to choose one by name: '
            + ', '.join(repr(w.title) for w in windows))
    if not 1 <= index <= len(windows):
        logger.warning(f'--window-index {index} out of range, using #1')
        index = 1
    return windows[index - 1]


def _tap_alt():
    # Windows only lets a process take the foreground if it sent the last
    # input; a synthetic alt tap satisfies that without side effects.
    _user32.keybd_event(win32con.VK_MENU, 0, 0, 0)
    _user32.keybd_event(win32con.VK_MENU, 0, win32con.KEYEVENTF_KEYUP, 0)


def focus_window(handle: int, timeout: float = 2.0) -> bool:
    """Restore (if minimised) and bring a window to the foreground.

    :returns: True if the window ended up in the foreground.
    """
    if win32gui.IsIconic(handle):
        win32gui.ShowWindow(handle, win32con.SW_RESTORE)

    attempts = (_focus_with_alt, _focus_with_attach)
    for attempt in attempts:
        try:
            attempt(handle)
        except Exception as e:  # pywin32 raises on refused foreground
            logger.debug(f'{attempt.__name__} failed: {e}')
        end = time.time() + timeout / len(attempts)
        while time.time() < end:
            if win32gui.GetForegroundWindow() == handle:
                return True
            time.sleep(0.05)
    return win32gui.GetForegroundWindow() == handle


def _focus_with_alt(handle: int):
    _tap_alt()
    win32gui.SetForegroundWindow(handle)


def _focus_with_attach(handle: int):
    """Fallback: borrow the current foreground thread's input focus."""
    fg = win32gui.GetForegroundWindow()
    fg_thread, _ = win32process.GetWindowThreadProcessId(fg)
    our_thread = _kernel32.GetCurrentThreadId()
    attached = False
    if fg_thread and fg_thread != our_thread:
        attached = bool(_user32.AttachThreadInput(our_thread, fg_thread, True))
    try:
        win32gui.BringWindowToTop(handle)
        win32gui.SetForegroundWindow(handle)
    finally:
        if attached:
            _user32.AttachThreadInput(our_thread, fg_thread, False)
