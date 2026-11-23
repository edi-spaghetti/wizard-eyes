"""Set the game layout and default zoom from the in-game Display Settings.

Works from any layout: only the settings tab, the Display Settings panel and
its drop-down need recognising. Everything inside the panel is placed
relative to its "Game client layout:" label. Each step checks its outcome; a
step that fails twice gives up (the caller warns) rather than clicking on.
"""
import logging
import time
from typing import Optional

import pyautogui

from . import login
from .login import MODERN, click_box, detect_layout, find, grab_window
from .window import focus_window

logger = logging.getLogger('wizard_eyes.display_settings')

login.TEMPLATES.update({
    'display_settings_tab': 'tabs/settings.npy',
    'display_settings_tab_selected': 'tabs/settings_selected.npy',
    'display_icon': 'display/icon.npy',
    'display_title': 'display/title.npy',
    'display_layout_label': 'display/layout_label.npy',
    'display_option_modern': 'display/option_modern.npy',
    'display_restore_zoom': 'display/restore_zoom.npy',
})

# centres relative to the layout label's top-left corner
DROPDOWN = (82, 26)
ZOOM_ICON = (12, -22)
# the drop-down's list starts below its own box
LIST_FROM = 40
ATTEMPTS = 2


def _click(match, origin):
    click_box(*match.screen_box(origin))


def _wait_for(handle, name: str, timeout: float = 3.0):
    """(match, image, origin) once a template shows; match None if not."""
    end = time.time() + timeout
    while True:
        img, origin = grab_window(handle)
        m = find(img, name)
        if m or time.time() > end:
            return m, img, origin
        time.sleep(0.2)


def open_panel(handle: int, log: logging.Logger) -> bool:
    """Open the settings tab on its Display page."""
    for _ in range(ATTEMPTS):
        img, origin = grab_window(handle)
        if find(img, 'display_layout_label') and find(img, 'display_title'):
            return True
        tab = find(img, 'display_settings_tab')
        if tab and not find(img, 'display_settings_tab_selected'):
            _click(tab, origin)
        if _wait_for(handle, 'display_title', 2.0)[0]:
            return True
        icon, _, origin = _wait_for(handle, 'display_icon', 1.0)
        if icon:
            _click(icon, origin)
            if _wait_for(handle, 'display_title')[0]:
                return True
    log.warning('Display settings: could not open the panel')
    return False


def set_modern_layout(handle: int, log: Optional[logging.Logger] = None
                      ) -> bool:
    """Switch to Resizable - Modern unless it's already set."""
    log = log or logger
    if detect_layout(grab_window(handle)[0]) == MODERN:
        return True
    focus_window(handle)
    if not open_panel(handle, log):
        return False
    for _ in range(ATTEMPTS):
        label, _, origin = _wait_for(handle, 'display_layout_label')
        if not label:
            break
        x = origin[0] + label.x + DROPDOWN[0]
        y = origin[1] + label.y + DROPDOWN[1]
        click_box(x - 30, y - 4, x + 30, y + 4)
        option, end = None, time.time() + 3
        while option is None and time.time() < end:
            time.sleep(0.2)
            img, origin = grab_window(handle)
            # the same text can show in the box itself: only look in the list
            top = label.y + LIST_FROM
            option = find(img[top:], 'display_option_modern')
            if option:
                option.y += top
        if option is None:
            continue
        log.info('Display settings: choosing Resizable - Modern')
        _click(option, origin)
        end = time.time() + 6
        while time.time() < end:
            time.sleep(0.5)
            if detect_layout(grab_window(handle)[0]) == MODERN:
                return True
    log.warning('Display settings: could not switch to Resizable - Modern')
    return False


def restore_default_zoom(handle: int, log: Optional[logging.Logger] = None
                         ) -> bool:
    """Choose "Restore Default Zoom" on the zoom icon's right-click menu."""
    log = log or logger
    focus_window(handle)
    if not open_panel(handle, log):
        return False
    for _ in range(ATTEMPTS):
        label, _, origin = _wait_for(handle, 'display_layout_label')
        if not label:
            break
        x = origin[0] + label.x + ZOOM_ICON[0]
        y = origin[1] + label.y + ZOOM_ICON[1]
        pyautogui.moveTo(x, y, 0.3)
        time.sleep(0.2)
        pyautogui.rightClick()
        option, _, origin = _wait_for(handle, 'display_restore_zoom')
        if option:
            log.info('Display settings: restoring the default zoom')
            _click(option, origin)
            time.sleep(0.8)
            if not find(grab_window(handle)[0], 'display_restore_zoom'):
                return True
    log.warning('Display settings: could not restore the default zoom')
    return False
