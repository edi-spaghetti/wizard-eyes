"""Start-up sanity checks for bots, made optional and cached.

Most of the time a bot is (re)started on a machine and account that were
fine last time, so checks that passed are remembered in
``config/checks.local.json`` (gitignored) and skipped while still valid:

- RuneLite profile: cached against the profile file (path, size, mtime) and
  the rules file, so it re-runs as soon as either changes.
- Game layout: a setting that can't be seen without looking, so a pass is
  cached per client window title (i.e. per character) for
  ``LAYOUT_CACHE_DAYS``. A wrong layout is switched to Resizable - Modern. Brightness is turned down
  to the minimum (also cached).
- Default zoom: cached the same way. The camera check sees the zoom for free
  on flat ground, so a zoom that looks wrong is always fixed, and one that
  can't be judged is only reset if it has no cached pass.

The client window is maximised first (``--skip-maximise`` leaves it alone).

Only passes are cached; a failure is re-checked (and warned about) on every
start. Command line:

    --skip-checks   skip all of these checks
    --recheck       ignore the cache and check everything now

The camera (north / top-down / zoom) is handled separately by
:func:`ensure_camera`, which runs on every start because the camera changes
between sessions; ``--skip-camera`` skips it.
"""
import hashlib
import json
import logging
import os
import time
from typing import Callable, Dict, Optional

from .file_path_utils import get_root

logger = logging.getLogger('wizard_eyes.startup_checks')

LAYOUT_CACHE_DAYS = 7


def cache_path() -> str:
    return os.path.join(get_root(), 'config', 'checks.local.json')


def load_cache() -> Dict:
    try:
        with open(cache_path(), 'r', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_cache(cache: Dict):
    try:
        with open(cache_path(), 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2)
    except OSError as e:
        logger.debug(f'Could not save check cache: {e}')


def _file_stamp(path: str) -> Optional[list]:
    try:
        st = os.stat(path)
        return [os.path.normcase(os.path.abspath(path)), st.st_size,
                int(st.st_mtime)]
    except OSError:
        return None


def _file_hash(path: str) -> Optional[str]:
    try:
        with open(path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except OSError:
        return None


# --------------------------------------------------------------------------
# individual checks
# --------------------------------------------------------------------------

def check_profile(cache: Dict, recheck: bool, log: logging.Logger) -> bool:
    from .runelite_profile import (active_profile, default_rules_path,
                                   log_profile_check)
    try:
        profile = active_profile()
    except Exception:
        profile = None
    key = {
        'profile': _file_stamp(profile.path) if profile else None,
        'rules': _file_hash(default_rules_path()),
    }
    if not recheck and key['profile'] and cache.get('profile') == key:
        log.info(f'RuneLite profile check: passed previously ({profile.name}), '
                 'skipping')
        return True

    result = log_profile_check(log)
    if result.ok and key['profile']:
        cache['profile'] = key
    else:
        cache.pop('profile', None)
    return result.ok


def check_layout(cache: Dict, recheck: bool, log: logging.Logger,
                 handle: int, title: str) -> bool:
    from .login import MODERN, UNKNOWN, check_layout as run_layout_check
    layouts = cache.setdefault('layout', {})
    entry = layouts.get(title or '')
    fresh = entry and time.time() - entry.get('checked_at', 0) < \
        LAYOUT_CACHE_DAYS * 86400
    if not recheck and fresh and entry.get('layout') == MODERN:
        log.info(f'Game layout check: passed previously for "{title}", '
                 'skipping')
        return True

    layout = run_layout_check(handle, log)
    if layout not in (MODERN, UNKNOWN):
        from .display_settings import set_modern_layout
        if set_modern_layout(handle, log):
            layout = MODERN
    if layout == MODERN:
        layouts[title or ''] = {'layout': layout, 'checked_at': int(time.time())}
    elif layout != UNKNOWN:
        layouts.pop(title or '', None)
    return layout in (MODERN, UNKNOWN)


def check_brightness(cache: Dict, recheck: bool, log: logging.Logger,
                     handle: int, title: str) -> bool:
    """Screen brightness at MINIMUM (user's rule: the templates depend on it).
    Only in game (the setting is in the Display Settings panel); cached like
    the layout, so a bot that has passed once sends no input for it."""
    from .login import MODERN, detect_layout, grab_window
    entries = cache.setdefault('brightness', {})
    entry = entries.get(title or '')
    fresh = entry and time.time() - entry.get('checked_at', 0) <         LAYOUT_CACHE_DAYS * 86400
    if not recheck and fresh and entry.get('minimum'):
        log.info(f'Brightness check: passed previously for "{title}", '
                 'skipping')
        return True
    if detect_layout(grab_window(handle)[0]) != MODERN:
        return True     # not in game (yet): next start
    from .display_settings import set_brightness_minimum
    if set_brightness_minimum(handle, log):
        entries[title or ''] = {'minimum': True,
                                'checked_at': int(time.time())}
        return True
    entries.pop(title or '', None)
    return False


# --------------------------------------------------------------------------

def run(client, log: Optional[logging.Logger] = None) -> None:
    """Run (or skip) the start-up checks for an application's client.

    Never raises and never stops a bot; problems are logged as warnings.
    """
    log = log or logger
    args = getattr(client, 'args', None)
    if getattr(args, 'skip_checks', False):
        log.info('Start-up checks skipped (--skip-checks)')
        return
    recheck = bool(getattr(args, 'recheck', False))

    cache = load_cache()
    checks: list = [lambda: check_profile(cache, recheck, log)]
    handle = getattr(client, '_win_handle', 0)
    if handle:
        checks.append(lambda: check_layout(
            cache, recheck, log, handle, getattr(client, 'title', '') or ''))
        checks.append(lambda: check_brightness(
            cache, recheck, log, handle, getattr(client, 'title', '') or ''))

    for check in checks:
        try:
            check()
        except Exception as e:  # never stop a bot over a check
            log.warning(f'Start-up check failed to run: {e}')
    save_cache(cache)


def ensure_maximised(client, log: Optional[logging.Logger] = None):
    """Maximise the client window (no input if it already is)."""
    log = log or logger
    handle = getattr(client, '_win_handle', 0)
    if getattr(getattr(client, 'args', None), 'skip_maximise', False) \
            or not handle:
        return
    try:
        from .window import maximise_window
        if not maximise_window(handle):
            log.warning('Could not maximise the client window')
        client._rect = None  # measured again at the new size
    except Exception as e:  # never stop a bot over a check
        log.warning(f'Could not maximise the client window: {e}')


def zoom_needs_reset(zoom_ok: Optional[bool], cached: bool) -> bool:
    """Wrong: always. Can't tell (uneven ground): unless it passed before."""
    return zoom_ok is False or (zoom_ok is None and not cached)


def ensure_zoom(client, state, log: logging.Logger):
    """Restore the default zoom if needed; returns the camera state."""
    from .camera import camera_state
    from .display_settings import restore_default_zoom
    args = getattr(client, 'args', None)
    if getattr(args, 'skip_checks', False) or state is None \
            or state.north is None:  # not in game
        return state
    cache = load_cache()
    passes = cache.setdefault('zoom', {})
    title = getattr(client, 'title', '') or ''
    cached = (not getattr(args, 'recheck', False)
              and time.time() - passes.get(title, 0)
              < LAYOUT_CACHE_DAYS * 86400)
    ok = state.zoom_ok is True
    if zoom_needs_reset(state.zoom_ok, cached):
        ok = restore_default_zoom(client._win_handle, log)
        state = camera_state(client._win_handle)
        ok = ok and state.zoom_ok is not False
        log.info(f'Camera: {state.summary()}')
    if ok:
        passes[title] = int(time.time())
    elif state.zoom_ok is False:
        passes.pop(title, None)
        log.warning('The zoom still looks wrong; set it to default by hand')
    save_cache(cache)
    return state


def ensure_camera(client, log: Optional[logging.Logger] = None):
    """Face north, tilt top-down and check zoom before the bot starts.

    Runs on every start (the camera moves between sessions, and is angled
    after logging in) but sends no input if the camera is already right,
    so it costs one screen grab. Skip with ``--skip-camera``.
    """
    log = log or logger
    args = getattr(client, 'args', None)
    handle = getattr(client, '_win_handle', 0)
    if getattr(args, 'skip_camera', False) or not handle:
        if handle:
            log.info('Camera check skipped (--skip-camera)')
        return None
    try:
        from .camera import ensure_camera as _ensure_camera
        return ensure_zoom(client, _ensure_camera(handle, log), log)
    except Exception as e:  # never stop a bot over a check
        log.warning(f'Camera check failed to run: {e}')
        return None
