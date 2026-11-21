"""Start-up sanity checks for bots, made optional and cached.

Most of the time a bot is (re)started on a machine and account that were
fine last time, so checks that passed are remembered in
``config/checks.local.json`` (gitignored) and skipped while still valid:

- RuneLite profile: cached against the profile file (path, size, mtime) and
  the rules file, so it re-runs as soon as either changes.
- Game layout: an account setting that can't be seen without looking, so a
  pass is cached per client window title (i.e. per character) for
  ``LAYOUT_CACHE_DAYS``.

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
    if layout == MODERN:
        layouts[title or ''] = {'layout': layout, 'checked_at': int(time.time())}
    elif layout != UNKNOWN:
        layouts.pop(title or '', None)
    return layout in (MODERN, UNKNOWN)


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

    for check in checks:
        try:
            check()
        except Exception as e:  # never stop a bot over a check
            log.warning(f'Start-up check failed to run: {e}')
    save_cache(cache)


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
        return _ensure_camera(handle, log)
    except Exception as e:  # never stop a bot over a check
        log.warning(f'Camera check failed to run: {e}')
        return None
