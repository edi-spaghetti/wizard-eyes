"""A heartbeat every bot sends without any code of its own.

``Application.run`` starts it when the bot was started by a farm
(``botfarm/bot-control.ps1`` sets the variables below in the VM), so a farm's
page and watchdog know every bot is alive, whatever the script:

- ``WIZARD_EYES_COORDINATOR``: the coordination service's URL (in a VM:
  ``http://10.0.2.2:PORT``)
- ``WIZARD_EYES_ROLE``: the bot's role in its farm

Every ``INTERVAL`` seconds, from a thread (never in the bot's way), it sends
``PUT /heartbeat/ROLE`` with what the engine knows: the script, how long it
has run, frames, whether it's sleeping (paused), the latest log
line, a rest it's taking (``resting``: kind and until, see
:mod:`wizard_eyes.rest`), the next scheduled one (``next_rest``: kind,
start, until; a farm doesn't plan work with the bot into it), ``banned`` once the login screen said so, and
``Application.progress`` (a bot may count whatever it gets done in
it: the farm's watchdog then also notices one that's alive but stuck).

No service, no problem: it's said once in the log and the bot carries on.
"""
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .coordination import Coordinator, CoordinatorUnavailable

logger = logging.getLogger('wizard_eyes.heartbeat')

ENV_URL = 'WIZARD_EYES_COORDINATOR'
ENV_ROLE = 'WIZARD_EYES_ROLE'
INTERVAL = 60.0
LINE_LENGTH = 200


def from_env(environ=None) -> Optional[Dict[str, str]]:
    environ = os.environ if environ is None else environ
    url, role = environ.get(ENV_URL), environ.get(ENV_ROLE)
    if not (url and role):
        return None
    return {'url': url, 'role': role}


def snapshot(app) -> Dict[str, Any]:
    """What the engine knows about a running application (read from another
    thread: only plain attributes, never the client's images)."""
    started = getattr(app, 'start_time', float('inf'))
    lines = getattr(app, 'msg_buffer', None) or []
    beat = {
        'script': Path(sys.argv[0]).name if sys.argv and sys.argv[0] else '',
        'pid': os.getpid(),
        'running_for': round(time.time() - started, 1)
        if started != float('inf') else 0.0,
        'frames': getattr(app, 'frame_number', 0),
        'sleeping': bool(getattr(app, 'sleeping', False)),
        'line': str(lines[-1])[:LINE_LENGTH] if lines else '',
    }
    if getattr(app, 'banned', False):
        beat['banned'] = True
    rest = getattr(app, 'resting', None)
    if rest is not None:
        beat['resting'] = {'kind': rest.kind, 'until': rest.end}
    schedule = getattr(app, 'rest_schedule', None)
    upcoming = getattr(schedule, 'next', None)
    if upcoming is not None:
        # (read only: planning one belongs to the bot's own thread)
        beat['next_rest'] = {'kind': upcoming.kind, 'start': upcoming.start,
                             'until': upcoming.end}
    progress = getattr(app, 'progress', None)
    if progress is not None:
        beat['progress'] = progress
    return beat


class Heartbeat:

    def __init__(self, url: str, role: str,
                 beat: Callable[[], Dict[str, Any]],
                 interval: float = INTERVAL, client=None):
        self.role = role
        self.beat = beat
        self.interval = interval
        self.client = client or Coordinator(url, role)
        self.stopped = threading.Event()
        self.failing = False
        self.sent = 0
        self.thread: Optional[threading.Thread] = None

    def send(self) -> bool:
        try:
            self.client.heartbeat(self.beat())
        except CoordinatorUnavailable as e:
            if not self.failing:
                logger.warning(f'No heartbeat to the farm ({e}); '
                               f'trying again quietly')
            self.failing = True
            return False
        except Exception as e:      # never the bot's problem
            if not self.failing:
                logger.warning(f'Heartbeat failed: {e!r}')
            self.failing = True
            return False
        if self.failing:
            logger.info('Heartbeat to the farm again')
        self.failing = False
        self.sent += 1
        return True

    def run(self):
        while not self.stopped.is_set():
            self.send()
            self.stopped.wait(self.interval)

    def start(self) -> 'Heartbeat':
        self.thread = threading.Thread(target=self.run, daemon=True,
                                       name='heartbeat')
        self.thread.start()
        return self

    def stop(self):
        self.stopped.set()


def start_for(app, environ=None) -> Optional[Heartbeat]:
    """The application's heartbeat, if the farm asked for one."""
    where = from_env(environ)
    if where is None:
        return None
    logger.info(f"Heartbeat to the farm as '{where['role']}'")
    return Heartbeat(where['url'], where['role'],
                     lambda: snapshot(app)).start()
