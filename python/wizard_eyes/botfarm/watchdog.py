"""Watches that a farm's bots keep working, and restarts one that doesn't.

Runs inside the farm's page (``wizard_eyes.botfarm.page`` or a farm's own,
with ``--watchdog``). Once a minute, per bot that should be running:

1. has it been heard from (its own report or the engine's heartbeat, see
   ``wizard_eyes.heartbeat``), and has its progress changed lately? Then all
   is well.
2. if not, look at it (the control script, ``-Action status``): is it
   at a meeting, logging in, waiting for a game update, paused, stopped for a
   reason a restart won't cure? Then leave it, and say so on the page.
3. otherwise save what it looked like and restart it, a few times at most.

A bot "should be running" from the moment it's started with the control
script until it's stopped with it (``wanted-ROLE.local`` in the farm's state
folder, written by that script). Nothing is ever done to one that isn't.

What "progress" is belongs to the farm: ``progress(report)`` gives anything
comparable from what a bot last sent (None: unknown, then only being heard
from counts). The default is the heartbeat's ``progress``
(``Application.progress``). A farm also adds its own stop reasons that a
restart won't cure (``no_cure``).
"""
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import (Any, Callable, Dict, Iterable, List, Optional, Pattern,
                    Tuple)

from ..coordination import last_seen

logger = logging.getLogger('wizard_eyes.botfarm.watchdog')

TICK_SECONDS = 60.0
REPORT_SECONDS = 8 * 60.0
"""Bots report / beat every minute or so; afk breaks are up to 4.5 min."""
PROGRESS_SECONDS = 45 * 60.0
"""Heard from, but its progress hasn't changed for this long."""
LOGIN_SECONDS = 25 * 60.0
"""Logging in for longer than this isn't getting anywhere."""
LOG_ALIVE_SECONDS = 10 * 60.0
MEETING_SECONDS = 40 * 60.0
GRACE_SECONDS = 5 * 60.0
"""After a start / restart has finished, or after this machine slept."""
MAX_PER_HOUR = 3
MAX_PER_DAY = 8
EVIDENCE_KEPT = 20

OK, WAIT, ALERT, RESTART = 'ok', 'wait', 'alert', 'restart'

PAUSED = re.compile(r'\bSleeping @')
RESTING = re.compile(r'Resting \(|Rest over|Logging back in')
REST_GRACE_SECONDS = 10 * 60.0
"""After a rest, time to log back in before it counts as idle."""
UPDATING = re.compile(r'game status page: .*trying again in')
LOGGING_IN = re.compile(
    r'Client state:|Clicking (Play|Ok|Click)|Launcher|RuneLite started|'
    r'Found RuneLite|starting the client again|logging back in|'
    r'Client window is now|Login screen|trying again in')
BANNED = 'the account is banned (login screen)'
NO_CURE: List[Tuple[Pattern, str]] = [
    (re.compile(r'account is banned|AccountBanned'), BANNED),
    (re.compile(r'\bPIN\b'), 'stopped at the bank PIN'),
    (re.compile(r'already logged in'), 'logged out again and again (is the '
                                       'account in use elsewhere?)'),
]
"""The engine's own stops a restart won't cure; farms add theirs."""
WHY = re.compile(r'stopping|Error|Exception|Traceback|Could not')


def judge(look: dict, stale: float,
          no_cure: Iterable[Tuple[Pattern, str]] = ()) -> Tuple[str, str]:
    """What to do about a bot that hasn't got on for ``stale`` seconds.

    :param look: ``vm`` (on?), ``running`` (the bot's task), ``log_age``
        (seconds since the log was written) and ``lines`` (its last lines).
    :param no_cure: the farm's own (pattern, reason) stops, before the
        engine's.
    """
    if not look['vm']:
        return RESTART, 'its VM is off'
    lines: List[str] = look['lines']
    tail = '\n'.join(lines[-6:])
    if not look['running']:
        said = '\n'.join(lines[-15:])
        for pattern, what in [*no_cure, *NO_CURE]:
            if pattern.search(said):
                return ALERT, what
        if not WHY.search(said):
            return ALERT, "stopped without saying why (the exit hotkey?)"
        why = [i for i in lines if WHY.search(i)]
        return RESTART, f'it stopped: {why[-1][-160:]}'
    if look['log_age'] > LOG_ALIVE_SECONDS:
        return RESTART, f'frozen ({look["log_age"] / 60:.0f} min without a ' \
                        f'line in its log)'
    if lines and RESTING.search(lines[-1]):
        return WAIT, 'resting'
    if lines and PAUSED.search(lines[-1]):
        return ALERT, 'paused (ctrl+p or Num Lock in the VM)'
    if UPDATING.search(tail):
        return WAIT, 'waiting for a game update'
    if lines and LOGGING_IN.search(lines[-1]) and stale < LOGIN_SECONDS:
        return WAIT, 'logging in'
    return RESTART, f'no progress for {stale / 60:.0f} min: ' \
                    f'{lines[-1][-120:] if lines else "empty log"}'


def heartbeat_progress(report: Optional[dict]) -> Any:
    """The default: ``Application.progress`` from the heartbeat."""
    return ((report or {}).get('heartbeat') or {}).get('progress')


class Bot:
    """What's known about one bot."""

    def __init__(self):
        self.progress: Any = None
        self.progress_at = 0.0
        self.bad_ticks = 0
        self.grace_until = 0.0
        self.busy = False
        self.state, self.text, self.since = OK, 'not looked at yet', 0.0


class Watchdog:

    def __init__(self, control, folder: Path, roles: Iterable[str],
                 bots: Callable[[], Dict[str, dict]],
                 inspect: Callable[[str, Optional[Path]], dict],
                 meetings: Optional[Callable[[], List[dict]]] = None,
                 progress: Callable[[Optional[dict]], Any] =
                 heartbeat_progress,
                 no_cure: Iterable[Tuple[Pattern, str]] = (),
                 clock: Callable[[], float] = time.time,
                 watch=None):
        """
        :param control: ``wizard_eyes.botfarm.control.Control``.
        :param roles: the farm's roles.
        :param bots: what the bots last sent (``Coordinator.bots``).
        :param inspect: look at a bot, saving a screenshot to the path given.
        :param meetings: all meetings, for farms whose bots meet up.
        :param progress: a bot's progress from what it last sent.
        :param watch: ``wizard_eyes.botfarm.alerts.Watch``.
        """
        self.control = control
        self.folder = Path(folder)
        self.roles = tuple(roles)
        self.bots = bots
        self.inspect = inspect
        self.meetings = meetings or (lambda: [])
        self.progress = progress
        self.no_cure = list(no_cure)
        self.clock = clock
        self.watch = watch
        self.known = {role: Bot() for role in self.roles}
        self.last_tick: Optional[float] = None
        self.path = self.folder / 'watchdog.local.json'
        self.restarts: Dict[str, List[float]] = {r: [] for r in self.roles}
        try:
            saved = json.loads(self.path.read_text())
            for r in self.roles:
                self.restarts[r] = [float(t) for t in saved.get(r, [])]
        except (OSError, ValueError, TypeError, AttributeError):
            pass

    def wanted(self, role: str) -> bool:
        try:
            text = (self.folder / f'wanted-{role}.local').read_text(
                'utf-8-sig')
        except OSError:
            return False
        return text.strip() == 'run'

    def status(self) -> dict:
        """For the page."""
        now = self.clock()
        return {r: {'state': b.state, 'text': b.text, 'since': b.since,
                    'restarts': sum(now - t < 86400
                                    for t in self.restarts[r])}
                for r, b in self.known.items()}

    def _say(self, role: str, state: str, text: str):
        b = self.known[role]
        if (b.state, b.text) != (state, text):
            if b.state != state or state not in (OK, WAIT):
                log = logger.info if state in (OK, WAIT) else logger.warning
                log(f'{role}: {state}: {text}')
            b.since = self.clock()
            # (an alert's text can change every minute)
            if self.watch is not None and b.state != state:
                self.watch.bot(role, state, text)
        b.state, b.text = state, text

    def tick(self):
        now = self.clock()
        if self.last_tick is not None and \
                now - self.last_tick > 3 * TICK_SECONDS:
            # this machine (and so the VMs) slept
            for b in self.known.values():
                b.grace_until = now + GRACE_SECONDS
        self.last_tick = now
        try:
            bots, meetings = self.bots(), self.meetings()
        except Exception as e:
            for role in self.roles:
                self._say(role, ALERT, f"can't tell: no coordination "
                                       f'service ({e})')
            return
        if self.watch is not None:
            self.watch.met(meetings)
        for role in self.roles:
            try:
                self._check(role, bots.get(role), meetings)
            except Exception as e:
                logger.exception(role)
                self._say(role, ALERT, f"couldn't be checked: {e!r}")
        # (whatever checking took isn't sleep)
        self.last_tick = self.clock()

    def _check(self, role: str, report: Optional[dict],
               meetings: List[dict]):
        b, now = self.known[role], self.clock()
        if not self.wanted(role):
            b.bad_ticks = 0
            b.grace_until = max(b.grace_until, now + GRACE_SECONDS)
            return self._say(role, OK, 'not watched (stopped by hand)')
        if self.control.busy(role):
            b.busy = True
            return self._say(role, WAIT, 'being started')
        if b.busy:
            b.busy = False
            b.grace_until = max(b.grace_until, now + GRACE_SECONDS)

        beat = (report or {}).get('heartbeat') or {}
        if beat.get('banned'):
            b.bad_ticks = 0
            return self._say(role, ALERT, BANNED)
        resting = beat.get('resting') or {}
        silent = now - last_seen(report)
        if resting and silent < REPORT_SECONDS:
            # a scheduled rest (wizard_eyes.rest): idle on purpose
            b.bad_ticks, b.progress_at = 0, now
            b.grace_until = max(b.grace_until, float(
                resting.get('until') or 0) + REST_GRACE_SECONDS)
            until = time.strftime('%H:%M', time.localtime(
                float(resting.get('until') or now)))
            return self._say(role, OK, f"resting ({resting.get('kind')}) "
                                       f"until {until}")

        progress = self.progress(report)
        if progress is None:
            b.progress, b.progress_at = None, now   # (unknown: never idle)
        elif b.progress is None or progress != b.progress:
            b.progress, b.progress_at = progress, now
        silent, idle = now - last_seen(report), now - b.progress_at
        if silent < REPORT_SECONDS and idle < PROGRESS_SECONDS:
            b.bad_ticks = 0
            return self._say(role, OK, 'working')
        if now < b.grace_until:
            return self._say(role, WAIT, 'just started')
        meeting = any(m['state'] in ('accepted', 'met') and
                      role in (m['by'], m['to']) and
                      now - m['updated'] < MEETING_SECONDS
                      for m in meetings)
        b.bad_ticks += 1
        if b.bad_ticks < 2:
            return self._say(role, WAIT, f'not heard from for '
                                         f'{silent / 60:.0f} min'
                             if silent >= REPORT_SECONDS else
                             f'no progress for {idle / 60:.0f} min')

        shot = self.folder / f'watchdog-{role}-{now:.0f}.local.png'
        look = self.inspect(role, shot)
        stale = min(silent, now) if silent >= REPORT_SECONDS else idle
        what, why = judge(look, stale, self.no_cure)
        if meeting and look['running'] and \
                look['log_age'] < LOG_ALIVE_SECONDS:
            what, why = WAIT, 'at a meeting'
        # (looking takes a while: stopped by hand meanwhile?)
        if not self.wanted(role) or self.control.busy(role):
            what = WAIT
        if what == RESTART:
            what, why = self._limited(role, what, why)
        if what == RESTART:
            self._evidence(role, now, why, look)
            self._restart(role, now, look)
            b.bad_ticks, b.busy = 0, True
            b.progress = None
            return self._say(role, RESTART, f'restarted: {why}')
        shot.unlink() if shot.exists() else None
        self._say(role, what, why)

    def _limited(self, role: str, what: str, why: str) -> Tuple[str, str]:
        now = self.clock()
        times = self.restarts[role] = [
            t for t in self.restarts[role] if now - t < 86400]
        hour = sum(now - t < 3600 for t in times)
        if hour >= MAX_PER_HOUR or len(times) >= MAX_PER_DAY:
            return ALERT, f'restarted {hour} times this hour, {len(times)} ' \
                          f'today: not again ({why})'
        return what, why

    def _restart(self, role: str, now: float, look: dict):
        # counted first: if starting it fails, that's an attempt too
        self.restarts[role].append(now)
        self.path.write_text(json.dumps(self.restarts))
        self.control.run(role, 'restart' if look['vm'] else 'start')

    def _evidence(self, role: str, now: float, why: str, look: dict):
        text = [why, f'log age {look["log_age"]:.0f}s, running '
                     f'{look["running"]}, vm {look["vm"]}', *look['lines']]
        (self.folder / f'watchdog-{role}-{now:.0f}.local.txt').write_text(
            '\n'.join(text), 'utf-8')
        for kind in ('txt', 'png'):
            old = sorted(self.folder.glob(f'watchdog-*-*.local.{kind}'),
                         key=lambda p: p.stat().st_mtime)[:-EVIDENCE_KEPT]
            for p in old:
                p.unlink()

    def run(self):
        while True:
            try:
                self.tick()
            except Exception:
                logger.exception('watchdog')
            time.sleep(TICK_SECONDS)

    def start(self):
        threading.Thread(target=self.run, daemon=True).start()


def parse_look(text: str) -> dict:
    """The control script's status: a line of JSON, checked."""
    line = next(i for i in text.splitlines() if i.startswith('{'))
    d = json.loads(line)
    lines = d.get('lines') or []
    if isinstance(lines, str):
        lines = [lines]
    return {'vm': bool(d['vm']), 'running': bool(d.get('running')),
            'log_age': float(d.get('log_age') or 0),
            'lines': [str(i) for i in lines]}
