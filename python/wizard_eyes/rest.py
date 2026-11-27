"""Rest in a bot's day: a night's sleep and short breaks, instead of playing
round the clock.

Any bot takes these command-line options (``Application.run`` reads them, so
a farm sets them in the role's args in farm.json):

- ``--sleep 23:00-07:00``: every day. Start and end each move by up to
  ``--rest-jitter`` minutes (default 45), drawn afresh every night.
- ``--breaks 50-130:3-15``: a break every 50 to 130 minutes of play, 3 to 15
  minutes long (never one that runs into the night's sleep).

Neither given: the bot plays without rests, as before. Times are this
machine's local time.

A bot stops at a moment that suits it (``Application.can_rest``), then does
nothing until the rest is over. The game logs an idle player out after about
5 minutes, which is what someone stepping away looks like too; the bot logs
back in afterwards (that needs ``--username``).
"""
import argparse
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, List, Optional, Sequence, Tuple

JITTER_MINUTES = 45.0
DAY_MINUTES = 24 * 60
SCHEDULED = ('sleep', 'break')
"""Kinds of rest from a schedule; a bot's own waits (``Application.rest``
with any other kind, e.g. 'waiting for herbs') aren't rests from play."""

Range = Tuple[float, float]


@dataclass
class Rest:
    kind: str           # 'sleep' or 'break'
    start: float        # epoch seconds
    end: float

    @property
    def minutes(self) -> float:
        return (self.end - self.start) / 60

    def __str__(self):
        until = datetime.fromtimestamp(self.end).strftime('%H:%M')
        return f'{self.kind} until {until} ({self.minutes:.0f} min)'


def parse_sleep(text: str) -> Tuple[int, int]:
    """'23:00-07:00' -> minutes of the day (1380, 420)."""
    m = re.fullmatch(r'\s*(\d{1,2}):(\d\d)\s*-\s*(\d{1,2}):(\d\d)\s*', text)
    if not m:
        raise ValueError(f'sleep {text!r}: expected HH:MM-HH:MM')
    h1, m1, h2, m2 = map(int, m.groups())
    if not (h1 < 24 and h2 < 24 and m1 < 60 and m2 < 60):
        raise ValueError(f'sleep {text!r}: not a time of day')
    start, end = h1 * 60 + m1, h2 * 60 + m2
    if start == end:
        raise ValueError(f'sleep {text!r}: starts when it ends')
    return start, end


def _range(text: str) -> Range:
    m = re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*(?:-\s*(\d+(?:\.\d+)?))?\s*',
                     text)
    if not m:
        raise ValueError(f'{text!r}: expected MIN-MAX (minutes)')
    a = float(m.group(1))
    b = float(m.group(2)) if m.group(2) else a
    if not 0 < a <= b:
        raise ValueError(f'{text!r}: expected 0 < MIN <= MAX')
    return a, b


def parse_breaks(text: str) -> Tuple[Range, Range]:
    """'50-130:3-15' -> ((50, 130), (3, 15)): every, how long (minutes)."""
    every, sep, length = text.partition(':')
    if not sep:
        raise ValueError(f'breaks {text!r}: expected EVERY:LENGTH, '
                         f'e.g. 50-130:3-15')
    return _range(every), _range(length)


class RestSchedule:

    def __init__(self, sleep: Optional[Tuple[int, int]] = None,
                 breaks: Optional[Tuple[Range, Range]] = None,
                 jitter: float = JITTER_MINUTES,
                 rng: Optional[random.Random] = None):
        self.sleep = sleep
        self.breaks = breaks
        self.jitter = jitter
        self.rng = rng or random.Random()
        self.next: Optional[Rest] = None

    def __str__(self):
        said = []
        if self.sleep:
            s, e = self.sleep
            said.append(f'sleep {s // 60:02d}:{s % 60:02d}-'
                        f'{e // 60:02d}:{e % 60:02d} '
                        f'(+-{self.jitter:.0f} min)')
        if self.breaks:
            (a, b), (c, d) = self.breaks
            said.append(f'a {c:g}-{d:g} min break every {a:g}-{b:g} min')
        return ', '.join(said) or 'no rests'

    def _moved(self, t: float) -> float:
        return t + self.rng.uniform(-self.jitter, self.jitter) * 60

    def next_sleep(self, now: float) -> Optional[Rest]:
        """The night's sleep we're in, or the next one."""
        if not self.sleep:
            return None
        start_min, end_min = self.sleep
        length = (end_min - start_min) % DAY_MINUTES
        today = datetime.fromtimestamp(now).replace(
            hour=0, minute=0, second=0, microsecond=0)
        for days in (-1, 0, 1, 2):
            day = today + timedelta(days=days)
            start = (day + timedelta(minutes=start_min)).timestamp()
            end = (day + timedelta(minutes=start_min + length)).timestamp()
            start, end = self._moved(start), self._moved(end)
            if end > now:
                return Rest('sleep', max(start, now), end)
        raise AssertionError('no sleep within two days')

    def plan(self, now: float) -> Optional[Rest]:
        """The next rest from ``now``: a break, if one comes before the
        night's sleep, else the sleep."""
        sleep = self.next_sleep(now)
        if self.breaks and (sleep is None or sleep.start > now):
            (a, b), (c, d) = self.breaks
            start = now + self.rng.uniform(a, b) * 60
            pause = Rest('break', start, start + self.rng.uniform(c, d) * 60)
            if sleep is None or pause.end <= sleep.start:
                return pause
        return sleep

    def due(self, now: float) -> Optional[Rest]:
        """The rest to take now, if one is due."""
        if self.next is None:
            self.next = self.plan(now)
        if self.next is not None and now >= self.next.start:
            if now >= self.next.end:      # (missed: e.g. the PC slept)
                self.next = self.plan(now)
                return self.due(now)
            return self.next
        return None

    def taken(self, now: float):
        """The rest is over (or was cut short): plan the next one."""
        self.next = self.plan(now)

    def upcoming(self, now: float) -> Optional[Rest]:
        """The next rest (due now or later), planning it if need be."""
        if self.next is None or now >= self.next.end:
            self.next = self.plan(now)
        return self.next


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--sleep', type=parse_sleep,
                   help='rest every day from..to, e.g. 23:00-07:00 '
                        '(local time)')
    p.add_argument('--breaks', type=parse_breaks,
                   help='EVERY:LENGTH in minutes, e.g. 50-130:3-15')
    p.add_argument('--rest-jitter', type=float, default=JITTER_MINUTES,
                   help='minutes the sleep start / end move each night')
    return p


def from_args(argv: Sequence[str]) -> Optional[RestSchedule]:
    """The schedule asked for on the command line; None for no rests."""
    args, _ = parser().parse_known_args(list(argv))
    if not (args.sleep or args.breaks):
        return None
    return RestSchedule(args.sleep, args.breaks, args.rest_jitter)
