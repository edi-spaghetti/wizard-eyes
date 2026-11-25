"""Whether the game is up, from Jagex's status page, and waiting for it.

The page is updated by hand and can be late either way, so it only decides
how patient to be, never whether to try at all:

* OFFLINE: try every few minutes, for hours if need be (maintenance takes
  about three), and right away once the page says ONLINE
* ONLINE, or the page can't be read: try every minute or so, for an hour

It's only read while a bot can't log in, at most every ``POLL_SECONDS``.
"""
import logging
import random
import re
import time
from typing import Optional, Tuple
from urllib.request import Request, urlopen

URL = ('https://secure.runescape.com/m=news/'
       'game-status-information-centre?oldschool=1')
ONLINE, OFFLINE = 'online', 'offline'
POLL_SECONDS = 120.0
FETCH_TIMEOUT = 10.0

RETRY = (45.0, 100.0)
"""Seconds between tries to log in."""
OFFLINE_RETRY = (270.0, 340.0)
BACK_ONLINE = (5.0, 30.0)
"""From seeing the page change to ONLINE to trying (bots don't all rush)."""
PATIENCE = 3600.0
OFFLINE_PATIENCE = 6 * 3600.0
STEP = 10.0

logger = logging.getLogger('wizard_eyes.game_status')
_last: Tuple[float, Optional[str]] = (-float('inf'), None)


def parse(html: str) -> Optional[str]:
    """The page's "Current Status", or None if it isn't clear."""
    # the status that doesn't apply is left in as a comment
    html = re.sub(r'<!--.*?-->', '', html, flags=re.S)
    m = re.search(r'id=["\']Status["\'][^>]*>(.*?)</div>', html, flags=re.S)
    if not m:
        return None
    text = re.sub(r'<[^>]+>', ' ', m.group(1)).upper()
    words = {w.lower() for w in re.findall(r'\b(ONLINE|OFFLINE)\b', text)}
    return words.pop() if len(words) == 1 else None


def fetch() -> Optional[str]:
    try:
        request = Request(URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urlopen(request, timeout=FETCH_TIMEOUT) as r:
            return parse(r.read().decode('iso-8859-1'))
    except Exception as e:  # never stop a login over this
        logger.info(f'Could not read the game status page: {e}')
        return None


def current() -> Optional[str]:
    """The game's status (read again when it's old), None if unknown."""
    global _last
    if time.time() - _last[0] >= POLL_SECONDS:
        _last = (time.time(), fetch())
    return _last[1]


class Waiting:
    """Waiting for the game to come back: :meth:`pause` between tries, until
    :attr:`expired`."""

    def __init__(self):
        self.started = time.time()
        self.deadline = self.started + PATIENCE

    @property
    def expired(self) -> bool:
        return time.time() > self.deadline

    def pause(self, why: str):
        status = current()
        offline = status == OFFLINE
        if offline:
            # an hour from the last time it was down
            self.deadline = max(self.deadline, min(
                time.time() + PATIENCE, self.started + OFFLINE_PATIENCE))
        wait = random.uniform(*(OFFLINE_RETRY if offline else RETRY))
        logger.info(f'{why}; game status page: {status or "unknown"}, trying '
                    f'again in {wait:.0f}s')
        end = time.time() + wait
        while time.time() < end:
            time.sleep(min(STEP, max(.1, end - time.time())))
            if offline and current() == ONLINE:
                logger.info('Game status page: online again')
                time.sleep(random.uniform(*BACK_ONLINE))
                return
