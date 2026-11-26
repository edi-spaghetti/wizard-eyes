"""Alerts to the owner's phone through a Telegram bot: a bot restarted or in
need of a human, working again, a meeting that failed. Sends only: nothing
it's told over Telegram does anything. Texts name roles, never accounts.

Set-up, once (``botfarm/setup.ps1`` walks through it):

1. make a bot with @BotFather, store its token in Windows' Credential Manager
   (asks for it; it's never logged, nor kept anywhere else)::

       cmdkey /generic:wizard-eyes/telegram /user:bot /pass

   (a farm can name another credential: farm.json ``alerts.credential``)

2. ``python -m wizard_eyes.botfarm.alerts pair`` prints a code: send the bot
   ``/start CODE``. Only that chat gets the alerts (kept in the farm's state
   folder).

``... alerts test`` sends a test message, ``... alerts status`` says whether
it's paired. ``--farm DIR`` / ``--folder DIR`` as in ``farm.py``.
"""
import argparse
import json
import logging
import queue
import secrets
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.request import Request, urlopen

from .farm import DEFAULT_CREDENTIAL, Farm, NoFarm

logger = logging.getLogger('botfarm.alerts')

CHAT = 'telegram.local.json'
API = 'https://api.telegram.org/bot{token}/{method}'
TIMEOUT = 15
TRIES = 4
TRY_AGAIN = 60.0
REPEAT_SECONDS = 1800.0
"""The same thing isn't said again before this."""
PAIR_SECONDS = 300.0


class TelegramError(Exception):
    """(Its text never holds the token.)"""


def stored_token(credential: str = DEFAULT_CREDENTIAL) -> Optional[str]:
    try:
        import win32cred
        blob = win32cred.CredRead(
            credential, win32cred.CRED_TYPE_GENERIC)['CredentialBlob']
    except Exception:
        return None
    text = blob.decode('utf-16-le') if isinstance(blob, bytes) else blob
    return text.strip() or None


class Telegram:

    def __init__(self, folder: Path,
                 token: Callable[[], Optional[str]] = stored_token,
                 opener=urlopen):
        self.path = Path(folder) / CHAT
        self.token = token
        self.opener = opener

    @classmethod
    def for_farm(cls, farm: Farm, folder: Optional[Path] = None):
        credential = farm.telegram_credential
        return cls(folder or farm.state_dir,
                   lambda: stored_token(credential))

    def chat(self) -> Optional[int]:
        try:
            return int(json.loads(self.path.read_text())['chat'])
        except (OSError, ValueError, KeyError, TypeError):
            return None

    @property
    def ready(self) -> bool:
        return self.chat() is not None and self.token() is not None

    def call(self, method: str, **data):
        token = self.token()
        if token is None:
            raise TelegramError('no token stored')
        try:
            request = Request(API.format(token=token, method=method),
                              json.dumps(data).encode(),
                              {'Content-Type': 'application/json'})
            with self.opener(request, timeout=TIMEOUT) as r:
                answer = json.load(r)
        except Exception as e:
            # (an error's own text could hold the address, with the token)
            raise TelegramError(f'{method}: {type(e).__name__} '
                                f'{getattr(e, "code", "")}') from None
        if not answer.get('ok'):
            raise TelegramError(f'{method}: refused')
        return answer['result']

    def send(self, text: str):
        self.call('sendMessage', chat_id=self.chat(), text=text)

    def pair(self, code: str, seconds: float = PAIR_SECONDS,
             clock=time.time, sleep=time.sleep) -> bool:
        """Wait for ``/start CODE``; that private chat gets the alerts."""
        end, offset = clock() + seconds, 0
        while clock() < end:
            for u in self.call('getUpdates', offset=offset, timeout=5):
                offset = u['update_id'] + 1
                m = u.get('message') or {}
                chat = m.get('chat') or {}
                if chat.get('type') == 'private' and secrets.compare_digest(
                        (m.get('text') or '').strip().encode(),
                        f'/start {code}'.encode()):
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    self.path.write_text(json.dumps({'chat': chat['id']}))
                    return True
            sleep(1)
        return False


class Alerts:
    """Queues messages and sends them from a thread of its own, so nothing
    waits for Telegram."""

    def __init__(self, telegram: Telegram, clock=time.time,
                 sleep=time.sleep):
        self.telegram = telegram
        self.clock = clock
        self.sleep = sleep
        self.said: Dict[str, float] = {}
        self.queue: 'queue.Queue[str]' = queue.Queue(maxsize=50)

    def say(self, text: str):
        now = self.clock()
        if now - self.said.get(text, -REPEAT_SECONDS) < REPEAT_SECONDS:
            return
        self.said[text] = now
        self.said = {t: at for t, at in self.said.items()
                     if now - at < REPEAT_SECONDS}
        try:
            self.queue.put_nowait(text)
        except queue.Full:
            logger.warning('too many alerts waiting')

    def deliver(self, text: str) -> bool:
        if not self.telegram.ready:
            return False
        for n in range(TRIES):
            try:
                self.telegram.send(text)
                return True
            except TelegramError as e:
                logger.warning(f'alert not sent: {e}')
                if n + 1 < TRIES:
                    self.sleep(TRY_AGAIN)
        return False

    def run(self):
        while True:
            try:
                self.deliver(self.queue.get())
            except Exception:
                logger.exception('alerts')

    def start(self):
        threading.Thread(target=self.run, daemon=True).start()


class Watch:
    """Turns what a watchdog and the coordination service's meetings show
    into alerts."""

    def __init__(self, say: Callable[[str], None], clock=time.time):
        self.say = say
        self.started = clock()
        self.trouble: Dict[str, bool] = {}
        self.meetings = set()

    def bot(self, role: str, state: str, text: str):
        """A watchdog state (called when it changes): 'alert' (needs a
        human), 'restart', 'ok', anything else is ignored."""
        if state in ('alert', 'restart'):
            self.trouble[role] = True
            self.say(f'{role.title()}: {text}' if state == 'restart' else
                     f'{role.title()} needs you: {text}')
        elif state == 'ok' and self.trouble.get(role):
            self.trouble[role] = False
            self.say(f'{role.title()}: {text}' if 'not watched' in text
                     else f'{role.title()} is working again')

    def met(self, meetings: List[dict]):
        for m in meetings:
            if m['state'] != 'aborted' or m['id'] in self.meetings:
                continue
            self.meetings.add(m['id'])
            if m['updated'] < self.started:
                continue
            why = '; '.join(f'{who}: {what}' for who, what in
                            (m.get('reasons') or {}).items())
            self.say(f'Meeting failed. {why}'.strip())


def main(argv=None):
    p = argparse.ArgumentParser(description='Telegram alerts of a bot farm')
    p.add_argument('what', choices=('pair', 'test', 'status'))
    p.add_argument('--farm', help='farm folder (default: as in farm.py)')
    p.add_argument('--folder', type=Path,
                   help="where the chat is kept (default: the farm's "
                        "state folder)")
    args = p.parse_args(argv)
    try:
        farm = Farm.load(args.farm)
    except NoFarm as e:
        if not args.folder:
            raise SystemExit(str(e))
        farm = Farm('Wizard Eyes', args.folder, args.folder)
    telegram = Telegram.for_farm(farm, args.folder)
    if telegram.token() is None:
        raise SystemExit(f'No token stored ({farm.telegram_credential}; '
                         f'see python -m wizard_eyes.botfarm.alerts -h)')
    if args.what == 'pair':
        code = f'{secrets.randbelow(10 ** 8):08d}'
        print(f'Send your bot this message: /start {code}', flush=True)
        print('paired' if telegram.pair(code) else 'nothing came')
    elif args.what == 'test':
        telegram.send(f'{farm.name}: test alert')
        print('sent')
    else:
        print('paired' if telegram.ready else 'not paired')


if __name__ == '__main__':
    main()
