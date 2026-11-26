"""A farm's web page (127.0.0.1 only, plus a keyed port for the phone).

Every farm gets this much from the engine, with nothing of its own:

- ``/``: the page (``page.html`` here, or the farm's own)
- ``/control`` (GET): per role, working? (its own report or the engine's
  heartbeat), its heartbeat, the control script's last lines, the watchdog's
  say; (POST ``{"role", "action"}``): start / restart / stop a bot
- ``/live/ROLE``: a still of the bot's VM screen
- with ``--watchdog``: bots that stop getting on are restarted
  (``wizard_eyes.botfarm.watchdog``), Telegram alerts if set up

A farm with its own page (e.g. a planner) makes a :class:`FarmPage` with
its HTML and extra routes (``get`` / ``post``); one without runs this::

    python -m wizard_eyes.botfarm.page --service http://127.0.0.1:8765 \\
        --control PATH\\botfarm\\bot-control.ps1 --watchdog

(``farm.json`` services: ``"module": "wizard_eyes.botfarm.page"``.)
"""
import argparse
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

from ..coordination import Coordinator, CoordinatorUnavailable, last_seen
from .alerts import Alerts, Telegram, Watch
from .control import Control
from .farm import Farm
from .live import Live, NoPicture
from .remote import REMOTE_PORT, Remote
from .watchdog import Watchdog, parse_look

logger = logging.getLogger('wizard_eyes.botfarm.page')

DEFAULT_PORT = 8766
MAX_BODY = 100_000
PAGE = Path(__file__).with_name('page.html')

GetRoute = Callable[[], Tuple[bytes, str]]
"""-> (body, content type)"""
PostRoute = Callable[[bytes], bytes]
"""request body -> JSON reply; ValueError (and the like) -> 400"""
BAD_REQUEST = (ValueError, KeyError, TypeError, AttributeError,
               CoordinatorUnavailable)


class FarmPage:
    """What a farm's page serves. ``get`` / ``post``: the farm's own routes
    (path -> function), next to the engine's."""

    def __init__(self, name: str, roles: Iterable[str],
                 service: Optional[str] = None,
                 control: Optional[Control] = None,
                 watchdog: Optional[Watchdog] = None,
                 live: Optional[Live] = None,
                 html: Path = PAGE,
                 get: Optional[Dict[str, GetRoute]] = None,
                 post: Optional[Dict[str, PostRoute]] = None,
                 folder: Optional[Path] = None):
        self.name = name
        self.roles = tuple(roles)
        self.service = service
        self.control = control or Control(None, folder or Path('.'),
                                          self.roles)
        self.watchdog = watchdog
        self.live = live
        self.html = Path(html)
        self.get = dict(get or {})
        self.post = dict(post or {})

    def page(self) -> bytes:
        text = self.html.read_text('utf-8')
        return text.replace('{{FARM}}', self.name).encode('utf-8')

    def bots(self) -> Dict[str, dict]:
        if not self.service:
            return {}
        try:
            return Coordinator(self.service, 'page').bots()
        except CoordinatorUnavailable:
            return {}

    def control_status(self) -> dict:
        bots = self.bots()
        status = self.control.status(
            {r: last_seen(b) for r, b in bots.items()}, time.time())
        status['farm'] = self.name
        for role, s in status['bots'].items():
            s['heartbeat'] = (bots.get(role) or {}).get('heartbeat')
        if self.watchdog is not None:
            for role, w in self.watchdog.status().items():
                if role in status['bots']:
                    status['bots'][role]['watch'] = w
        return status

    def run_control(self, body: bytes) -> bytes:
        ask = json.loads(body)
        self.control.run(ask['role'], ask['action'])
        return json.dumps(self.control_status()).encode()


def make_handler(page: FarmPage, remote: Optional[Remote] = None):
    """``remote``: for the port the phone uses, where every request needs the
    key (its own port, as a request's Host can be made up)."""

    class Handler(BaseHTTPRequestHandler):
        timeout = 30

        def log_message(self, fmt, *args):
            logger.debug(fmt, *args)

        def reply(self, code: int, body: bytes, kind='application/json',
                  **headers):
            self.send_response(code)
            self.send_header('Content-Type', kind if kind.startswith('image/')
                             else f'{kind}; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Frame-Options', 'DENY')
            for name, value in headers.items():
                self.send_header(name.replace('_', '-'), value)
            self.end_headers()
            self.wfile.write(body)

        def fail(self, code: int, message: str):
            self.reply(code, json.dumps({'error': message}).encode())

        def allowed(self) -> bool:
            """Only this page: another site open in the browser can reach
            127.0.0.1 too."""
            if remote is not None:
                return remote.host_ok(self.headers.get('Host'),
                                      self.headers.get('Origin'))
            if 'X-Forwarded-For' in self.headers:
                return False    # (a proxy, on the port without a key)
            port = self.server.server_address[1]
            hosts = (f'127.0.0.1:{port}', f'localhost:{port}')
            origin = self.headers.get('Origin')
            return (self.headers.get('Host') in hosts and
                    (origin is None or
                     origin in [f'http://{h}' for h in hosts]))

        def has_key(self) -> bool:
            return remote is None or \
                remote.key_ok(remote.key_in(self.headers))

        def log_in(self, body: bytes):
            key = remote.key_in_form(body)
            if not remote.key_ok(key):
                logger.warning('phone page: wrong key')
                time.sleep(1)
                return self.reply(401, remote.login_page, 'text/html')
            self.reply(303, b'', Location='/', Set_Cookie=remote.cookie(key))

        def do_GET(self):
            if not self.allowed():
                return self.fail(403, 'wrong host')
            if not self.has_key():
                if self.path == '/':
                    return self.reply(401, remote.login_page, 'text/html')
                return self.fail(401, 'key needed')
            path = self.path.split('?')[0]
            if path == '/':
                self.reply(200, page.page(), 'text/html')
            elif path == '/control':
                self.reply(200, json.dumps(page.control_status()).encode())
            elif path.startswith('/live/'):
                self.live_picture(path[len('/live/'):])
            elif path in page.get:
                body, kind = page.get[path]()
                self.reply(200, body, kind)
            else:
                self.fail(404, 'not found')

        def live_picture(self, role: str):
            if page.live is None:
                return self.fail(404, 'no live pictures')
            try:
                jpeg = page.live.picture(role)
            except ValueError as e:
                return self.fail(404, str(e))
            except NoPicture as e:
                return self.fail(503, str(e))
            self.reply(200, jpeg, 'image/jpeg')

        def do_POST(self):
            try:
                length = int(self.headers.get('Content-Length', 0))
            except ValueError:
                return self.fail(400, 'bad length')
            if not 0 <= length <= MAX_BODY:
                return self.fail(413, 'too big')
            body = self.rfile.read(length)
            if remote is not None and self.allowed():
                if self.path == '/login':
                    return self.log_in(body)
                if not self.has_key():
                    return self.fail(401, 'key needed')
            if not self.allowed() or self.headers.get_content_type() != \
                    'application/json':
                return self.fail(403, 'not from the farm page')
            route = page.run_control if self.path == '/control' else \
                page.post.get(self.path)
            if route is None:
                return self.fail(404, 'not found')
            try:
                out = route(body)
            except BAD_REQUEST as e:
                return self.fail(400, str(e))
            except OSError as e:
                return self.fail(500, str(e))
            self.reply(200, out)

    return Handler


def serve(page: FarmPage, port: int,
          remote: Optional[Remote] = None) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(('127.0.0.1', port),
                                 make_handler(page, remote))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def make_watchdog(farm: Farm, folder: Path, service: str, control: Control,
                  **kwargs) -> Watchdog:
    """The farm's watchdog, with its Telegram alerts (silent until they're
    set up, see ``wizard_eyes.botfarm.alerts``). ``kwargs``: the farm's
    own ``progress`` / ``no_cure`` / ``meetings``."""
    client = Coordinator(service, 'watchdog')
    alerts = Alerts(Telegram.for_farm(farm, folder))
    alerts.start()
    kwargs.setdefault('meetings', lambda: client._call('GET', '/meetings'))
    return Watchdog(control, folder, farm.roles, client.bots,
                    lambda role, shot: parse_look(control.look(role, shot)),
                    watch=Watch(alerts.say), **kwargs)


def add_args(p: argparse.ArgumentParser, port: int = DEFAULT_PORT):
    """The options every farm page takes."""
    p.add_argument('--port', type=int, default=port)
    p.add_argument('--remote-port', type=int, default=REMOTE_PORT,
                   help='port for tailscale serve, key needed '
                        '(see wizard_eyes.botfarm.remote; 0: none)')
    p.add_argument('--service', help='the coordination service (bots\' '
                                     'reports and heartbeats)')
    p.add_argument('--control', help='script that starts / stops a bot '
                                     '(see wizard_eyes.botfarm.control)')
    p.add_argument('--watchdog', action='store_true',
                   help='restart bots that stop getting on (needs --service '
                        'and --control)')
    p.add_argument('--folder', type=Path,
                   help="state folder (default: the farm's state_dir)")
    p.add_argument('--farm', help='farm folder (default: $WIZARD_EYES_FARM, '
                   'set for the farm services, or the machine default)')


def serve_both(page: FarmPage, args, farm: Farm, folder: Path):
    """The page on ``--port``, and with the key on ``--remote-port``."""
    serve(page, args.port)
    if args.remote_port:
        serve(page, args.remote_port, Remote(folder, farm.name))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    add_args(p)
    args = p.parse_args(argv)
    if args.watchdog and not (args.service and args.control):
        p.error('--watchdog needs --service and --control')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    farm = Farm.load(args.farm)
    folder = args.folder or farm.state_dir
    folder.mkdir(parents=True, exist_ok=True)
    control = Control(args.control, folder, farm.roles)
    watchdog = make_watchdog(farm, folder, args.service, control) \
        if args.watchdog else None
    page = FarmPage(farm.name, farm.roles, args.service, control, watchdog,
                    Live(folder, farm.vms), folder=folder)
    serve_both(page, args, farm, folder)
    if watchdog is not None:
        watchdog.start()
    logger.info(f'{farm.name} page on http://127.0.0.1:{args.port}')
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
