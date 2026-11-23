"""A small coordination service so bots on different machines / VMs can
share their state, agree on shared values (e.g. prices) and arrange
meetings (e.g. to trade).

Nothing here knows about any particular bot: statuses, shared values and
the items in a meeting are whatever the bots put in them.

* :class:`Store`: the state and its rules, saved to a JSON file
* :func:`serve`: an HTTP server around a store (stdlib only)
* :class:`Coordinator`: the client a bot uses; short timeouts, and every
  failure is a :class:`CoordinatorUnavailable` so a bot can carry on alone

Run the server on the host::

    python -m wizard_eyes.coordination serve --port 8765

It binds to 127.0.0.1 by default. VirtualBox NAT guests reach the host's
127.0.0.1 at ``http://10.0.2.2:<port>`` when the VM has
``--nat-localhostreachable1 on``, so nothing is exposed on the network.

Meetings
--------
A meeting is proposed by one bot to another and moves through::

    proposed -> accepted -> (both arrive) -> met -> done
             -> declined
    any open state -> aborted (by either party, or expired)

``arrive`` / ``done`` are per party: the meeting is ``met`` once both have
arrived and ``done`` once both report done. A meeting whose ``expires``
time passes while still open becomes ``aborted`` ('expired'). Results
(``done``) and reasons (``abort``) are kept per party for the record.
"""
import argparse
import json
import logging
import os
import threading
import time
import uuid
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib import error as urlerror
from urllib import request as urlrequest

logger = logging.getLogger('wizard_eyes.coordination')

DEFAULT_PORT = 8765
OPEN = ('proposed', 'accepted', 'met')
FINAL = ('declined', 'done', 'aborted')
KEEP_FINAL_SECONDS = 7 * 24 * 3600


class CoordinationError(ValueError):
    """A request the store refuses (unknown meeting, wrong party, bad
    transition)."""


class Store:
    """Bots' statuses, shared values and meetings, optionally saved to
    ``path`` after every change. Thread safe."""

    def __init__(self, path: Optional[str] = None,
                 clock: Callable[[], float] = time.time):
        self.path = Path(path) if path else None
        self.clock = clock
        self.lock = threading.Lock()
        self.data = {'bots': {}, 'shared': {}, 'meetings': {}}
        if self.path and self.path.exists():
            loaded = json.loads(self.path.read_text())
            for k in self.data:
                self.data[k] = loaded.get(k, {})

    # ------------------------------------------------------------ helpers
    def _save(self):
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.data, indent=1))
        os.replace(tmp, self.path)

    def _expire(self):
        now = self.clock()
        meetings = self.data['meetings']
        for mid, m in list(meetings.items()):
            if m['state'] in OPEN and m.get('expires') and now > m['expires']:
                m['state'] = 'aborted'
                m['reasons']['_server'] = 'expired'
                m['updated'] = now
            elif (m['state'] in FINAL
                  and now - m['updated'] > KEEP_FINAL_SECONDS):
                del meetings[mid]

    # ------------------------------------------------------------- bots
    def set_status(self, bot: str, status: Dict[str, Any]) -> dict:
        with self.lock:
            entry = {'status': status, 'updated': self.clock()}
            self.data['bots'][bot] = entry
            self._save()
            return deepcopy(entry)

    def bots(self) -> Dict[str, dict]:
        with self.lock:
            return deepcopy(self.data['bots'])

    # ----------------------------------------------------------- shared
    def set_shared(self, key: str, value: Any, by: str) -> dict:
        with self.lock:
            entry = {'value': value, 'by': by, 'updated': self.clock()}
            self.data['shared'][key] = entry
            self._save()
            return deepcopy(entry)

    def shared(self) -> Dict[str, dict]:
        with self.lock:
            return deepcopy(self.data['shared'])

    # --------------------------------------------------------- meetings
    def propose(self, by: str, to: str, details: Dict[str, Any],
                expires_in: float) -> dict:
        if by == to:
            raise CoordinationError('a bot cannot meet itself')
        with self.lock:
            self._expire()
            now = self.clock()
            m = {'id': uuid.uuid4().hex[:12], 'by': by, 'to': to,
                 'state': 'proposed', 'details': details,
                 'created': now, 'updated': now,
                 'expires': now + expires_in,
                 'arrived': {}, 'results': {}, 'reasons': {}}
            self.data['meetings'][m['id']] = m
            self._save()
            return deepcopy(m)

    def meetings(self, bot: Optional[str] = None,
                 open_only: bool = False) -> List[dict]:
        with self.lock:
            self._expire()
            out = [m for m in self.data['meetings'].values()
                   if (bot is None or bot in (m['by'], m['to']))
                   and (not open_only or m['state'] in OPEN)]
            return deepcopy(sorted(out, key=lambda m: m['created']))

    def act(self, meeting_id: str, bot: str, action: str,
            info: Any = None) -> dict:
        """``accept`` / ``decline`` (the invitee, while proposed),
        ``arrive`` (either party, once accepted), ``done`` (either party,
        once met; ``info`` is its result), ``abort`` (either party, while
        open; ``info`` is the reason)."""
        with self.lock:
            self._expire()
            m = self.data['meetings'].get(meeting_id)
            if m is None:
                raise CoordinationError(f'no meeting {meeting_id}')
            if bot not in (m['by'], m['to']):
                raise CoordinationError(f'{bot} is not in meeting {meeting_id}')
            state = m['state']
            if action in ('accept', 'decline'):
                if bot != m['to'] or state != 'proposed':
                    raise CoordinationError(
                        f'{bot} cannot {action} a {state} meeting')
                m['state'] = 'accepted' if action == 'accept' else 'declined'
            elif action == 'arrive':
                if state not in ('accepted', 'met'):
                    raise CoordinationError(f'cannot arrive at a {state} meeting')
                m['arrived'][bot] = self.clock()
                if len(m['arrived']) == 2:
                    m['state'] = 'met'
            elif action == 'done':
                if state != 'met':
                    raise CoordinationError(f'cannot finish a {state} meeting')
                m['results'][bot] = info
                if len(m['results']) == 2:
                    m['state'] = 'done'
            elif action == 'abort':
                if state not in OPEN:
                    raise CoordinationError(f'cannot abort a {state} meeting')
                m['state'] = 'aborted'
                m['reasons'][bot] = info
            else:
                raise CoordinationError(f'unknown action {action!r}')
            m['updated'] = self.clock()
            self._save()
            return deepcopy(m)

    def snapshot(self) -> dict:
        with self.lock:
            self._expire()
            return deepcopy(self.data)


# ------------------------------------------------------------------ server

def make_handler(store: Store, token: Optional[str] = None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            logger.debug(fmt % args)

        def _reply(self, code: int, body: Any):
            raw = json.dumps(body).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _body(self) -> Any:
            n = int(self.headers.get('Content-Length') or 0)
            return json.loads(self.rfile.read(n) or b'null')

        def _route(self, method: str):
            if token and self.headers.get('X-Token') != token:
                return self._reply(403, {'error': 'bad token'})
            parts = [p for p in self.path.split('?')[0].split('/') if p]
            query = dict(q.split('=', 1) for q in
                         (self.path.split('?', 1)[1:] or [''])[0].split('&')
                         if '=' in q)
            try:
                body = self._body() if method in ('PUT', 'POST') else None
                if method == 'GET' and parts == ['state']:
                    return self._reply(200, store.snapshot())
                if method == 'GET' and parts == ['bots']:
                    return self._reply(200, store.bots())
                if method == 'PUT' and len(parts) == 2 and parts[0] == 'bots':
                    return self._reply(200, store.set_status(parts[1], body))
                if method == 'GET' and parts == ['shared']:
                    return self._reply(200, store.shared())
                if method == 'PUT' and len(parts) == 2 and parts[0] == 'shared':
                    return self._reply(200, store.set_shared(
                        parts[1], body['value'], body['by']))
                if method == 'GET' and parts == ['meetings']:
                    return self._reply(200, store.meetings(
                        query.get('bot'), query.get('open') == '1'))
                if method == 'POST' and parts == ['meetings']:
                    return self._reply(200, store.propose(
                        body['by'], body['to'], body.get('details') or {},
                        float(body.get('expires_in', 1800))))
                if (method == 'POST' and len(parts) == 3
                        and parts[0] == 'meetings'):
                    return self._reply(200, store.act(
                        parts[1], body['bot'], parts[2], body.get('info')))
            except CoordinationError as e:
                return self._reply(409, {'error': str(e)})
            except (KeyError, TypeError, ValueError) as e:
                return self._reply(400, {'error': f'bad request: {e}'})
            return self._reply(404, {'error': 'not found'})

        def do_GET(self):
            self._route('GET')

        def do_PUT(self):
            self._route('PUT')

        def do_POST(self):
            self._route('POST')

    return Handler


def serve(store: Store, host: str = '127.0.0.1', port: int = DEFAULT_PORT,
          token: Optional[str] = None) -> ThreadingHTTPServer:
    """Start serving in a background thread; returns the server (call
    ``shutdown()`` to stop). ``port=0`` picks a free port."""
    server = ThreadingHTTPServer((host, port), make_handler(store, token))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# ------------------------------------------------------------------ client

class CoordinatorUnavailable(RuntimeError):
    """The service couldn't be reached or refused the request."""


class Coordinator:
    """Client for one bot. Every call fails fast (``timeout``) with
    :class:`CoordinatorUnavailable`, so callers can fall back to working
    alone."""

    def __init__(self, url: str, bot: str, token: Optional[str] = None,
                 timeout: float = 3.0):
        self.url = url.rstrip('/')
        self.bot = bot
        self.token = token
        self.timeout = timeout

    def _call(self, method: str, path: str, body: Any = None) -> Any:
        data = None if body is None else json.dumps(body).encode()
        req = urlrequest.Request(self.url + path, data=data, method=method)
        req.add_header('Content-Type', 'application/json')
        if self.token:
            req.add_header('X-Token', self.token)
        try:
            with urlrequest.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urlerror.HTTPError as e:
            try:
                msg = json.loads(e.read()).get('error')
            except ValueError:
                msg = e.reason
            raise CoordinatorUnavailable(f'{method} {path}: {e.code} {msg}')
        except (urlerror.URLError, OSError, ValueError) as e:
            raise CoordinatorUnavailable(f'{method} {path}: {e}')

    # bots
    def post_status(self, status: Dict[str, Any]) -> dict:
        return self._call('PUT', f'/bots/{self.bot}', status)

    def bots(self) -> Dict[str, dict]:
        return self._call('GET', '/bots')

    # shared values
    def set_shared(self, key: str, value: Any) -> dict:
        return self._call('PUT', f'/shared/{key}',
                          {'value': value, 'by': self.bot})

    def shared(self) -> Dict[str, dict]:
        return self._call('GET', '/shared')

    # meetings
    def propose(self, to: str, details: Dict[str, Any],
                expires_in: float = 1800) -> dict:
        return self._call('POST', '/meetings', {
            'by': self.bot, 'to': to, 'details': details,
            'expires_in': expires_in})

    def meetings(self, open_only: bool = True) -> List[dict]:
        return self._call('GET', f'/meetings?bot={self.bot}'
                                 f'&open={int(open_only)}')

    def act(self, meeting_id: str, action: str, info: Any = None) -> dict:
        return self._call('POST', f'/meetings/{meeting_id}/{action}',
                          {'bot': self.bot, 'info': info})

    def accept(self, meeting_id: str) -> dict:
        return self.act(meeting_id, 'accept')

    def decline(self, meeting_id: str) -> dict:
        return self.act(meeting_id, 'decline')

    def arrive(self, meeting_id: str) -> dict:
        return self.act(meeting_id, 'arrive')

    def done(self, meeting_id: str, result: Any = None) -> dict:
        return self.act(meeting_id, 'done', result)

    def abort(self, meeting_id: str, reason: str) -> dict:
        return self.act(meeting_id, 'abort', reason)


def main(argv=None):
    from .file_path_utils import get_root
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('serve', help='run the coordination service')
    s.add_argument('--host', default='127.0.0.1')
    s.add_argument('--port', type=int, default=DEFAULT_PORT)
    s.add_argument('--state', default=str(
        Path(get_root()) / 'data' / 'coordination' / 'state.local.json'))
    s.add_argument('--token', default=os.environ.get('WIZARD_EYES_COORD_TOKEN'))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    server = serve(Store(args.state), args.host, args.port, args.token)
    logger.info(f'coordination service on http://{args.host}:{args.port} '
                f'(state {args.state})')
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == '__main__':
    main()
