"""Who may use a farm's page from elsewhere (the phone).

A farm page answers on 127.0.0.1 only; ``tailscale serve`` passes requests
from the phone to a second port (``REMOTE_PORT`` by default), where every
request needs the secret key and the host name set here. Nothing set: all
refused. In a terminal of your own (the key is shown once, only its hash is
kept; a new key shuts out everything that used the old)::

    python -m wizard_eyes.botfarm.remote host NAME.TAILNET.ts.net
    python -m wizard_eyes.botfarm.remote key
    python -m wizard_eyes.botfarm.remote off

Settings are in the farm's state folder (``--farm DIR`` / ``--folder DIR`` as
in ``farm.py``); a running page sees changes at once. The page asks for the
key and keeps it in a cookie; programs send ``Authorization: Bearer KEY``.
"""
import argparse
import hashlib
import hmac
import html
import json
import re
import secrets
from http.cookies import CookieError, SimpleCookie
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs

from .farm import state_folder

REMOTE_PORT = 8767
SETTINGS = 'remote.local.json'
COOKIE_SECONDS = 400 * 24 * 3600
LETTERS = 'abcdefghjkmnpqrstuvwxyz23456789'
"""(Nothing that's easily mistaken when typed on a phone.)"""
KEY_LENGTH = 24

LOGIN_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title><style>:root{{color-scheme:dark}}
body{{margin:0;background:#151a15;color:#e4eae0;font:17px system-ui,sans-serif}}
form{{max-width:340px;margin:20vh auto;padding:16px;display:grid;gap:12px}}
input,button{{font:inherit;padding:10px;border-radius:6px;
border:1px solid #343d33;background:#1e251e;color:inherit}}</style></head><body>
<form method="post" action="/login"><label for="key">{title} key</label>
<input id="key" name="key" type="password" autocomplete="current-password"
autocapitalize="none" required autofocus><button>Open</button></form>
</body></html>"""


def new_key() -> str:
    k = ''.join(secrets.choice(LETTERS) for _ in range(KEY_LENGTH))
    return '-'.join(k[i:i + 4] for i in range(0, KEY_LENGTH, 4))


def tidy(key: str) -> str:
    return re.sub(r'[\s-]', '', key).lower()


def hashed(key: str) -> str:
    # (random, 118 bits: a plain hash is enough)
    return hashlib.sha256(tidy(key).encode()).hexdigest()


def cookie_name(name: str) -> str:
    return '__Host-' + (re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')
                        or 'farm')


class Remote:
    """``name``: the farm's name, shown on the login page; the cookie is
    named after it."""

    def __init__(self, folder: Path, name: str = 'Wizard Eyes'):
        self.path = Path(folder) / SETTINGS
        self.name = name
        self.cookie_name = cookie_name(name)
        self.login_page = LOGIN_PAGE.format(
            title=html.escape(name)).encode()

    def settings(self) -> dict:
        try:
            got = json.loads(self.path.read_text('utf-8'))
        except (OSError, ValueError):
            return {}
        return got if isinstance(got, dict) else {}

    def change(self, **values):
        new = {**self.settings(), **values}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(new, indent=1), 'utf-8')
        tmp.replace(self.path)

    def host_ok(self, host: Optional[str], origin: Optional[str]) -> bool:
        """Like a local page: only its own name, only its own pages."""
        name = self.settings().get('host')
        return bool(name) and host is not None and host.lower() == name and \
            (origin is None or origin == f'https://{name}')

    def key_ok(self, key: Optional[str]) -> bool:
        want = self.settings().get('key_sha256')
        return bool(want and key) and hmac.compare_digest(hashed(key), want)

    def key_in(self, headers) -> Optional[str]:
        auth = headers.get('Authorization') or ''
        if auth.startswith('Bearer '):
            return auth[len('Bearer '):]
        try:
            cookie = SimpleCookie(headers.get('Cookie') or '').get(
                self.cookie_name)
        except CookieError:
            return None
        return cookie.value if cookie else None

    @staticmethod
    def key_in_form(body: bytes) -> str:
        return parse_qs(body.decode('utf-8', 'replace')).get('key', [''])[0]

    def cookie(self, key: str) -> str:
        # (Lax: Strict isn't sent when a link in another app opens the page)
        return (f'{self.cookie_name}={tidy(key)}; Path=/; '
                f'Max-Age={COOKIE_SECONDS}; Secure; HttpOnly; SameSite=Lax')


def main(argv=None):
    p = argparse.ArgumentParser(description='A farm page from the phone')
    p.add_argument('what', choices=('host', 'key', 'off', 'status'))
    p.add_argument('name', nargs='?', help='host name (for "host")')
    p.add_argument('--farm', help='farm folder (default: as in farm.py)')
    p.add_argument('--folder', type=Path,
                   help="where the settings are kept (default: the farm's "
                        "state folder)")
    args = p.parse_args(argv)
    remote = Remote(state_folder(args.folder, args.farm))
    if args.what == 'host':
        name = (args.name or '').lower().rstrip('.')
        if not re.fullmatch(r'[a-z0-9-]+(\.[a-z0-9-]+)+', name):
            p.error('host needs a name like motherbear.example.ts.net')
        remote.change(host=name)
        print(f'host: {name}')
    elif args.what == 'key':
        key = new_key()
        remote.change(key_sha256=hashed(key))
        print(f'New key (shown once): {key}')
    elif args.what == 'off':
        remote.change(key_sha256=None)
        print('No key is accepted any more')
    else:
        s = remote.settings()
        print(f"host: {s.get('host') or 'not set'}; "
              f"key: {'set' if s.get('key_sha256') else 'not set'}")


if __name__ == '__main__':
    main()
