"""Entering the Bank PIN from Windows Credential Manager.

The PIN is never stored in the repo, the remote assets or any file the
engine writes. The user stores it once, per account, in Windows Credential
Manager (encrypted under their Windows login), by typing it at the prompt
of::

    cmdkey /generic:wizard-eyes/bank-pin/<account> /user:<account> /pass

(run in their own terminal; ``/pass`` with no value asks for it without
showing it). ``<account>`` is the name RuneLite shows in its title bar.

The engine reads it only when the PIN prompt is showing, types the digits
on the numpad (the prompt accepts keyboard input, so the shuffled buttons
don't matter) and drops it. It is never logged.

Expectation: the prompt is gone within a few seconds of the last digit.
If not, :class:`BankPinEntry` reports failure and must not be retried
automatically (repeated wrong PINs lock the bank).
"""

import random
import re
import time
from typing import Optional

import win32api
import win32con
import win32cred
import pywintypes

TARGET_PREFIX = 'wizard-eyes/bank-pin/'
CLOSE_SECONDS = 5.0


def account_from_title(title: str) -> Optional[str]:
    """'RuneLite - Afantii' -> 'Afantii'."""
    m = re.match(r'^RuneLite\s*-\s*(.+)$', title or '')
    return m.group(1).strip() if m else None


def store_command(account: str) -> str:
    return f'cmdkey /generic:{TARGET_PREFIX}{account} /user:{account} /pass'


def _read(account: str) -> Optional[str]:
    try:
        cred = win32cred.CredRead(TARGET_PREFIX + account,
                                  win32cred.CRED_TYPE_GENERIC)
    except pywintypes.error:
        return None
    blob = cred.get('CredentialBlob') or b''
    try:
        pin = blob.decode('utf-16-le')
    except UnicodeDecodeError:
        pin = blob.decode('latin-1')
    pin = pin.strip('\x00').strip()
    return pin if re.fullmatch(r'\d{4}', pin) else None


def has_pin(account: str) -> bool:
    return _read(account) is not None


def _press_numpad(digit: str):
    vk = win32con.VK_NUMPAD0 + int(digit)
    win32api.keybd_event(vk, 0, 0, 0)
    time.sleep(random.uniform(0.05, 0.1))
    win32api.keybd_event(vk, 0, win32con.KEYEVENTF_KEYUP, 0)


class BankPinEntry:
    """Type the stored PIN once and check the prompt goes away.

    ``step()`` returns True when no PIN prompt is showing (typed it, or
    there was none), None while waiting for the prompt to close, and False
    on failure: no PIN stored for the account, or the prompt still showing
    after typing it. On False the bot should stop and ask the user.
    """

    def __init__(self, client, account: Optional[str] = None):
        self.client = client
        self.account = account or account_from_title(
            getattr(client, 'title', ''))
        self.typed_at: Optional[float] = None

    @property
    def log(self):
        return self.client.logger

    def step(self) -> Optional[bool]:
        bank = self.client.bank
        if self.typed_at is None:
            if not bank.pin_requested():
                return True
            if not self.account:
                self.log.error("Bank PIN prompt showing, but the account "
                               "name isn't known (window title "
                               f"{getattr(self.client, 'title', '')!r})")
                return False
            pin = _read(self.account)
            if pin is None:
                self.log.error(
                    f'Bank PIN prompt showing, and no PIN is stored for '
                    f'{self.account}. Enter it in the client, or store it '
                    f'(in your own terminal) with: '
                    f'{store_command(self.account)}')
                return False
            self.log.info(f'Entering the stored Bank PIN for {self.account}')
            for digit in pin:
                _press_numpad(digit)
                time.sleep(random.uniform(0.8, 1.3))
            del pin
            self.typed_at = time.time()
            return None
        if not bank.pin_requested():
            self.log.info('Bank PIN accepted')
            return True
        if time.time() - self.typed_at > CLOSE_SECONDS:
            self.log.error('The Bank PIN prompt is still showing after '
                           'typing the stored PIN: not trying again (wrong '
                           'PIN? check the stored one). Stopping.')
            return False
        return None
