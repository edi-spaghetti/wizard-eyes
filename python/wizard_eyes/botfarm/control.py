"""Starting and stopping a farm's bots from its page.

How is up to a script on the host (``botfarm/bot-control.ps1``); it's run as
``SCRIPT -Role ROLE -Action start|restart|stop -Log FILE`` and writes a line
per step to FILE, the last one starting with OK or FAILED.
"""
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ACTIONS = ('start', 'restart', 'stop')
WORKING_SECONDS = 180
"""A bot that reported this recently is working."""
SHOWN = 6
LOOK_SECONDS = 150
CREATE_NO_WINDOW = 0x08000000


class Control:
    """``script``: None = no control (the page only shows); ``roles``: the
    farm's roles (``Farm.roles``)."""

    def __init__(self, script: Optional[str], folder: Path,
                 roles: Iterable[str], popen=subprocess.Popen):
        self.script = script
        self.folder = Path(folder)
        self.roles = tuple(roles)
        self.popen = popen
        self.lock = threading.Lock()
        self.jobs: Dict[str, object] = {}
        self.what: Dict[str, dict] = {}

    def log(self, role: str) -> Path:
        return self.folder / f'control-{role}.local.log'

    def busy(self, role: str) -> bool:
        job = self.jobs.get(role)
        return job is not None and job.poll() is None

    def run(self, role, action):
        """Start a job; ValueError if it can't be."""
        if self.script is None:
            raise ValueError('no control script')
        if not isinstance(role, str) or role not in self.roles or \
                action not in ACTIONS:
            raise ValueError('unknown role or action')
        with self.lock:
            if self.busy(role):
                raise ValueError(f'busy with the {role}')
            self.log(role).write_text('')
            # no pipes: what it starts (a VM) would hold them open
            self.jobs[role] = self.popen(
                ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass',
                 '-File', self.script, '-Role', role, '-Action', action,
                 '-Log', str(self.log(role))],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW)
            self.what[role] = {'action': action, 'started': time.time()}

    def look(self, role: str, shot: Optional[Path] = None,
             run=subprocess.run) -> str:
        """The script's ``-Action status`` output (takes some seconds)."""
        if self.script is None:
            raise ValueError('no control script')
        args = ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass',
                '-File', self.script, '-Role', role, '-Action', 'status']
        if shot is not None:
            args += ['-Shot', str(shot)]
        return run(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                   stderr=subprocess.DEVNULL, timeout=LOOK_SECONDS,
                   creationflags=CREATE_NO_WINDOW
                   ).stdout.decode('utf-8', 'replace')

    def lines(self, role: str) -> List[str]:
        try:
            text = self.log(role).read_text('utf-8-sig', 'replace')
        except OSError:
            return []
        return [i for i in text.splitlines() if i.strip()][-SHOWN:]

    def status(self, updated: Dict[str, float], now: float) -> dict:
        """Per role, for the page. ``updated``: when each last reported."""
        out = {}
        for role in self.roles:
            age = now - updated[role] if updated.get(role) else None
            out[role] = dict(
                self.what.get(role, {}), busy=self.busy(role),
                lines=self.lines(role), age=age,
                working=age is not None and age < WORKING_SECONDS)
        return {'available': self.script is not None, 'bots': out}
