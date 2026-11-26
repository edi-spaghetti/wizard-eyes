"""Pictures of the bots' VMs for a farm page (a still about once a second).

``VBoxManage controlvm VM screenshotpng`` reads the VM's screen without
touching it (no input, the bot carries on), with or without a window. The
picture is made smaller and sent as a JPEG; viewers asking within
``FRESH_SECONDS`` of each other share one picture.
"""
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Tuple

import cv2
import numpy

VBOXMANAGE = r'C:\Program Files\Oracle\VirtualBox\VBoxManage.exe'
FRESH_SECONDS = 0.8
WIDTH = 960
QUALITY = 65
TIMEOUT = 10
CREATE_NO_WINDOW = 0x08000000


class NoPicture(Exception):
    """The VM is off, or VirtualBox didn't give a picture."""


def vbox_png(vm: str, path: Path) -> bytes:
    done = subprocess.run(
        [VBOXMANAGE, 'controlvm', vm, 'screenshotpng', str(path)],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE, timeout=TIMEOUT,
        creationflags=CREATE_NO_WINDOW)
    if done.returncode != 0:
        why = done.stderr.decode('utf-8', 'replace')
        raise NoPicture('VM is off' if 'not currently running' in why
                        else 'no picture from VirtualBox')
    try:
        return path.read_bytes()
    finally:
        path.unlink(missing_ok=True)


def to_jpeg(png: bytes, width: int = WIDTH, quality: int = QUALITY) -> bytes:
    img = cv2.imdecode(numpy.frombuffer(png, numpy.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise NoPicture('unreadable picture')
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, round(h * width / w)),
                         interpolation=cv2.INTER_AREA)
    ok, data = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise NoPicture('could not make a JPEG')
    return data.tobytes()


class Live:
    """``vms``: role -> VM name (``Farm.vms``); ``folder``: for the
    VirtualBox file, deleted straight after."""

    def __init__(self, folder: Path, vms: Dict[str, str],
                 grab: Callable[[str, Path], bytes] = vbox_png,
                 clock: Callable[[], float] = time.monotonic):
        self.folder = Path(folder)
        self.vms = {role: vm for role, vm in vms.items() if vm}
        self.grab = grab
        self.clock = clock
        self.locks = {role: threading.Lock() for role in self.vms}
        self.last: Dict[str, Tuple[float, bytes]] = {}

    def picture(self, role: str) -> bytes:
        """A JPEG; ValueError for an unknown role, NoPicture if none."""
        if role not in self.vms:
            raise ValueError('unknown role')
        with self.locks[role]:     # (one VBoxManage per VM at a time)
            got = self.last.get(role)
            if got and self.clock() - got[0] < FRESH_SECONDS:
                return got[1]
            path = self.folder / f'live-{role}.local.png'
            try:
                jpeg = to_jpeg(self.grab(self.vms[role], path))
            except (OSError, subprocess.SubprocessError) as e:
                raise NoPicture(f'no picture ({type(e).__name__})')
            self.last[role] = (self.clock(), jpeg)
            return jpeg
