"""A small always-on-top strip over the game client showing what the
application is doing (``--log-overlay``): its latest log lines.

Clicks go straight through it to the game, it never takes the focus, and the
application doesn't see it (it grabs the client's own window). The strip is
its own process, fed through a pipe: it can't slow the application down, and
goes when the application does.

    overlay = LogOverlay(client.title)
    overlay.write('a line')     # or a status line: "Time 0h 1m 2s 0.5 - ..."
"""
import logging
import os
import queue
import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import List, Optional

LINES = 5
WIDTH = 720
CHAR_WIDTH = 7
LINE_HEIGHT = 16
LEFT, TOP = 190, 34
"""From the window's corner: clear of the mouse-over text."""
RIGHT_MARGIN = 320
OPACITY = 0.72
HIDDEN = '200x40+-9000+-9000'
"""Off every screen (hiding and showing again would take the focus)."""
REFRESH_MS = 400
START_SECONDS = 1.5
"""For the strip to come up: the client is activated after that."""
CREATE_NO_WINDOW = 0x08000000
NO_ACTIVATE = 0x08000000

STATUS = re.compile(r'Time (\d+h \d+m \d+s) [\d.]+ ?-? ?(.*)$')


def tidy(line: str) -> Optional[str]:
    """What's worth showing of a line, None if nothing."""
    line = re.sub(r'\s+', ' ', line).strip()
    m = STATUS.match(line)
    if not m:
        return line or None
    clock, rest = m.groups()
    rest = re.sub(r'^world hop in: \S+ hours( - )?', '', rest)
    rest = re.sub(r'(- )?AFK: [\d.]+', '', rest)
    rest = re.sub(r':? [\d.]+$', '', rest).strip(' -')
    return f'{clock}  {rest}' if rest else None


class Lines:
    """The last different lines: a status that only changes its clock
    replaces the one before."""

    def __init__(self, n: int = LINES):
        self.lines = deque(maxlen=n)

    def add(self, line: str):
        line = tidy(line)
        if line is None:
            return
        if self.lines and self.lines[-1].split('  ', 1)[-1] == \
                line.split('  ', 1)[-1]:
            self.lines[-1] = line
        else:
            self.lines.append(line)

    def text(self, chars: int) -> str:
        return '\n'.join(i[:chars] for i in self.lines)


class LogOverlay(logging.Handler):
    """The application's side. Nothing here ever raises, or waits once it is up."""

    def __init__(self, title: str):
        super().__init__(logging.INFO)
        self.lines: 'queue.Queue[str]' = queue.Queue(maxsize=200)
        package = str(Path(__file__).resolve().parents[1])
        path = os.environ.get('PYTHONPATH')
        env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONPATH=(
            package + (os.pathsep + path if path else '')))
        self.process = subprocess.Popen(
            [sys.executable, '-m', 'wizard_eyes.log_overlay', title],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, env=env,
            creationflags=CREATE_NO_WINDOW)
        threading.Thread(target=self._feed, daemon=True).start()
        time.sleep(START_SECONDS)

    def _feed(self):
        try:
            while True:
                line = self.lines.get().replace('\n', ' ')
                self.process.stdin.write((line + '\n').encode('utf-8'))
                self.process.stdin.flush()
        except (OSError, ValueError):
            pass        # the strip is gone: carry on without

    def write(self, line: str):
        try:
            self.lines.put_nowait(line)
        except queue.Full:
            pass

    def emit(self, record):
        self.write(record.getMessage())


# ------------------------------------------------------------ the strip

def run(title: str):
    import ctypes
    import tkinter as tk

    import win32con
    import win32gui

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        pass
    lines, closed = Lines(), threading.Event()

    def feed():
        for raw in sys.stdin.buffer:
            lines.add(raw.decode('utf-8', 'replace'))
        closed.set()

    threading.Thread(target=feed, daemon=True).start()

    front = win32gui.GetForegroundWindow()
    window = tk.Tk()
    window.withdraw()
    window.title('log overlay')
    window.overrideredirect(True)
    window.attributes('-topmost', True, '-alpha', OPACITY)
    label = tk.Label(window, bg='black', fg='#9fe89f', justify='left',
                     anchor='nw', font=('Consolas', 9), padx=6, pady=3)
    label.pack(fill='both', expand=True)
    window.geometry(HIDDEN)
    window.update_idletasks()
    handle = ctypes.windll.user32.GetParent(window.winfo_id())
    style = win32gui.GetWindowLong(handle, win32con.GWL_EXSTYLE)
    win32gui.SetWindowLong(
        handle, win32con.GWL_EXSTYLE,
        style | win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT
        | win32con.WS_EX_TOOLWINDOW | NO_ACTIVATE)
    window.deiconify()
    window.update()
    # showing it can still take the focus: give it back, or the
    # application's keys would come here
    if front and win32gui.GetForegroundWindow() != front:
        try:
            win32gui.SetForegroundWindow(front)
        except win32gui.error:
            pass

    def place():
        target = win32gui.FindWindow(None, title)
        if not target or win32gui.IsIconic(target):
            window.geometry(HIDDEN)
            return
        x1, y1, x2, _ = win32gui.GetWindowRect(target)
        width = min(WIDTH, max(200, x2 - x1 - LEFT - RIGHT_MARGIN))
        label.config(text=lines.text(width // CHAR_WIDTH))
        window.geometry(f'{width}x{LINES * LINE_HEIGHT + 8}'
                        f'+{x1 + LEFT}+{y1 + TOP}')

    def tick():
        if closed.is_set():
            window.destroy()
            return
        try:
            place()
        except win32gui.error:
            pass        # the client is just going away
        window.after(REFRESH_MS, tick)

    tick()
    window.mainloop()


if __name__ == '__main__':
    run(sys.argv[1])
