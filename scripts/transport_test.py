"""Test transports (stairs, ladders...): use one and record where it goes.

Walks to the nearest object with the given name and option (as
``walk_test.py --object``), clicks the option, then expects a jump: the
plane changes or the position moves more than a few tiles at once, within a
few seconds. The outcome (object, where we stood, where we landed) is
recorded with :meth:`Transports.record` (local recordings file, see
``wizard_eyes.navigation.transports``) and compared with what was known
before: a recording, a manual entry or a guess.

If no jump happens, the object is hovered and clicked once more; a second
miss gives up.

    venv\\Scripts\\python scripts\\transport_test.py --object Staircase --action Climb-up
"""
import os
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from walk_test import Tile, WalkTest, chebyshev  # noqa: E402
from wizard_eyes.navigation.transports import (  # noqa: E402
    Transport, Transports,
)


class TransportTest(WalkTest):

    JUMP_TILES = 3
    """A move of more than this many tiles at once counts as a transport."""

    JUMP_SECONDS = 8.0
    """Time allowed from the click to landing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.used_from: Optional[Tile] = None
        self.use_attempts = 0
        self.known: Optional[Transport] = None
        self.transports: Optional[Transports] = None

    def setup(self):
        super().setup()
        if not self.args.object:
            raise SystemExit('transport_test needs --object (and --action)')
        # all guesses, so their accuracy can be checked too
        self.transports = Transports(min_confidence=0.0)
        self.log.info(f'Transport test: {len(self.transports)} transports '
                      f'known')

    def use_object(self, pos: Tile):
        """Hover and click the object (base class); once clicked, the
        jump check takes over (see :meth:`action`)."""
        super().use_object(pos)
        if self.obj_clicked_at:
            self.used_from = pos
            self.use_attempts += 1
            action = self.args.action or self.obj.actions[0]
            self.known = self.transports.get(self.obj.tile, action)
            k = self.known
            known = (f'{k.source} -> {k.to} +-{k.to_radius} (confidence '
                     f'{k.confidence})' if k else 'nothing known')
            self.log.info(f'Transport test: expect a jump within '
                          f'{self.JUMP_SECONDS:.0f}s ({known})')

    def action(self):
        if self.used_from is None:
            super().action()  # walk to the object and click it
            return
        # clicked: don't walk anywhere, just watch for the jump
        pos = self.position()
        if pos is None:
            self.msg.append('no position')
            return
        if pos != self.last_pos:
            self.last_pos = pos
            self.last_moved = time.time()
        self.wait_for_jump(pos)

    def wait_for_jump(self, pos: Tile):
        start = self.used_from
        jumped = pos[2] != start[2] or chebyshev(pos, start) > self.JUMP_TILES
        now = time.time()
        if jumped:
            # let the landing settle (a tick or two) before recording
            if now - self.last_moved < 1.2:
                self.msg.append(f'landed near {pos}, settling')
                return
            self.record(pos)
            self.stop()
            return
        if now - self.obj_clicked_at < self.JUMP_SECONDS:
            self.msg.append(f'waiting for jump from {start}')
            return
        # expectation failed: no jump
        self.log.warning(f'Transport test: expected a jump after clicking '
                         f'{self.obj.name!r}, still at {pos} after '
                         f'{self.JUMP_SECONDS:.0f}s (mouse text '
                         f'{self.client.mouse_options.state!r})')
        if self.use_attempts >= 2:
            self.give_up('no jump after 2 clicks')
            return
        if pos != start:
            self.give_up(f'moved to {pos} without jumping; not where we '
                         f'expected to be, not retrying')
            return
        self.log.info('Transport test: looking at the object again')
        self.used_from = None
        self.obj_clicked_at = 0.0
        self.obj_hovers = 0

    def record(self, landed: Tile):
        action = self.args.action or self.obj.actions[0]
        k = self.known
        if k is None:
            verdict = 'new'
        else:
            off = chebyshev(k.to, landed) if k.to[2] == landed[2] else None
            ok = off is not None and off <= k.to_radius
            verdict = (f'{k.source} {"matches" if ok else "WRONG"}: '
                       f'expected {k.to} +-{k.to_radius}, off by {off}')
        t = self.transports.record(
            self.obj, action, self.used_from, landed,
            seconds=round(time.time() - self.obj_clicked_at, 1))
        self.log.info(f'Transport test: {self.obj.name!r} #{self.obj.id} '
                      f'{action}: {self.used_from} -> {landed} ({verdict}); '
                      f'recorded (x{t.count}) in '
                      f'{self.transports.recorded_path}')


def main():
    app = TransportTest()
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
