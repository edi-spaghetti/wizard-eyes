"""Test transports (stairs, ladders...): use one and record where it goes.

Walks to the nearest object with the given name and option (as
``walk_test.py --object``), clicks the option, then expects a jump: the
plane changes or the position moves more than a few tiles at once, within a
few seconds. The outcome (object, where we stood, where we landed) is
appended to ``maps/transports/observed.json`` in the data directory, along
with what a simple guess would have predicted, to see how far guesses can be
trusted.

If no jump happens, the object is hovered and clicked once more; a second
miss gives up.

    venv\\Scripts\\python scripts\\transport_test.py --object Staircase --action Climb-up
"""
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from walk_test import Tile, WalkTest, chebyshev  # noqa: E402
from wizard_eyes.file_path_utils import get_root  # noqa: E402
from wizard_eyes.navigation.object_map import MapObject  # noqa: E402

OBSERVED = 'data/maps/transports/observed.json'

PARTNER_ACTIONS = {
    # option used: options the object at the other end usually has
    'climb-up': re.compile('^climb(-down)?$', re.I),
    'climb-down': re.compile('^climb(-up)?$', re.I),
}


class TransportTest(WalkTest):

    JUMP_TILES = 3
    """A move of more than this many tiles at once counts as a transport."""

    JUMP_SECONDS = 8.0
    """Time allowed from the click to landing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.used_from: Optional[Tile] = None
        self.use_attempts = 0
        self.guess: Optional[Tile] = None

    def setup(self):
        super().setup()
        if not self.args.object:
            raise SystemExit('transport_test needs --object (and --action)')

    def guess_destination(self, obj: MapObject, pos: Tile,
                          action: str) -> Optional[Tile]:
        """Simple guess: a matching object on the floor above / below (or
        about 6400 tiles north for "down" from the surface) within a few
        tiles; land on its usable tile closest to where we stood."""
        action_l = action.lower()
        partner = PARTNER_ACTIONS.get(action_l)
        if partner is None:
            return None
        x, y, z = obj.tile
        if action_l == 'climb-up':
            spots = [(x, y, z + 1)]
            if y > 6400:
                spots.append((x, y - 6400, 0))
        else:
            spots = [(x, y, z - 1)] if z > 0 else [(x, y + 6400, 0)]
        for spot in spots:
            others = self.om.find(action=partner, near=spot, radius=5)
            for other in others:
                tiles = self.om.interaction_tiles(other, self.cm)
                if tiles:
                    return min(tiles, key=lambda t: chebyshev(
                        t, (pos[0], pos[1] + spot[1] - y, spot[2])))
        return None

    def use_object(self, pos: Tile):
        """Hover and click the object (base class); once clicked, the
        jump check takes over (see :meth:`action`)."""
        super().use_object(pos)
        if self.obj_clicked_at:
            self.used_from = pos
            self.use_attempts += 1
            action = self.args.action or self.obj.actions[0]
            self.guess = self.guess_destination(self.obj, pos, action)
            self.log.info(f'Transport test: expect a jump within '
                          f'{self.JUMP_SECONDS:.0f}s (guess: {self.guess})')

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
        o = self.obj
        entry = {
            'object_id': o.id,
            'name': o.name,
            'action': action,
            'object_tile': list(o.tile),
            'from': list(self.used_from),
            'to': list(landed),
            'guess': list(self.guess) if self.guess else None,
            'guess_off_by': (chebyshev(self.guess, landed)
                             if self.guess and self.guess[2] == landed[2]
                             else None),
            'seconds': round(time.time() - self.obj_clicked_at, 1),
            'recorded_at': datetime.now(timezone.utc).isoformat(
                timespec='seconds'),
        }
        path = os.path.join(get_root(), OBSERVED)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = []
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
        data.append(entry)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)
        self.log.info(f'Transport test: {o.name!r} #{o.id} {action}: '
                      f'{self.used_from} -> {landed} (guess {self.guess}, '
                      f'off by {entry["guess_off_by"]}); recorded in {path}')


def main():
    app = TransportTest()
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
