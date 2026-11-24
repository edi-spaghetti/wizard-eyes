"""Walk to the Grand Exchange and open it.

The booths are thin counters, so a clerk behind one (``config/npcs.json``) is
clicked instead, and only once we've come to a standstill: a click while the
camera still moves can miss."""
import time
from typing import List, Optional

from ..game_entities.entity import GameEntity
from ..grand_exchange import GrandExchange
from ..interactions import UseOption
from . import map_labels
from .collision_map import Tile
from .npc_map import MapNPC
from .walker import Walker

CLERK = 'grand_exchange_clerk'
REACH = 2
"""Closer to a clerk than this: no need to walk."""
MAX_WALK = 120
STILL_SECONDS = 1.3
"""Position unchanged for this long (two ticks): standing still."""
OPEN_SECONDS = 4.0
MAX_TRIES = 2


class OpenGE:
    """Call :meth:`update` every frame, then :meth:`step`: None while
    working on it, True once the GE's main screen is open, False: failed
    (``failed`` says why).

    Expectation: the main screen opens within ``OPEN_SECONDS`` of the click;
    tried again once."""

    def __init__(self, app):
        self.app = app
        self.client = app.client
        self.ge = GrandExchange(app.client)
        self.clerk: Optional[MapNPC] = None
        self.entity: Optional[GameEntity] = None
        self.walker: Optional[Walker] = None
        self.use: Optional[UseOption] = None
        self.clicked_at = 0.0
        self.tries = 0
        self.still_since = 0.0
        self.last_pos: Optional[Tile] = None
        self.failed: Optional[str] = None

    def position(self) -> Optional[Tile]:
        pos = tuple(int(i) for i in
                    self.client.gauges.minimap.gps.get_coordinates())
        return None if pos[0] == -1 else pos

    def clerks(self, pos: Tile) -> List[MapNPC]:
        return map_labels.npc_map().find(CLERK, pos, MAX_WALK)

    def update(self):
        if self.walker is not None:
            self.walker.update()
        if self.entity is not None:
            self.app._update_game_entities(self.entity)

    def _fail(self, why: str) -> bool:
        self.failed = why
        self.client.logger.warning(f'Open GE: {why}')
        return False

    def _standing_still(self, pos: Tile) -> bool:
        now = time.time()
        if pos != self.last_pos:
            self.last_pos, self.still_since = pos, now
        return now - self.still_since >= STILL_SECONDS

    def _target(self, clerk: MapNPC):
        self.clerk = clerk
        self.entity = self.client.game_screen.create_game_entity(
            'npc', 'npc', (0, 0), self.client, self.client,
            tile_width=1, tile_height=1)
        self.entity.set_global_coordinates(*clerk.tile)
        self.app._update_game_entities(self.entity)

    def step(self) -> Optional[bool]:
        if self.ge.update() == 'main':
            return True
        pos = self.position()
        if pos is None:
            return None
        still = self._standing_still(pos)

        if self.clicked_at:
            if time.time() - self.clicked_at < OPEN_SECONDS:
                return None
            self.clicked_at, self.use = 0.0, None
            if self.tries >= MAX_TRIES:
                return self._fail('the Grand Exchange did not open')

        if self.walker is None and self.clerk is None:
            clerks = self.clerks(pos)
            if not clerks:
                return self._fail(f'no clerk within {MAX_WALK} tiles')
            if clerks[0].distance(pos) > REACH:
                # the tiles in front of the booths
                goals = [t for o in map_labels.object_map().find(
                             'Grand Exchange booth', 'Exchange', near=pos,
                             radius=MAX_WALK)
                         for t in [o.tile] if any(
                             c.distance(t) <= REACH for c in clerks)]
                if not goals:
                    return self._fail('no booth near a clerk')
                self.walker = Walker(self.app, goals)
                self.walker.update()
        if self.walker is not None:
            result = self.walker.step()
            if result is None:
                return None
            walker, self.walker = self.walker, None
            if result is False:
                return self._fail(f'could not walk there: {walker.failed}')
            return None
        if not still:
            return None
        if self.clerk is None:
            clerk = self.clerks(pos)[0]
            if clerk.distance(pos) > REACH:
                return self._fail(f'ended up at {pos}, not near a clerk')
            self._target(clerk)
            return None
        if self.use is None:
            self.tries += 1
            self.use = UseOption(self.app, self.entity, 'Exchange',
                                 ['Grand Exchange Clerk'])
        result = self.use.step()
        if result is False:
            return self._fail('no "Exchange" on the clerk at '
                              f'{self.clerk.tile}')
        if result:
            self.clicked_at = time.time()
        return None
