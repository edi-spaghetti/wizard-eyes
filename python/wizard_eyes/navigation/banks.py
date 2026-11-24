"""Getting (back) to a bank, from the map data (``map_labels``).

    trip = BankTrip(app, ['strength_cape'])   # teleports the script may use
    ...  # every frame, after updating gauges, game_screen, mouse_options
    trip.update()
    result = trip.step()   # None: on the way, True: in view, False: gave up
    trip.entity            # the bank, once in view
    trip.seconds_left      # how long the game may take to walk us to it
"""
import random
from typing import Dict, Optional, Sequence

from ..game_entities.entity import GameEntity
from ..item_teleport import ItemTeleport
from ..teleports import Teleports
from . import map_labels
from .collision_map import CollisionMap, Tile
from .object_map import WALL_TYPES, MapObject, ObjectMap
from .walker import InView, Walker

BOOTH_RADIUS = 12
"""Banks further than this aren't on screen."""
SEARCH_RADIUS = 100
NEAR_STEPS = 25
"""A bank no further than this is walked to rather than teleporting."""
MAX_STEPS = 200
SPREAD = 4
"""The last click lands up to this many tiles short of the booth."""


def bank_entity(app, thing: map_labels.MapThing) -> GameEntity:
    """Game entity over a bank object or banker."""
    width, height = thing.size
    entity = app.client.game_screen.create_game_entity(
        'bank', 'bank', (0, 0), app.client, app.client,
        tile_width=width, tile_height=height,
    )
    entity.set_global_coordinates(*map_labels.anchor(thing))
    app._update_game_entities(entity)
    return entity


def nearest_bank(app, pos: Tile, radius: int = BOOTH_RADIUS,
                 walls: bool = False) -> Optional[tuple]:
    """The bank object or banker nearest to a position, as (thing, game
    entity), or None.

    :param walls: counters too (Grand Exchange): these need a click on their
        edge of the tile."""
    things = map_labels.find_banks(pos, radius, walls=walls)
    if not things:
        return None
    return things[0], bank_entity(app, things[0])


def path_to_bank(pos: Tile, om: ObjectMap, cm: CollisionMap,
                 walls: bool = False) -> Optional[tuple]:
    """The bank object with the shortest walk, as (object, path)."""
    goals: Dict[Tile, MapObject] = {}
    for thing in map_labels.find_banks(pos, SEARCH_RADIUS, walls=walls,
                                       om=om):
        if isinstance(thing, MapObject):
            for tile in om.interaction_tiles(thing, cm):
                goals.setdefault(tile, thing)
    if not goals:
        return None
    path = cm.find_path(pos, list(goals), max_nodes=200_000)
    if path is None or len(path) - 1 > MAX_STEPS:
        return None
    return goals[path[-1]], path


class BankTrip:
    """Go to a bank: walk if one is close, else use one of the teleports
    (in random order) and walk from there; without teleports, walk. Each
    teleport is tried once. Gives up if nothing gets us to a bank."""

    def __init__(self, app, teleports: Sequence[str] = (),
                 om: Optional[ObjectMap] = None,
                 cm: Optional[CollisionMap] = None,
                 catalogue: Optional[Teleports] = None,
                 rng: Optional[random.Random] = None, walls: bool = False):
        self.app = app
        self.client = app.client
        catalogue = catalogue or Teleports()
        usable = catalogue.usable()
        unknown = [t for t in teleports if t not in usable]
        if unknown:
            raise ValueError(f'Unknown or unsupported teleports {unknown}; '
                             f'usable: {usable}')
        self.methods = [catalogue.methods[t] for t in teleports]
        (rng or random.Random()).shuffle(self.methods)
        self.om = om or map_labels.object_map()
        self.cm = cm or CollisionMap()
        self.walls = walls
        self.tele: Optional[ItemTeleport] = None
        self.teleported = False
        self.obj: Optional[MapObject] = None
        self.walker: Optional[Walker] = None
        self._tabs_updated = False
        self._booth: Optional[GameEntity] = None
        self.entity: Optional[GameEntity] = None
        self.seconds_left = 0.0
        self.failed: Optional[str] = None

    def _position(self) -> Optional[Tile]:
        x, y, z = self.client.gauges.minimap.gps.get_coordinates()
        return None if x == -1 else (int(x), int(y), int(z))

    def choose(self, pos: Tile):
        """Walk or teleport from here."""
        found = path_to_bank(pos, self.om, self.cm, self.walls)
        steps = len(found[1]) - 1 if found else None
        if self.methods and not self.teleported and (
                steps is None or steps > NEAR_STEPS):
            method = self.methods.pop(0)
            self.client.logger.info(
                f'Bank trip: {method.key} from {pos} (walking: {steps})')
            self.tele = ItemTeleport(self.app, method)
            return
        if found is None:
            self.failed = f'no bank can be walked to from {pos}'
            self.client.logger.error(f'Bank trip: {self.failed}')
            return
        self.obj, path = found
        self._booth = bank_entity(self.app, self.obj)
        self.walker = Walker(
            self.app, [path[-1]], cm=self.cm, spread=SPREAD,
            done=InView(self.client, self._booth))
        self.client.logger.info(f'Bank trip: {self.obj}, {steps} steps')

    def update(self):
        if self.tele is not None:
            self.tele.update()
        elif self.walker is not None:
            if not self._tabs_updated:
                # the side panel can cover the booth; once is enough (it
                # doesn't move, and updating it takes about a second)
                self.client.tabs.update()
                self._tabs_updated = True
            self.walker.update()
            self.app._update_game_entities(self._booth)

    def step(self) -> Optional[bool]:
        if self.failed is not None:
            return False
        if self.entity is not None:
            return True
        if self.tele is not None:
            result = self.tele.step()
            if result is not None:
                # failed: try the next one, or walk
                self.teleported = result
                self.tele = None
            return None
        if self.walker is None:
            pos = self._position()
            if pos is None:
                self.app.msg.append('no position')
            else:
                self.choose(pos)
            return False if self.failed is not None else None
        result = self.walker.step()
        if result is False:
            self.failed = self.walker.failed
        elif result:
            self.entity = self._booth
            left = self.om.path_to(self.obj, self._position(), self.cm) or []
            self.seconds_left = len(left) * Walker.SECONDS_PER_TILE
        return result

    @property
    def wall(self) -> Optional[int]:
        """The side of its tile the booth is on (0 west, 1 north, 2 east,
        3 south), if it's a wall / counter."""
        return self.obj.orientation \
            if self.obj is not None and self.obj.type in WALL_TYPES else None
