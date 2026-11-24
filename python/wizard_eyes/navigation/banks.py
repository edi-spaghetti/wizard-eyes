"""Getting (back) to a bank, from the map data (``map_labels``).

    trip = BankTrip(app, ['strength_cape'])   # teleports the script may use
    ...  # every frame, after updating gauges, game_screen, mouse_options
    trip.update()
    result = trip.step()   # None: on the way, True: in view, False: gave up
    trip.entity            # the bank, once in view
    trip.seconds_left      # how long the game may take to walk us to it
"""
import random
import time
from typing import Dict, List, Optional, Sequence

from ..game_entities.entity import GameEntity
from ..house_teleport import HouseTeleport
from ..icon_teleport import IconTeleport
from ..interactions import UseOption
from ..item_teleport import ItemTeleport
from ..minigame_teleport import MinigameTeleport
from ..teleports import Method, Teleports
from . import map_labels
from .collision_map import CollisionMap, Tile
from .object_map import WALL_TYPES, MapObject, ObjectMap
from .walker import InView, Walker, chebyshev

BOOTH_RADIUS = 12
"""Banks further than this aren't on screen."""
SEARCH_RADIUS = 100
NEAR_STEPS = 25
"""A bank no further than this is walked to rather than teleporting."""
MAX_STEPS = 200
SAME_PLACE = 30
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


def make_teleport(app, method: Method
                  ):
    if method.minigame:
        return MinigameTeleport(app, method)
    if method.needs == 'house':
        return HouseTeleport(app, method)
    if method.icon:
        return IconTeleport(app, method)
    return ItemTeleport(app, method)


class Climb:
    """Use an object next to us (a ladder, stairs) and check where it takes
    us: ``{"object", "option", "tile": [x, y, z], "lands": [x, y, z, r]}``."""

    SECONDS = 8.0
    SETTLE = 1.5

    def __init__(self, app, what: dict):
        self.app = app
        self.client = app.client
        self.what = what
        self.entity = app.client.game_screen.create_game_entity(
            what['object'], what['object'], (0, 0), app.client, app.client)
        self.entity.set_global_coordinates(*what['tile'])
        self.use = UseOption(app, self.entity, what['option'],
                             [what['object']])
        self.used_at = 0.0
        self.there_since = 0.0
        self.failed: Optional[str] = None

    def update(self):
        if not self.used_at:
            self.app._update_game_entities(self.entity)

    def step(self) -> Optional[bool]:
        now = time.time()
        if not self.used_at:
            result = self.use.step()
            if result is False:
                self.failed = (f'no {self.what["option"]!r} on the '
                               f'{self.what["object"]}')
                return False
            if result:
                self.used_at = now
            return None
        x, y, z = self.client.gauges.minimap.gps.get_coordinates()
        tx, ty, tz, r = self.what['lands']
        if z == tz and max(abs(x - tx), abs(y - ty)) <= r:
            self.there_since = self.there_since or now
            return True if now - self.there_since > self.SETTLE else None
        if now - self.used_at > self.SECONDS:
            self.failed = (f'{self.what["option"]} on the '
                           f'{self.what["object"]}: at {(x, y, z)}')
            return False
        return None


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
        self.tele = None
        self.method: Optional[Method] = None
        self.then: List[dict] = []
        self.climb: Optional[Climb] = None
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
        # not to where we are
        self.methods = [
            m for m in self.methods if not any(
                t[2] == pos[2] and chebyshev(pos, t) <= SAME_PLACE
                for t in m.lands['tiles'])]
        found = path_to_bank(pos, self.om, self.cm, self.walls)
        steps = len(found[1]) - 1 if found else None
        if self.methods and not self.teleported and (
                steps is None or steps > NEAR_STEPS):
            method = self.methods.pop(0)
            self.client.logger.info(
                f'Bank trip: {method.key} from {pos} (walking: {steps})')
            self.method = method
            self.tele = make_teleport(self.app, method)
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
        elif self.climb is not None:
            self.climb.update()
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
                self.then = list(self.method.then) if result else []
            return None
        if self.climb is not None or self.then:
            if self.climb is None:
                self.climb = Climb(self.app, self.then.pop(0))
            result = self.climb.step()
            if result is False:
                self.failed = self.climb.failed
                return False
            if result:
                self.climb = None
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
