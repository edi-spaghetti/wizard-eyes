"""Test traveller: walk to a tile using the cache-built collision map.

Plans a path with ``CollisionMap.find_path`` and follows it by clicking the
furthest path tile that's safely inside the minimap, re-clicking shortly
before reaching it. At a closed door it stops on the tile before, hovers the
door's game entity on the game screen and clicks only if the mouse text says
"Open"; it walks on only once the door shows "Close" (open).

Every click comes with an expectation of what happens next (see
:class:`Expectation`): the player reaches the clicked tile in time, without
straying from a shortest route there. When that fails the walker stops,
re-checks the situation (e.g. hovers the door again) before doing anything
else, and gives up after a couple of failures instead of retrying blindly.

With ``--object`` it walks to the nearest object of that name (by walking
distance, from the cache-built object index) instead, to a tile the object
can be used from, then hovers the object and clicks it once the mouse text
shows the option given with ``--action`` (default: its first option).

Needs the camera north / top-down (start-up checks do this) and the grid
info overlay for coordinates. Run::

    venv\\Scripts\\python scripts\\walk_test.py --to 3253 3266 --run-time 300
    venv\\Scripts\\python scripts\\walk_test.py --object "Bank booth" --action Bank
"""
import math
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from wizard_eyes.application import Application
from wizard_eyes.game_entities.entity import GameEntity
from wizard_eyes.navigation.collision_map import DOOR_COST, CollisionMap
from wizard_eyes.navigation.object_map import MapObject, ObjectMap, describe

Tile = Tuple[int, int, int]
DoorKey = Tuple[Tile, Tile]


def chebyshev(a: Tile, b: Tile) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


@dataclass
class Expectation:
    """What a click should lead to."""
    what: str
    start: Tile
    goal: Tile
    cost: int
    """Planned route cost start -> goal (steps, plus DOOR_COST per door)."""
    deadline: float
    final: bool
    """True if the goal is where we stop (target or door); otherwise the
    next click is made while still walking."""
    door: Optional[DoorKey] = None
    """Door the route goes through, if any: failures are put down to it."""


class WalkTest(Application):

    RECLICK_DISTANCE = 4
    """Click the next waypoint once this close (tiles) to the current one."""

    MOVING_SECONDS = 1.5
    """Counted as stopped once the position hasn't changed for this long."""

    SECONDS_PER_TILE = 0.7
    """Walking pace allowance (walking is 0.6s per tile, running 0.3s)."""

    MAX_DETOUR = 2
    """A position further than this (route cost) from any shortest route
    to the clicked tile means we're not going where we expected."""

    MAX_FAILURES = 2
    """Failed expectations in a row before giving up."""

    MAX_DOOR_FAILURES = 2
    """Failures at one door (didn't open, or couldn't get through) before
    giving up."""

    DOOR_HOVERS = 6
    """Hovers over a door's tiles without finding Open / Close before
    trying (once) to walk through."""

    DOOR_OPEN_SECONDS = 2.0
    """Wait after clicking "Open" before checking the door again (the
    player may have to step next to it, then it takes a tick to swing)."""

    MINIMAP_MARGIN = 8
    """Pixels kept clear of the minimap's safe radius."""

    def create_parser(self):
        parser = super().create_parser()
        target = parser.add_mutually_exclusive_group(required=True)
        target.add_argument('--to', type=int, nargs='+',
                            help='Target tile: x y [z] (z defaults to the '
                                 'current plane).')
        target.add_argument('--object',
                            help='Walk to the nearest object with this '
                                 'name and use it.')
        parser.add_argument('--action',
                            help='With --object: the option to click '
                                 '(default: the first one).')
        parser.add_argument('--radius', type=int, default=60,
                            help='With --object: search radius in tiles.')
        parser.add_argument('--no-doors', action='store_true',
                            help="Don't route through doors.")
        return parser

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cm: Optional[CollisionMap] = None
        self.target: Optional[Tile] = None
        self.path: List[Tile] = []
        self.clicks = 0
        self.clicked_at = 0.0
        self.last_pos: Optional[Tile] = None
        self.last_moved = 0.0
        self.started = time.time()
        # expectations
        self.expect: Optional[Expectation] = None
        self.failures = 0
        self.detour_cache: Dict[Tile, int] = {}
        # doors
        self.door: Optional[GameEntity] = None
        self.door_key: Optional[DoorKey] = None
        self.door_hovers = 0
        self.door_clicked_at = 0.0
        self.door_open: Optional[DoorKey] = None
        """Door seen open (mouse text "Close"): we may walk through it."""
        self.door_blind: Optional[DoorKey] = None
        """Door we couldn't find on screen and are trying to walk through
        anyway (once)."""
        self.door_failures: Dict[DoorKey, int] = {}
        # objects
        self.om: Optional[ObjectMap] = None
        self.obj: Optional[MapObject] = None
        self.obj_entity: Optional[GameEntity] = None
        self.obj_clicked_at = 0.0
        self.obj_hovers = 0
        self.obj_menu_at = 0.0
        self.obj_menu_tries = 0

    def setup(self):
        self.args = getattr(self, 'args', None) or self.parse_args()
        self.cm = CollisionMap()
        if self.args.object:
            self.om = ObjectMap()
        # mouse text: white action ("Open") and cyan object name ("Gate")
        self.client.mouse_options.add_colours('white', 'cyan')
        log = self.client.logger
        log.info(f'Walk test: collision data {self.cm.meta}')

    # helpers

    @property
    def log(self):
        return self.client.logger

    def give_up(self, reason: str):
        self.log.warning(f'Walk test: giving up: {reason} '
                         f'(after {time.time() - self.started:.0f}s, '
                         f'{self.clicks} clicks)')
        self.stop()

    def position(self) -> Optional[Tile]:
        x, y, z = self.client.gauges.minimap.gps.get_coordinates()
        if x == -1:
            return None
        return int(x), int(y), int(z)

    def route_cost(self, a: Tile, b: Tile) -> Optional[int]:
        path = self.cm.find_path(a, b, doors=not self.args.no_doors,
                                 max_nodes=20_000)
        if path is None:
            return None
        return len(path) - 1 + DOOR_COST * len(self.cm.doors_on_path(path))

    def plan(self, pos: Tile) -> bool:
        t = time.perf_counter()
        path = self.cm.find_path(pos, self.target,
                                 doors=not self.args.no_doors)
        ms = (time.perf_counter() - t) * 1000
        if path is None:
            self.log.warning(f'Walk test: no path {pos} -> {self.target}')
            return False
        doors = [path[i] for i in self.cm.doors_on_path(path)]
        self.log.info(f'Walk test: path {pos} -> {self.target}: '
                      f'{len(path) - 1} steps, doors at {doors}, '
                      f'planned in {ms:.0f}ms')
        self.path = path
        return True

    def max_click_tiles(self) -> float:
        mm = self.client.gauges.minimap
        return (mm.safe_radius - self.MINIMAP_MARGIN) / mm.tile_size

    def minimap_xy(self, pos: Tile, tile: Tile) -> Tuple[int, int]:
        """Screen point of a tile's centre on the minimap."""
        mm = self.client.gauges.minimap
        x1, y1, _, _ = mm.get_bbox()
        ox, oy = mm.orb_xy
        dx, dy = tile[0] - pos[0], tile[1] - pos[1]
        return (x1 + ox + dx * mm.tile_size,
                y1 + oy - dy * mm.tile_size)

    def click(self, x: int, y: int):
        self.client.screen.click_aoi(x - 1, y - 1, x + 1, y + 1, speed=0.5)
        self.clicked_at = time.time()

    def stop_walking(self, pos: Tile):
        """Click our own tile on the minimap, so the game stops walking us
        somewhere we didn't expect."""
        self.click(*self.minimap_xy(pos, pos))
        self.log.info(f'Walk test: stopping at {pos}')

    # expectations

    def walk_to(self, pos: Tile, tile: Tile, cost: int, final: bool,
                door: Optional[DoorKey] = None):
        """Click a tile on the minimap and record what should happen."""
        x, y = self.minimap_xy(pos, tile)
        what = f'walk {pos} -> {tile}' + (f' through door {door}'
                                           if door else '')
        self.log.info(f'Walk test: expect {what} (cost {cost})')
        self.click(x, y)
        self.clicks += 1
        self.expect = Expectation(
            what, pos, tile, cost,
            deadline=time.time() + 2 + cost * self.SECONDS_PER_TILE,
            final=final, door=door)
        self.detour_cache = {}

    def detour(self, pos: Tile) -> Optional[int]:
        """How far off a shortest route start -> goal a position is (0 on
        one), or None if the goal can't be reached from it."""
        e = self.expect
        if pos not in self.detour_cache:
            a = self.route_cost(e.start, pos)
            b = self.route_cost(pos, e.goal)
            self.detour_cache[pos] = (None if a is None or b is None
                                      else a + b - e.cost)
        return self.detour_cache[pos]

    def check_expectation(self, pos: Tile, moving: bool) -> Optional[bool]:
        """True if met, False if failed (already handled), None if still in
        progress."""
        e = self.expect
        now = time.time()
        if pos == e.goal:
            return True
        if not e.final and moving and \
                chebyshev(pos, e.goal) <= self.RECLICK_DISTANCE:
            return True
        reason = None
        detour = self.detour(pos)
        if detour is None:
            # standing where the map data says we can't: data mismatch,
            # not evidence we're lost; the other checks still apply
            self.msg.append(f'{pos} not walkable in map data')
        elif detour > self.MAX_DETOUR:
            reason = (f'went off route (detour {detour}), e.g. a door or '
                      f'gate on the way is shut')
        elif not moving and now - self.clicked_at > self.MOVING_SECONDS:
            reason = f'stopped {chebyshev(pos, e.goal)} tiles short'
        elif now > e.deadline:
            reason = 'took too long'
        if reason is None:
            return None
        self.fail(pos, moving, f'expected {e.what}, but {reason}, at {pos}')
        return False

    def fail(self, pos: Tile, moving: bool, message: str):
        e = self.expect
        self.expect = None
        self.failures += 1
        self.log.warning(f'Walk test: {message} '
                         f'(failure {self.failures}/{self.MAX_FAILURES})')
        if moving:
            self.stop_walking(pos)
        if e is not None and e.door is not None:
            # the door is the likely cause: look at it again before
            # trusting it's open
            n = self.door_failures.get(e.door, 0) + 1
            self.door_failures[e.door] = n
            if self.door_open == e.door:
                self.door_open = None
            if self.door_blind == e.door:
                self.door_blind = None
            if n >= self.MAX_DOOR_FAILURES:
                self.give_up(f"can't get through door {e.door} ({n} "
                             f"failures)")
                return
            self.log.info(f'Walk test: re-checking door {e.door}')
        if self.failures >= self.MAX_FAILURES:
            self.give_up(f'{self.failures} failed expectations in a row')

    # door handling

    def door_entity(self, a: Tile, b: Tile) -> GameEntity:
        """Game entity covering both tiles either side of a door, so the
        door (on the edge between them) is inside its screen box.

        Entity keys put the anchor at the north-west tile; width extends
        east and height extends south."""
        key = (a, b)
        if self.door is not None and self.door_key == key:
            return self.door
        x = min(a[0], b[0])
        y = max(a[1], b[1])
        width = abs(a[0] - b[0]) + 1
        height = abs(a[1] - b[1]) + 1
        entity = self.client.game_screen.create_game_entity(
            'door', 'door', (0, 0), self.client, self.client,
            tile_width=width, tile_height=height,
        )
        entity.set_global_coordinates(x, y, a[2])
        self.door, self.door_key = entity, key
        self.door_hovers = 0
        self.door_clicked_at = 0.0
        self._update_game_entities(entity)
        return entity

    def check_door(self, pos: Tile, nxt: Tile) -> None:
        """Standing next to a door: find out whether it's open (mouse text
        "Close ...") and open it if it's shut ("Open ..."). Sets
        ``door_open`` once it's seen open."""
        door = self.door_entity(pos, nxt)
        key = self.door_key
        now = time.time()
        if now - self.door_clicked_at < self.DOOR_OPEN_SECONDS:
            self.msg.append('waiting for door to open')
            return
        text = self.client.mouse_options.state or ''
        hovering = door.is_inside(*self.client.screen.mouse_xy)
        if hovering and re.search('^[Cc]lose', text):
            self.log.info(f'Walk test: door {key} is open ({text!r})')
            self.door_open = key
            return
        if self.door_clicked_at and hovering and \
                re.search('^[Oo]pen', text):
            # clicked "Open" earlier and it still says "Open"
            n = self.door_failures.get(key, 0) + 1
            self.door_failures[key] = n
            self.door_clicked_at = 0.0
            self.log.warning(f'Walk test: door {key} still shut after '
                             f'clicking Open ({text!r}), failure {n}/'
                             f'{self.MAX_DOOR_FAILURES}')
            if n >= self.MAX_DOOR_FAILURES:
                self.give_up(f"door {key} won't open")
            return
        if self._click_entity(door, 1.2, 1.8, '^[Oo]pen'):
            self.log.info(f'Walk test: clicked Open on door {key}')
            self.door_clicked_at = time.time()
            self.door_hovers = 0
            return
        if self.door_clicked_at and hovering:
            # clicked "Open" and the option is gone from the door's tiles:
            # it swung open, out of the way (its "Close" is elsewhere)
            self.log.info(f'Walk test: door {key} opened ("Open" gone '
                          f'after clicking, mouse text {text!r})')
            self.door_open = key
            return
        self.door_hovers += 1
        if self.door_hovers >= self.DOOR_HOVERS:
            # no "Open" / "Close" on the door's tiles: maybe someone opened
            # it and it swung out of the way. Walk through once, with the
            # usual expectation; a failure brings us back here.
            self.door_hovers = 0
            if self.door_failures.get(key, 0):
                self.give_up(f"can't find door {key} on screen (mouse "
                             f"text {text!r}) and walking through failed")
                return
            self.log.info(f'Walk test: door {key} not found after '
                          f'{self.DOOR_HOVERS} hovers (mouse text '
                          f'{text!r}), trying to walk through once')
            self.door_blind = key

    # objects

    def choose_object(self, pos: Tile) -> bool:
        found = self.om.nearest_reachable(
            pos, self.cm, name=self.args.object, action=self.args.action,
            radius=self.args.radius, doors=not self.args.no_doors)
        if found is None:
            nearby = self.om.find(self.args.object, self.args.action,
                                  near=pos, radius=self.args.radius)
            self.log.warning(
                f'Walk test: no reachable {self.args.object!r} within '
                f'{self.args.radius} tiles of {pos}; in range:\n'
                + describe(nearby[:10], pos))
            return False
        self.obj, path = found
        self.target = path[-1]
        tiles = self.om.interaction_tiles(self.obj, self.cm)
        self.log.info(f'Walk test: object {self.obj}, size {self.obj.size}, '
                      f'used from {tiles}; walking to {self.target} '
                      f'({len(path) - 1} steps)')
        return True

    def object_entity(self) -> GameEntity:
        """Game entity over the object's tiles. Entity keys put the anchor
        at the north-west tile; width extends east, height south."""
        if self.obj_entity is None:
            o = self.obj
            sx, sy = o.size
            entity = self.client.game_screen.create_game_entity(
                'object', 'object', (0, 0), self.client, self.client,
                tile_width=sx, tile_height=sy,
            )
            entity.set_global_coordinates(o.x, o.y + sy - 1, o.plane)
            self.obj_entity = entity
            self._update_game_entities(entity)
        return self.obj_entity

    def use_object(self, pos: Tile):
        entity = self.object_entity()
        action = self.args.action or self.obj.actions[0]
        if self.obj_clicked_at:
            if time.time() - self.obj_clicked_at > 3:
                self.log.info(f'Walk test: used {self.obj.name!r} '
                              f'({action}) from {pos} in '
                              f'{time.time() - self.started:.0f}s')
                self.stop()
            return
        # the name can differ for varbit objects: match the option only,
        # and log the full mouse text. The option isn't always first: the
        # reader can put "/ N more options" before it. Not followed by "-"
        # or a letter, so "Climb" doesn't match "Climb-up".
        pattern = f'{re.escape(action)}(?![-A-Za-z])'
        if entity.context_menu is not None or self.obj_menu_at:
            self.use_object_menu(entity, action)
            return
        if self._click_entity(entity, 1.0, 1.5, pattern):
            self.obj_clicked_at = time.time()
            self.log.info(f'Walk test: clicked {self.obj.name!r} with mouse '
                          f'text {self.client.mouse_options.state!r}')
            return
        text = self.client.mouse_options.state or ''
        name = re.sub(r'\W', '', self.obj.name).lower()
        if entity.is_inside(*self.client.screen.mouse_xy) and \
                name and name in re.sub(r'\W', '', text).lower() and \
                self.client.ocr is not None:
            # hovering the right object, but the option isn't the left
            # click one: use the right-click menu
            self.log.info(f'Walk test: {action!r} not the left-click option '
                          f'({text!r}), right-clicking')
            if self._right_click(entity, set_ocr=True):
                self.obj_menu_at = time.time()
                self.obj_menu_tries += 1
            return
        self.obj_hovers += 1
        if self.obj_hovers > 20:
            self.give_up(f'no {action!r} on {self.obj} after '
                         f'{self.obj_hovers} hovers (mouse text '
                         f'{self.client.mouse_options.state!r})')

    def use_object_menu(self, entity: GameEntity, action: str):
        """Right-clicked the object: find the option in the menu (read by
        OCR, lower case, e.g. "climb-down staircase") and click it. Expect
        the menu to appear and contain the option; otherwise close it and
        give up after a second try."""
        menu = entity.context_menu
        waited = time.time() - self.obj_menu_at
        values = [i.value for i in menu.items] if menu is not None else []
        # OCR drops spaces: a line reads option + object name, e.g.
        # "climb-downstaircase"; match exactly that so "climb" doesn't
        # pick "climb-upstaircase"
        wanted = {re.sub(r'\s', '', f'{action}{n}').lower()
                  for n in self.obj.names or [self.obj.name]}
        for item in (menu.items if menu is not None else []):
            if re.sub(r'\s', '', item.value) in wanted:
                item.click(tmin=1.0, tmax=1.5, pause_before_click=True)
                self.obj_clicked_at = time.time()
                self.obj_menu_at = 0.0
                self.log.info(f'Walk test: clicked menu option '
                              f'{item.value!r} on {self.obj.name!r}')
                return
        if waited < 2.0:
            self.msg.append(f'reading menu: {values}')
            return
        # expectation failed: no menu, or no such option in it
        self.log.warning(f'Walk test: expected {action!r} in the right-click '
                         f'menu of {self.obj.name!r}, menu items: {values}')
        if menu is not None:
            self.client.screen.mouse_away_object(menu)
        self.obj_menu_at = 0.0
        if self.obj_menu_tries >= 2:
            self.give_up(f'no {action!r} in the right-click menu after '
                         f'{self.obj_menu_tries} tries')

    # main loop

    def update(self):
        self.client.gauges.update()
        self.client.game_screen.update()
        self.client.mouse_options.update()
        if self.door is not None:
            self._update_game_entities(self.door)
        if self.obj_entity is not None:
            self._update_game_entities(self.obj_entity)

    def action(self):
        pos = self.position()
        if pos is None:
            self.msg.append('no position')
            return

        now = time.time()
        if pos != self.last_pos:
            self.last_pos = pos
            self.last_moved = now
        moving = now - self.last_moved < self.MOVING_SECONDS

        if self.target is None:
            if self.args.object:
                if not self.choose_object(pos):
                    self.stop()
                    return
            else:
                to = self.args.to
                self.target = (to[0], to[1],
                               to[2] if len(to) > 2 else pos[2])

        # check the last click did what we expected before anything else
        if self.expect is not None:
            met = self.check_expectation(pos, moving)
            if met is None:
                self.msg.append(f'{self.expect.what}: '
                                f'{chebyshev(pos, self.expect.goal)} left')
                return
            if not met:
                return
            self.log.info(f'Walk test: done: {self.expect.what}')
            self.failures = 0
            self.expect = None

        if pos == self.target:
            if self.obj is not None:
                if moving:
                    self.msg.append('waiting to stop next to object')
                else:
                    self.use_object(pos)
                return
            self.log.info(f'Walk test: arrived at {pos} in '
                          f'{now - self.started:.0f}s, {self.clicks} clicks')
            self.stop()
            return

        if not self.plan(pos):
            self.stop()
            return
        remaining = self.path
        doors = self.cm.doors_on_path(remaining)
        through: Optional[DoorKey] = None
        if doors and doors[0] == 0:
            key = (remaining[0], remaining[1])
            if self.door_open == key or self.door_blind == key:
                through = key
                doors = doors[1:]
            elif moving:
                self.msg.append('waiting to stop at door')
                return
            else:
                self.check_door(pos, remaining[1])
                return

        # furthest tile we can click on the minimap without passing a
        # (closed) door
        limit = doors[0] if doors else len(remaining) - 1
        reach = self.max_click_tiles()
        best = 0
        for k in range(1, limit + 1):
            t = remaining[k]
            if math.hypot(t[0] - pos[0], t[1] - pos[1]) > reach:
                break
            best = k
        if best == 0:
            self.msg.append('nothing in reach')
            return
        cost = best + (DOOR_COST if through else 0)
        self.walk_to(pos, remaining[best], cost, final=best == limit,
                     door=through)
        if through:
            # past this door (or failed at it): look again next time
            self.door_open = self.door_blind = None


def main():
    app = WalkTest()
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
