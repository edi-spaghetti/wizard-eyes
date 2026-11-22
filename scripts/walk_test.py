"""Test traveller: walk to a tile using the cache-built collision map.

Plans a path with ``CollisionMap.find_path`` and follows it by clicking the
furthest path tile that's safely inside the minimap, re-clicking shortly
before reaching it. At a closed door it stops on the tile before, hovers the
door's game entity on the game screen and clicks only if the mouse text says
"Open" (already open doors, "Close ...", are walked through).

Needs the camera north / top-down (start-up checks do this) and the grid
info overlay for coordinates. Run::

    venv\\Scripts\\python scripts\\walk_test.py --to 3253 3266 --run-time 300
"""
import math
import re
import time
from typing import List, Optional, Tuple

from wizard_eyes.application import Application
from wizard_eyes.game_entities.entity import GameEntity
from wizard_eyes.navigation.collision_map import CollisionMap

Tile = Tuple[int, int, int]


class WalkTest(Application):

    RECLICK_DISTANCE = 4
    """Click the next waypoint once this close (tiles) to the current one."""

    MOVING_SECONDS = 1.5
    """Counted as stopped once the position hasn't changed for this long."""

    DOOR_HOVERS = 6
    """Hovers over a door's tiles before assuming it's already open."""

    MINIMAP_MARGIN = 8
    """Pixels kept clear of the minimap's safe radius."""

    def create_parser(self):
        parser = super().create_parser()
        parser.add_argument('--to', type=int, nargs='+', required=True,
                            help='Target tile: x y [z] (z defaults to the '
                                 'current plane).')
        parser.add_argument('--no-doors', action='store_true',
                            help="Don't route through doors.")
        return parser

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cm: Optional[CollisionMap] = None
        self.target: Optional[Tile] = None
        self.path: List[Tile] = []
        self.waypoint: Optional[Tile] = None
        self.waypoint_final = False
        self.clicks = 0
        self.clicked_at = 0.0
        self.last_pos: Optional[Tile] = None
        self.last_moved = 0.0
        self.door: Optional[GameEntity] = None
        self.door_key = None
        self.door_attempts = 0
        self.door_passed: Optional[Tile] = None
        self.door_walk_tries = 0
        self.started = time.time()

    def setup(self):
        self.args = getattr(self, 'args', None) or self.parse_args()
        self.cm = CollisionMap()
        # mouse text: white action ("Open") and cyan object name ("Gate")
        self.client.mouse_options.add_colours('white', 'cyan')
        log = self.client.logger
        log.info(f'Walk test: collision data {self.cm.meta}')

    # helpers

    @property
    def log(self):
        return self.client.logger

    def position(self) -> Optional[Tile]:
        x, y, z = self.client.gauges.minimap.gps.get_coordinates()
        if x == -1:
            return None
        return int(x), int(y), int(z)

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
        self._update_game_entities(entity)
        return entity

    def handle_door(self, pos: Tile, nxt: Tile) -> None:
        """Standing next to a door: open it on the game screen if it's
        closed (mouse text "Open ..."), carry on if it's already open."""
        door = self.door_entity(pos, nxt)
        text = self.client.mouse_options.state or ''
        if door.is_inside(*self.client.screen.mouse_xy) and                 re.search('^[Cc]lose', text):
            self.log.info(f'Walk test: door already open ({text!r})')
            self.door_passed = pos
            return
        if door.clicked:
            self.msg.append('waiting for door to open')
            return
        clicked = self._click_entity(door, 1.2, 1.8, '^[Oo]pen')
        if clicked:
            self.log.info(f'Walk test: opening door at {self.door_key} '
                          f'({text!r})')
            self.door_passed = pos
            self.door_attempts = 0
            return
        self.door_attempts += 1
        if self.door_attempts >= self.DOOR_HOVERS:
            # no "Open" / "Close" on the door's tiles: most likely someone
            # opened it and it swung out of the way. Try walking through;
            # if that doesn't move us, we look for it again.
            self.log.info(f'Walk test: door not found after '
                          f'{self.door_attempts} hovers (mouse text '
                          f'{text!r}), trying to walk through')
            self.door_passed = pos
            self.door_attempts = 0
            self.door_walk_tries += 1
            if self.door_walk_tries > 3:
                self.log.warning('Walk test: stuck at door, giving up')
                self.stop()

    # main loop

    def update(self):
        self.client.gauges.update()
        self.client.game_screen.update()
        self.client.mouse_options.update()
        if self.door is not None:
            self._update_game_entities(self.door)

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
            to = self.args.to
            self.target = (to[0], to[1], to[2] if len(to) > 2 else pos[2])

        if pos == self.target:
            self.log.info(f'Walk test: arrived at {pos} in '
                          f'{now - self.started:.0f}s, {self.clicks} clicks')
            self.stop()
            return

        # keep walking to the last clicked tile; the game picks its own
        # shortest route there, which may differ from ours tile by tile
        if self.waypoint is not None and pos != self.waypoint:
            d = max(abs(pos[0] - self.waypoint[0]),
                    abs(pos[1] - self.waypoint[1]))
            if moving:
                if self.waypoint_final or d > self.RECLICK_DISTANCE:
                    self.msg.append(f'walking to {self.waypoint} ({d} left)')
                    return
            elif now - self.clicked_at < self.MOVING_SECONDS:
                self.msg.append('waiting for movement')
                return
            else:
                self.log.info(f'Walk test: stopped at {pos}, {d} short of '
                              f'{self.waypoint}')
                if pos == self.door_passed:
                    # tried to go through the door and didn't: it's shut
                    self.door_passed = None

        if not self.plan(pos):
            self.stop()
            return
        remaining = self.path
        doors = self.cm.doors_on_path(remaining)
        if doors and doors[0] == 0 and self.door_passed == pos:
            # door dealt with from here; walk through it
            doors = doors[1:]

        if doors and doors[0] == 0:
            if moving:
                self.msg.append('waiting to stop at door')
            else:
                self.handle_door(pos, remaining[1])
            return

        # furthest tile we can click on the minimap without passing a door
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
        candidate = remaining[best]

        x, y = self.minimap_xy(pos, candidate)
        self.log.info(f'Walk test: at {pos}, clicking {candidate} '
                      f'({best} steps along, {len(remaining) - 1} to go)')
        self.click(x, y)
        self.clicks += 1
        self.waypoint = candidate
        self.waypoint_final = best == limit


def main():
    app = WalkTest()
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
