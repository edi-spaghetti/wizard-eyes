"""Build per-tile collision from map regions and object definitions.

Follows the game client's collision rules:

- a tile whose settings have bit 0x1 is blocked (floor);
- where plane 1's settings have the bridge bit (0x2), everything on that
  column moves down one plane (the bridge surface becomes plane 0 and what's
  under it is dropped);
- walls (location types 0-3) block the edges they sit on;
- objects (types 9-21) block every tile they cover;
- ground decorations (type 22) block only if their definition says so;
- objects with interact type 0 never block.

The result per region is a ``uint8 [plane, x, y]`` movement mask: one bit
per direction the player can step from that tile (see ``MOVE_*``), plus a
second mask of steps that are blocked only by a door (a wall object with an
"Open" option), which a bot can open.
"""

from typing import Dict, Iterable, Optional, Tuple

import numpy

from .maps import (
    PLANES, REGION_SIZE, SETTING_BLOCKED, SETTING_BRIDGE, Region,
    region_id, region_xy,
)
from .objects import ObjectDefinition

# client collision flags
WALL_NW = 0x1
WALL_N = 0x2
WALL_NE = 0x4
WALL_E = 0x8
WALL_SE = 0x10
WALL_S = 0x20
WALL_SW = 0x40
WALL_W = 0x80
OBJECT = 0x100
FLOOR_DECORATION = 0x40000
FLOOR = 0x200000
FULL = OBJECT | FLOOR_DECORATION | FLOOR

# movement mask bits
MOVE_N = 0x1
MOVE_E = 0x2
MOVE_S = 0x4
MOVE_W = 0x8
MOVE_NE = 0x10
MOVE_SE = 0x20
MOVE_SW = 0x40
MOVE_NW = 0x80

DIRECTIONS = (
    # bit, dx, dy
    (MOVE_N, 0, 1),
    (MOVE_E, 1, 0),
    (MOVE_S, 0, -1),
    (MOVE_W, -1, 0),
    (MOVE_NE, 1, 1),
    (MOVE_SE, 1, -1),
    (MOVE_SW, -1, -1),
    (MOVE_NW, -1, 1),
)

# wall flags per (type, orientation): (dx, dy, flag) for each tile touched
_WALLS = {
    0: {  # straight wall
        0: ((0, 0, WALL_W), (-1, 0, WALL_E)),
        1: ((0, 0, WALL_N), (0, 1, WALL_S)),
        2: ((0, 0, WALL_E), (1, 0, WALL_W)),
        3: ((0, 0, WALL_S), (0, -1, WALL_N)),
    },
    1: {  # diagonal corner (also type 3)
        0: ((0, 0, WALL_NW), (-1, 1, WALL_SE)),
        1: ((0, 0, WALL_NE), (1, 1, WALL_SW)),
        2: ((0, 0, WALL_SE), (1, -1, WALL_NW)),
        3: ((0, 0, WALL_SW), (-1, -1, WALL_NE)),
    },
    2: {  # L-shaped corner
        0: ((0, 0, WALL_W | WALL_N), (-1, 0, WALL_E), (0, 1, WALL_S)),
        1: ((0, 0, WALL_N | WALL_E), (0, 1, WALL_S), (1, 0, WALL_W)),
        2: ((0, 0, WALL_E | WALL_S), (1, 0, WALL_W), (0, -1, WALL_N)),
        3: ((0, 0, WALL_S | WALL_W), (0, -1, WALL_N), (-1, 0, WALL_E)),
    },
}
_WALLS[3] = _WALLS[1]


class CollisionBuilder:
    """Accumulates client collision flags for any number of regions."""

    def __init__(self, objects: Dict[int, ObjectDefinition]):
        self.objects = objects
        self.flags: Dict[int, numpy.ndarray] = {}
        self.door_flags: Dict[int, numpy.ndarray] = {}
        self.regions = set()

    @staticmethod
    def _grid(store: Dict[int, numpy.ndarray], rid: int) -> numpy.ndarray:
        grid = store.get(rid)
        if grid is None:
            grid = numpy.zeros((PLANES, REGION_SIZE, REGION_SIZE),
                               numpy.uint32)
            store[rid] = grid
        return grid

    def _flag(self, plane: int, wx: int, wy: int, bits: int, door=False):
        rid = region_id(wx // REGION_SIZE, wy // REGION_SIZE)
        store = self.door_flags if door else self.flags
        self._grid(store, rid)[
            plane, wx % REGION_SIZE, wy % REGION_SIZE] |= bits

    @staticmethod
    def is_door(definition: ObjectDefinition) -> bool:
        return any(a and a.lower() == 'open' for a in definition.actions)

    def add_region(self, region: Region):
        self.regions.add(region.id)
        grid = self._grid(self.flags, region.id)
        settings = region.settings
        bridge = (settings[1] & SETTING_BRIDGE) != 0

        # blocked floor tiles
        for z in range(PLANES):
            blocked = (settings[z] & SETTING_BLOCKED) != 0
            if z > 0:
                grid[z - 1][blocked & bridge] |= FLOOR
            grid[z][blocked & ~bridge] |= FLOOR

        bx, by = region.base
        for loc in region.locations:
            definition = self.objects.get(loc.id)
            if definition is None:
                continue
            plane = loc.plane - 1 if bridge[loc.x, loc.y] else loc.plane
            if plane < 0:
                continue
            wx, wy = bx + loc.x, by + loc.y
            t = loc.type

            if t == 22:
                if definition.interact_type == 1:
                    self._flag(plane, wx, wy, FLOOR_DECORATION)
            elif t <= 3:
                if definition.interact_type != 0:
                    door = self.is_door(definition)
                    for dx, dy, bits in _WALLS[t][loc.orientation]:
                        self._flag(plane, wx + dx, wy + dy, bits, door=door)
            elif t >= 9:
                if definition.interact_type != 0:
                    sx, sy = definition.size(loc.orientation)
                    for ix in range(sx):
                        for iy in range(sy):
                            self._flag(plane, wx + ix, wy + iy, OBJECT)
            # 4-8: wall decorations never block

    def _padded(self, rid: int, doors_closed: bool) -> numpy.ndarray:
        """Flags for a region with a one-tile border taken from its
        neighbours (fully blocked where there's no neighbour)."""
        out = numpy.full((PLANES, REGION_SIZE + 2, REGION_SIZE + 2), FULL,
                         numpy.uint32)
        rx, ry = region_xy(rid)
        n = REGION_SIZE
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nid = region_id(rx + dx, ry + dy)
                if nid not in self.regions:
                    continue
                src = self.flags[nid]
                if doors_closed and nid in self.door_flags:
                    src = src | self.door_flags[nid]
                # destination slice in padded coords, source slice in region
                xs = {-1: (slice(0, 1), slice(n - 1, n)),
                      0: (slice(1, n + 1), slice(0, n)),
                      1: (slice(n + 1, n + 2), slice(0, 1))}[dx]
                ys = {-1: (slice(0, 1), slice(n - 1, n)),
                      0: (slice(1, n + 1), slice(0, n)),
                      1: (slice(n + 1, n + 2), slice(0, 1))}[dy]
                out[:, xs[0], ys[0]] = src[:, xs[1], ys[1]]
        return out

    def movement(self, rid: int, doors_closed: bool = True) -> numpy.ndarray:
        """uint8 [plane, x, y] mask of directions the player can step."""
        p = self._padded(rid, doors_closed)
        n = REGION_SIZE

        def at(dx, dy):
            return p[:, 1 + dx:1 + dx + n, 1 + dy:1 + dy + n]

        def clear(dx, dy, bits):
            return (at(dx, dy) & (bits | FULL)) == 0

        north = clear(0, 1, WALL_S)
        east = clear(1, 0, WALL_W)
        south = clear(0, -1, WALL_N)
        west = clear(-1, 0, WALL_E)
        ne = north & east & clear(1, 1, WALL_SW | WALL_S | WALL_W)
        se = south & east & clear(1, -1, WALL_NW | WALL_N | WALL_W)
        sw = south & west & clear(-1, -1, WALL_NE | WALL_N | WALL_E)
        nw = north & west & clear(-1, 1, WALL_SE | WALL_S | WALL_E)

        mask = numpy.zeros((PLANES, n, n), numpy.uint8)
        for bit, ok in ((MOVE_N, north), (MOVE_E, east), (MOVE_S, south),
                        (MOVE_W, west), (MOVE_NE, ne), (MOVE_SE, se),
                        (MOVE_SW, sw), (MOVE_NW, nw)):
            mask[ok] |= bit
        # nothing moves off a fully blocked tile
        mask[(p[:, 1:n + 1, 1:n + 1] & FULL) != 0] = 0
        return mask


    def doors(self, rid: int) -> numpy.ndarray:
        """uint8 [plane, x, y] mask of steps blocked only by doors."""
        closed = self.movement(rid, doors_closed=True)
        opened = self.movement(rid, doors_closed=False)
        return opened & ~closed


def build_movement(regions: Iterable[Region],
                   objects: Dict[int, ObjectDefinition],
                   ) -> Tuple[Dict[int, numpy.ndarray],
                              Dict[int, numpy.ndarray]]:
    """Movement and door masks for every region."""
    builder = CollisionBuilder(objects)
    for region in regions:
        builder.add_region(region)
    movement, doors = {}, {}
    for rid in sorted(builder.regions):
        movement[rid] = builder.movement(rid)
        door = builder.doors(rid)
        if door.any():
            doors[rid] = door
    return movement, doors
