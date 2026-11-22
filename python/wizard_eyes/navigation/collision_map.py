"""Per-tile movement data built from the game cache (see
``utility/map_data.py``), and a pathfinder over it.

The data file holds, per 64x64 region, a ``uint8 [plane, x, y]`` movement
mask keyed ``m<region id>`` (bits: directions the player can step from the
tile) and, for regions with doors, a mask keyed ``d<region id>`` of steps
blocked only by a door that can be opened. Bits match
``wizard_eyes.cache.collision.MOVE_*``.
"""

import heapq
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy

from ..file_path_utils import ensure_local, get_root

REGION_SIZE = 64

MOVE_N = 0x1
MOVE_E = 0x2
MOVE_S = 0x4
MOVE_W = 0x8
MOVE_NE = 0x10
MOVE_SE = 0x20
MOVE_SW = 0x40
MOVE_NW = 0x80

# same order the game client tries neighbours in
STEPS = (
    (MOVE_W, -1, 0),
    (MOVE_E, 1, 0),
    (MOVE_S, 0, -1),
    (MOVE_N, 0, 1),
    (MOVE_SW, -1, -1),
    (MOVE_SE, 1, -1),
    (MOVE_NW, -1, 1),
    (MOVE_NE, 1, 1),
)
_STEP_BITS = {(dx, dy): bit for bit, dx, dy in STEPS}

Tile = Tuple[int, int, int]

DEFAULT_PATH = 'data/maps/collision/movement.npz'

DOOR_COST = 5
"""Extra steps a door costs (stop, click it, wait for it to open)."""


def read_meta(path: Union[str, Path]) -> dict:
    with numpy.load(path) as npz:
        return json.loads(str(npz['meta']))


def region_of(x: int, y: int) -> int:
    return ((x >> 6) << 8) | (y >> 6)


class CollisionMap:
    """Read-only per-tile movement masks for the whole world."""

    def __init__(self, path: Union[str, Path, None] = None):
        if path is None:
            path = Path(get_root()) / DEFAULT_PATH
            ensure_local(str(path))
        self.path = Path(path)
        self._npz = numpy.load(self.path)
        self._keys = set(self._npz.files)
        self._cache: Dict[str, Optional[numpy.ndarray]] = {}

    @property
    def meta(self) -> dict:
        return json.loads(str(self._npz['meta']))

    def _grid(self, key: str) -> Optional[numpy.ndarray]:
        try:
            return self._cache[key]
        except KeyError:
            grid = self._npz[key] if key in self._keys else None
            self._cache[key] = grid
            return grid

    def _lookup(self, prefix: str, x: int, y: int, z: int) -> int:
        grid = self._grid(f'{prefix}{region_of(x, y)}')
        if grid is None or not 0 <= z < grid.shape[0]:
            return 0
        return int(grid[z, x & 63, y & 63])

    def mask(self, x: int, y: int, z: int) -> int:
        """Directions the player can step from a tile (0 if unknown)."""
        return self._lookup('m', x, y, z)

    def door_mask(self, x: int, y: int, z: int) -> int:
        """Directions blocked from a tile only by a door."""
        return self._lookup('d', x, y, z)

    def walkable(self, x: int, y: int, z: int) -> bool:
        return self.mask(x, y, z) != 0

    def can_step(self, x: int, y: int, z: int, dx: int, dy: int,
                 doors: bool = False) -> bool:
        bit = _STEP_BITS[(dx, dy)]
        m = self.mask(x, y, z)
        if doors:
            m |= self.door_mask(x, y, z)
        return bool(m & bit)

    def is_door_step(self, a: Tile, b: Tile) -> bool:
        """True if stepping from a to b goes through a (closed) door."""
        bit = _STEP_BITS.get((b[0] - a[0], b[1] - a[1]))
        return bool(bit and self.door_mask(*a) & bit)

    def neighbours(self, tile: Tile, doors: bool = True):
        """(next tile, cost) for every possible step."""
        x, y, z = tile
        m = self.mask(x, y, z)
        d = self.door_mask(x, y, z) if doors else 0
        for bit, dx, dy in STEPS:
            if m & bit:
                yield (x + dx, y + dy, z), 1
            elif d & bit:
                yield (x + dx, y + dy, z), 1 + DOOR_COST

    def find_path(self, start: Tile, end: Union[Tile, Iterable[Tile]],
                  doors: bool = True, max_nodes: int = 500_000,
                  ) -> Optional[List[Tile]]:
        """Shortest walking path from start to end, inclusive, or None.

        ``end`` is one tile, or several (e.g. the tiles an object can be
        used from): the path then goes to whichever is closest to walk to.

        Every step costs 1 (diagonals too, like in game); going through a
        closed door costs ``DOOR_COST`` more. Same plane only for now.
        """
        if isinstance(end, tuple) and len(end) == 3 \
                and isinstance(end[0], int):
            ends = [end]
        else:
            ends = list(end)
        # unwalkable goals (e.g. an object's own tile) can never be
        # reached; searching for them would scan everything reachable
        goals = {e for e in ends if e[2] == start[2]
                 and (self.walkable(*e) or self.door_mask(*e))}
        if start in ends:
            return [start]
        if not goals:
            return None

        if len(goals) <= 64:
            goal_list = list(goals)

            def h(t):
                return min(max(abs(t[0] - gx), abs(t[1] - gy))
                           for gx, gy, _ in goal_list)
        else:
            def h(_):
                return 0

        parents = {start: None}
        cost = {start: 0}
        heap = [(h(start), 0, start)]
        expanded = 0
        while heap:
            _, g, tile = heapq.heappop(heap)
            if tile in goals:
                path = [tile]
                while parents[path[-1]] is not None:
                    path.append(parents[path[-1]])
                return path[::-1]
            if g > cost[tile]:
                continue
            expanded += 1
            if expanded > max_nodes:
                return None
            for nxt, step in self.neighbours(tile, doors=doors):
                ng = g + step
                if ng < cost.get(nxt, 1 << 30):
                    cost[nxt] = ng
                    parents[nxt] = tile
                    heapq.heappush(heap, (ng + h(nxt), ng, nxt))
        return None

    def doors_on_path(self, path: List[Tile]) -> List[int]:
        """Indexes i where the step path[i] -> path[i + 1] is a door."""
        return [i for i in range(len(path) - 1)
                if self.is_door_step(path[i], path[i + 1])]
