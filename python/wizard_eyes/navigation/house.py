"""The player-owned house: a per-account map of rooms stitched together from
template chunks.

A house is an instance: every room is a copy of an 8x8 template chunk from
the game cache (around x 1856-2111, y 7040-7167, one copy per house
style), turned by a quarter turn or more and put in a cell of the house
grid. Which template goes where, and how it's turned, is the player's own
design, so it's recorded per account (``scripts/house_map.py`` walks the
house to learn it) in ``data/maps/house/<account>.local.json``, never
published.

Inside the house RuneLite's grid info overlay reads the *template* tile the
player stands on, not where they are in the house. This module converts
between the two:

* house tiles: a made-up frame where cell (cx, cy) on floor z covers
  x ``HOUSE_ORIGIN[0] + 8 * cx`` ... ``+ 7`` (y likewise), z = the floor's
  template plane (1 is the ground floor). The frame is fixed by the first
  room mapped (cell 0, 0).
* template tiles: what the overlay reads; a template used by more than one
  room is told apart by the previous position.

:class:`HouseCollisionMap` turns the template collision with each room, so
the usual pathfinder and walker work in house tiles.

Logging out in the house puts the player outside the house portal on the
next login, so a bot must check where it is after every login rather than
assume it's still in the house (the portal's world tile is kept as
``portal`` once seen).
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ..file_path_utils import get_root
from .collision_map import (CollisionMap, MOVE_E, MOVE_N, MOVE_NE, MOVE_NW,
                            MOVE_S, MOVE_SE, MOVE_SW, MOVE_W, Tile,
                            _STEP_BITS)

CHUNK = 8
HOUSE_ORIGIN = (6400, 6400)
"""House tile of cell (0, 0)'s south-west corner. Far outside the real map,
so house tiles can't be mistaken for world tiles."""

TEMPLATE_AREA = (1856, 7040, 2112, 7168)
"""x1, y1, x2, y2 (exclusive) of the room templates in the cache."""

LAYOUT_PATH = 'data/maps/house/{account}.local.json'

# a quarter turn clockwise
_CW = {MOVE_N: MOVE_E, MOVE_E: MOVE_S, MOVE_S: MOVE_W, MOVE_W: MOVE_N,
       MOVE_NE: MOVE_SE, MOVE_SE: MOVE_SW, MOVE_SW: MOVE_NW,
       MOVE_NW: MOVE_NE}

SIDES = {'north': (0, 1), 'east': (1, 0), 'south': (0, -1), 'west': (-1, 0)}


def in_template_area(tile: Optional[Tile]) -> bool:
    """Whether an overlay tile is a house room template (i.e. the player
    is in a house)."""
    if not tile:
        return False
    x1, y1, x2, y2 = TEMPLATE_AREA
    return x1 <= tile[0] < x2 and y1 <= tile[1] < y2


EMPTY_TEMPLATE = (1864, 7040)
"""The grass of an empty cell, in the first house style (other styles are
64 tiles east each). Every empty cell uses it, and the player can walk
onto it through a doorway that leads nowhere."""


def is_empty_template(tile: Optional[Tile]) -> bool:
    """Whether an overlay tile is on an empty cell's grass (outside every
    room)."""
    if not in_template_area(tile):
        return False
    return ((tile[0] - TEMPLATE_AREA[0]) % 64 // CHUNK * CHUNK
            == EMPTY_TEMPLATE[0] - TEMPLATE_AREA[0]
            and tile[1] // CHUNK * CHUNK == EMPTY_TEMPLATE[1])


def rotate_local(x: int, y: int, r: int) -> Tuple[int, int]:
    """Template chunk-local tile -> house chunk-local tile, turned ``r``
    quarter turns clockwise."""
    for _ in range(r % 4):
        x, y = y, CHUNK - 1 - x
    return x, y


def unrotate_local(x: int, y: int, r: int) -> Tuple[int, int]:
    return rotate_local(x, y, 4 - r % 4)


def rotate_vector(dx: int, dy: int, r: int) -> Tuple[int, int]:
    """A step in the template -> the same step in the house."""
    for _ in range(r % 4):
        dx, dy = dy, -dx
    return dx, dy


def rotate_mask(mask: int, r: int) -> int:
    """Movement bits (template directions) -> house directions."""
    for _ in range(r % 4):
        out = 0
        for bit, to in _CW.items():
            if mask & bit:
                out |= to
        mask = out
    return mask


def rotation_from_step(template_step: Tuple[int, int],
                       house_step: Tuple[int, int]) -> Optional[int]:
    """The rotation that best turns a step seen in the template into a step
    made in the house (None if nothing moved or no turn points the same
    way)."""
    if template_step == (0, 0):
        return None
    best, best_dot = None, 0
    for r in range(4):
        dx, dy = rotate_vector(*template_step, r)
        dot = dx * house_step[0] + dy * house_step[1]
        if dot > best_dot:
            best, best_dot = r, dot
    return best


@dataclass
class Room:
    cell: Tuple[int, int, int]
    """(cx, cy, z) in the house grid; z is the floor's template plane."""
    template: Tuple[int, int, int]
    """South-west tile of the template chunk (x, y, plane)."""
    rotation: int
    """Quarter turns clockwise from the template."""
    furniture: Dict[str, str] = field(default_factory=dict)
    """Hotspot name -> what's built there, as seen in game
    ({"Jewellery box space": "Ornate Jewellery Box"})."""

    @property
    def origin(self) -> Tile:
        """House tile of the cell's south-west corner."""
        cx, cy, z = self.cell
        return (HOUSE_ORIGIN[0] + CHUNK * cx, HOUSE_ORIGIN[1] + CHUNK * cy, z)

    def contains_template(self, tile: Tile) -> bool:
        tx, ty, tz = self.template
        return (tx <= tile[0] < tx + CHUNK and ty <= tile[1] < ty + CHUNK
                and tile[2] == tz)

    def contains(self, tile: Tile) -> bool:
        ox, oy, z = self.origin
        return (ox <= tile[0] < ox + CHUNK and oy <= tile[1] < oy + CHUNK
                and tile[2] == z)

    def to_house(self, tile: Tile) -> Tile:
        tx, ty, _ = self.template
        lx, ly = rotate_local(tile[0] - tx, tile[1] - ty, self.rotation)
        ox, oy, z = self.origin
        return ox + lx, oy + ly, z

    def to_template(self, tile: Tile) -> Tile:
        ox, oy, _ = self.origin
        lx, ly = unrotate_local(tile[0] - ox, tile[1] - oy, self.rotation)
        tx, ty, tz = self.template
        return tx + lx, ty + ly, tz

    def as_json(self) -> dict:
        return {'cell': list(self.cell), 'template': list(self.template),
                'rotation': self.rotation, 'furniture': self.furniture}

    @classmethod
    def from_json(cls, d: dict) -> 'Room':
        return cls(tuple(d['cell']), tuple(d['template']), d['rotation'],
                   dict(d.get('furniture', {})))


def template_chunk(tile: Tile) -> Tile:
    return tile[0] // CHUNK * CHUNK, tile[1] // CHUNK * CHUNK, tile[2]


def cell_of(tile: Tile) -> Tuple[int, int, int]:
    return ((tile[0] - HOUSE_ORIGIN[0]) // CHUNK,
            (tile[1] - HOUSE_ORIGIN[1]) // CHUNK, tile[2])


class HouseLayout:
    """One account's house: its rooms, and the portal's world tile."""

    def __init__(self, account: str, rooms: Iterable[Room] = (),
                 portal: Optional[Tile] = None, path: Optional[Path] = None):
        self.account = account
        self.rooms: Dict[Tuple[int, int, int], Room] = {
            r.cell: r for r in rooms}
        self.portal = portal
        self.path = path or self.path_for(account)

    @staticmethod
    def path_for(account: str) -> Path:
        return Path(get_root()) / LAYOUT_PATH.format(account=account)

    @classmethod
    def load(cls, account: str, path: Optional[Path] = None
             ) -> 'HouseLayout':
        """The account's saved layout, or an empty one."""
        path = path or cls.path_for(account)
        if not path.exists():
            return cls(account, path=path)
        with open(path, encoding='utf-8') as f:
            raw = json.load(f)
        portal = tuple(raw['portal']) if raw.get('portal') else None
        return cls(account, [Room.from_json(r) for r in raw['rooms']],
                   portal, path)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        raw = {'account': self.account, 'updated': time.strftime('%Y-%m-%d'),
               'portal': list(self.portal) if self.portal else None,
               'rooms': [r.as_json() for r in sorted(
                   self.rooms.values(), key=lambda r: r.cell)]}
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(raw, f, indent=1)

    def add(self, room: Room):
        self.rooms[room.cell] = room

    def room_at(self, tile: Tile) -> Optional[Room]:
        """The room a house tile is in."""
        return self.rooms.get(cell_of(tile))

    def rooms_with_template(self, tile: Tile) -> List[Room]:
        return [r for r in self.rooms.values() if r.contains_template(tile)]

    def to_house(self, tile: Tile, near: Optional[Tile] = None
                 ) -> Optional[Tile]:
        """Overlay (template) tile -> house tile. A template used by more
        than one room needs ``near`` (the last house position) to pick."""
        rooms = self.rooms_with_template(tile)
        if not rooms:
            return None
        if len(rooms) == 1:
            return rooms[0].to_house(tile)
        if near is None:
            return None
        options = [r.to_house(tile) for r in rooms]
        return min(options, key=lambda t: (t[2] != near[2], max(
            abs(t[0] - near[0]), abs(t[1] - near[1]))))

    def to_template(self, tile: Tile) -> Optional[Tile]:
        room = self.room_at(tile)
        return room.to_template(tile) if room else None

    def hotspots(self, object_map, name, floor: Optional[int] = None
                 ) -> List[Tuple[Room, List[Tile]]]:
        """Rooms with a hotspot (e.g. "Jewellery box space") and the house
        tiles it covers."""
        out = []
        for room in self.rooms.values():
            if floor is not None and room.cell[2] != floor:
                continue
            tx, ty, tz = room.template
            for o in object_map.find(name, plane=tz):
                if room.contains_template(o.tile):
                    out.append((room, [room.to_house(t)
                                       for t in o.footprint()]))
        return out

    def describe(self) -> str:
        """A text picture of each floor: rooms by grid cell."""
        lines = []
        for z in sorted({c[2] for c in self.rooms}):
            cells = [c for c in self.rooms if c[2] == z]
            xs = [c[0] for c in cells]
            ys = [c[1] for c in cells]
            lines.append(f'floor {z}:')
            for cy in range(max(ys), min(ys) - 1, -1):
                row = []
                for cx in range(min(xs), max(xs) + 1):
                    room = self.rooms.get((cx, cy, z))
                    row.append(f'{room.template[0]},{room.template[1]}'
                               f' r{room.rotation}' if room else '.' * 12)
                lines.append('  ' + ' | '.join(f'{s:>12}' for s in row))
        return '\n'.join(lines)


class HousePosition:
    """Turns the overlay's template tile into a house tile every frame.

        tracker = HousePosition(layout)
        ...  # after client.gauges.update():
        pos = tracker.update(client.gauges.minimap.gps)

    With ``set_gps`` the GPS is given the house tile, so game entities
    placed in house tiles and minimap clicks work as they do outside.
    ``in_house`` is False once the overlay reads a world tile again
    (left through the portal, teleported out, or logged out and back in:
    that puts the player outside the portal)."""

    def __init__(self, layout: HouseLayout, set_gps: bool = True):
        self.layout = layout
        self.set_gps = set_gps
        self.template: Optional[Tile] = None
        self.house: Optional[Tile] = None
        self.in_house = False

    def update(self, gps) -> Optional[Tile]:
        x, y, z = gps.get_coordinates()
        if x == -1:
            return self.house
        tile = (int(x), int(y), int(z))
        if not in_template_area(tile):
            if tile != self.house:  # not our own house tile fed back
                self.in_house = False
                self.house = None
                self.template = None
            return None
        self.in_house = True
        self.template = tile
        house = self.layout.to_house(tile, near=self.house)
        if house is not None:
            self.house = house
            if self.set_gps:
                gps.set_coordinates(*house, add_history=False)
        return house


class HouseCollisionMap(CollisionMap):
    """Movement in house tiles, from each room's template collision turned
    with the room. A step between rooms needs both sides open (the
    templates' doorways line up); diagonal steps between rooms aren't
    allowed. No doors (rooms join through open doorways)."""

    def __init__(self, layout: HouseLayout, base: CollisionMap):
        # no super().__init__: nothing of our own to load
        self.layout = layout
        self.base = base
        self.path = base.path

    @property
    def meta(self) -> dict:
        return self.base.meta

    def _raw(self, x: int, y: int, z: int) -> int:
        room = self.layout.room_at((x, y, z))
        if room is None:
            return 0
        t = room.to_template((x, y, z))
        return rotate_mask(self.base.mask(*t), room.rotation)

    def mask(self, x: int, y: int, z: int) -> int:
        m = self._raw(x, y, z)
        if not m:
            return 0
        cell = cell_of((x, y, z))
        for (dx, dy), bit in _STEP_BITS.items():
            if not m & bit:
                continue
            other = (x + dx, y + dy, z)
            if cell_of(other) == cell:
                continue
            back = _STEP_BITS[(-dx, -dy)]
            if dx and dy or not self._raw(*other) & back:
                m &= ~bit
        return m

    def door_mask(self, x: int, y: int, z: int) -> int:
        return 0
