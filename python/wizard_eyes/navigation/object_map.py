"""Every named or usable object on the map, built from the game cache (see
``utility/map_data.py``), with lookups like "nearest Bank booth" or
"Ladder with Climb-up", and the tiles an object can be used from.

The data file holds one entry per object placement (arrays ``x``, ``y``,
``plane``, ``id``, ``type``, ``orientation``) and a JSON dictionary of the
object definitions they use (``definitions``), see
``wizard_eyes.cache.object_data``.

Objects that change with game state (doors, quest objects) have a base
definition plus ``transforms``: the definitions they can turn into. The
current state can't be read from the cache, so lookups match an object if
any of its forms match.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Pattern, Sequence, Tuple, Union

import numpy

from ..file_path_utils import ensure_local, get_root
from .collision_map import CollisionMap, Tile

DEFAULT_PATH = 'data/maps/objects/objects.npz'

Match = Union[str, Pattern, None]
"""Text to match: a string matches exactly (ignoring case), a compiled
regex matches with ``search``, None matches anything."""

# location types
WALL_TYPES = range(0, 4)
WALL_DECORATION_TYPES = range(4, 9)
GROUND_DECORATION = 22

# access_block bits: sides an object can NOT be used from
BLOCK_NORTH = 0x1
BLOCK_EAST = 0x2
BLOCK_SOUTH = 0x4
BLOCK_WEST = 0x8

# for a straight wall (type 0), the tile on the other side by orientation
_ACROSS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}


@dataclass(frozen=True)
class ObjectDefinition:
    id: int
    name: str
    actions: Tuple[Optional[str], ...]
    size_x: int
    size_y: int
    interact_type: int
    access_block: int
    transforms: Tuple[int, ...]

    @property
    def blocks(self) -> bool:
        return self.interact_type != 0


@dataclass(frozen=True)
class MapObject:
    """One object placed on the map."""
    id: int
    x: int
    y: int
    plane: int
    type: int
    orientation: int
    """0 west, 1 north, 2 east, 3 south."""
    definition: ObjectDefinition
    variants: Tuple[ObjectDefinition, ...] = ()
    """Definitions the object can turn into (empty for most objects)."""

    @property
    def tile(self) -> Tile:
        return self.x, self.y, self.plane

    @property
    def forms(self) -> Tuple[ObjectDefinition, ...]:
        return (self.definition,) + self.variants

    @property
    def name(self) -> str:
        """The base name, or the first variant's name for objects whose
        base definition is unnamed."""
        for d in self.forms:
            if d.name not in ('null', ''):
                return d.name
        return self.definition.name

    @property
    def names(self) -> List[str]:
        return sorted({d.name for d in self.forms
                       if d.name not in ('null', '')})

    @property
    def actions(self) -> List[str]:
        seen = []
        for d in self.forms:
            for a in d.actions:
                if a and a not in seen:
                    seen.append(a)
        return seen

    @property
    def size(self) -> Tuple[int, int]:
        """Tiles covered (x, y) after rotation; walls and decorations are
        always one tile."""
        if self.type not in (9, 10, 11, GROUND_DECORATION):
            return 1, 1
        d = self.definition
        if self.orientation in (1, 3):
            return d.size_y, d.size_x
        return d.size_x, d.size_y

    def footprint(self) -> List[Tile]:
        sx, sy = self.size
        return [(self.x + i, self.y + j, self.plane)
                for i in range(sx) for j in range(sy)]

    def distance(self, tile: Tile) -> int:
        """Tiles (Chebyshev) from a tile to the nearest covered tile."""
        sx, sy = self.size
        dx = max(self.x - tile[0], 0, tile[0] - (self.x + sx - 1))
        dy = max(self.y - tile[1], 0, tile[1] - (self.y + sy - 1))
        return max(dx, dy)

    def access_block(self) -> int:
        """Sides the object can't be used from, rotated with it."""
        mask = self.definition.access_block & 0xF
        r = self.orientation
        if r:
            mask = ((mask << r) & 0xF) | (mask >> (4 - r))
        return mask

    def __repr__(self):
        return (f'<MapObject {self.name!r} #{self.id} at {self.tile} '
                f'type {self.type} rot {self.orientation} {self.actions}>')


def _matches(text: Optional[str], match: Match) -> bool:
    if match is None:
        return True
    if not text:
        return False
    if isinstance(match, str):
        return text.lower() == match.lower()
    return match.search(text) is not None


class ObjectMap:
    """Read-only index of the objects on the map."""

    def __init__(self, path: Union[str, Path, None] = None):
        if path is None:
            path = Path(get_root()) / DEFAULT_PATH
            ensure_local(str(path))
        self.path = Path(path)
        with numpy.load(self.path) as npz:
            self.x = npz['x'].astype(numpy.int32)
            self.y = npz['y'].astype(numpy.int32)
            self.plane = npz['plane']
            self.ids = npz['id']
            self.types = npz['type']
            self.orientations = npz['orientation']
            self.meta = json.loads(str(npz['meta']))
            raw = json.loads(str(npz['definitions']))
        self.definitions: Dict[int, ObjectDefinition] = {
            int(k): ObjectDefinition(int(k), v[0], tuple(v[1]), v[2], v[3],
                                     v[4], v[5], tuple(v[6]))
            for k, v in raw.items()
        }
        self._match_cache: Dict[tuple, numpy.ndarray] = {}

    def __len__(self):
        return len(self.ids)

    def forms(self, oid: int) -> List[ObjectDefinition]:
        d = self.definitions[oid]
        return [d] + [self.definitions[t] for t in d.transforms
                      if t in self.definitions]

    def matching_ids(self, name: Match = None,
                     action: Match = None) -> numpy.ndarray:
        """Object ids with a form matching both name and action."""
        key = (name, action)
        if key not in self._match_cache:
            ids = []
            for oid, d in self.definitions.items():
                forms = self.forms(oid)
                if (any(_matches(f.name, name) for f in forms)
                        and (action is None or any(
                            _matches(a, action)
                            for f in forms for a in f.actions))):
                    ids.append(oid)
            self._match_cache[key] = numpy.array(sorted(ids), numpy.int32)
        return self._match_cache[key]

    def _object(self, i: int) -> MapObject:
        oid = int(self.ids[i])
        forms = self.forms(oid)
        return MapObject(oid, int(self.x[i]), int(self.y[i]),
                         int(self.plane[i]), int(self.types[i]),
                         int(self.orientations[i]), forms[0],
                         tuple(forms[1:]))

    def find(self, name: Match = None, action: Match = None,
             near: Optional[Tile] = None, radius: Optional[int] = None,
             plane: Optional[int] = None, limit: Optional[int] = None,
             ) -> List[MapObject]:
        """Objects matching a name and / or option.

        :param near: sort by distance from this tile, and keep to its plane.
        :param radius: with ``near``, only objects within this many tiles.
        :param plane: only objects on this plane (default: ``near``'s).
        """
        sel = numpy.ones(len(self.ids), bool)
        if name is not None or action is not None:
            sel &= numpy.isin(self.ids, self.matching_ids(name, action))
        if plane is None and near is not None:
            plane = near[2]
        if plane is not None:
            sel &= self.plane == plane
        if near is not None and radius is not None:
            # objects are at most a few tiles across; filter roughly here,
            # exactly below
            pad = radius + 8
            sel &= numpy.abs(self.x - near[0]) <= pad
            sel &= numpy.abs(self.y - near[1]) <= pad
        found = [self._object(i) for i in numpy.flatnonzero(sel)]
        if near is not None:
            if radius is not None:
                found = [o for o in found if o.distance(near) <= radius]
            found.sort(key=lambda o: o.distance(near))
        return found[:limit] if limit else found

    def nearest(self, near: Tile, name: Match = None, action: Match = None,
                radius: Optional[int] = None) -> Optional[MapObject]:
        """The closest matching object in a straight line (see
        :meth:`nearest_reachable` for walking distance)."""
        found = self.find(name, action, near=near, radius=radius, limit=1)
        return found[0] if found else None

    def at(self, tile: Tile) -> List[MapObject]:
        """Objects covering a tile."""
        return self.find(near=tile, radius=0)

    # -- using objects ----------------------------------------------------

    def interaction_tiles(self, obj: MapObject,
                          collision: CollisionMap) -> List[Tile]:
        """Walkable tiles the object can be used from.

        - Walls and doors: the wall's own tile and, for straight walls, the
          tile on the other side.
        - Objects that block movement: tiles next to their edges (not
          corners), except on sides the object can't be used from.
        - Everything else (decorations, objects you can walk over): the
          tiles it covers and the tiles next to them.

        Walls between the player and an object aren't known here (the
        movement data only says where the player can step), so a tile on
        the far side of a wall next to an object can be returned.
        """
        z = obj.plane
        tiles = []
        if obj.type in WALL_TYPES:
            tiles.append(obj.tile)
            if obj.type == 0:
                dx, dy = _ACROSS[obj.orientation]
                tiles.append((obj.x + dx, obj.y + dy, z))
        elif obj.definition.blocks and obj.type != GROUND_DECORATION \
                and obj.type not in WALL_DECORATION_TYPES:
            sx, sy = obj.size
            blocked = obj.access_block()
            if not blocked & BLOCK_WEST:
                tiles += [(obj.x - 1, obj.y + j, z) for j in range(sy)]
            if not blocked & BLOCK_EAST:
                tiles += [(obj.x + sx, obj.y + j, z) for j in range(sy)]
            if not blocked & BLOCK_SOUTH:
                tiles += [(obj.x + i, obj.y - 1, z) for i in range(sx)]
            if not blocked & BLOCK_NORTH:
                tiles += [(obj.x + i, obj.y + sy, z) for i in range(sx)]
        else:
            covered = obj.footprint()
            tiles += covered
            for x, y, _ in covered:
                tiles += [(x - 1, y, z), (x + 1, y, z),
                          (x, y - 1, z), (x, y + 1, z)]
        seen, out = set(), []
        for t in tiles:
            if t not in seen and collision.walkable(*t):
                seen.add(t)
                out.append(t)
        return out

    def path_to(self, obj: MapObject, start: Tile, collision: CollisionMap,
                doors: bool = True) -> Optional[List[Tile]]:
        """Shortest walking path from start to a tile the object can be
        used from, or None."""
        goals = self.interaction_tiles(obj, collision)
        if not goals:
            return None
        return collision.find_path(start, goals, doors=doors)

    def nearest_reachable(self, start: Tile, collision: CollisionMap,
                          name: Match = None, action: Match = None,
                          radius: int = 50, doors: bool = True,
                          ) -> Optional[Tuple[MapObject, List[Tile]]]:
        """The matching object with the shortest walk from start, and the
        path to it, searching objects within ``radius`` tiles."""
        candidates = self.find(name, action, near=start, radius=radius)
        goals: Dict[Tile, MapObject] = {}
        for obj in candidates:
            for t in self.interaction_tiles(obj, collision):
                goals.setdefault(t, obj)
        if not goals:
            return None
        path = collision.find_path(start, list(goals), doors=doors)
        if path is None:
            return None
        return goals[path[-1]], path


def describe(objects: Sequence[MapObject], origin: Optional[Tile] = None,
             ) -> str:
    """One line per object, for logs."""
    lines = []
    for o in objects:
        dist = f' {o.distance(origin):3d} tiles' if origin else ''
        lines.append(f'{o.name:<28} #{o.id:<6} {o.tile}{dist} '
                     f'{", ".join(o.actions)}')
    return '\n'.join(lines)
