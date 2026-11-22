"""Transports: objects that move the player somewhere else (stairs,
ladders, ...), and where they go.

The game cache has every staircase and ladder with its options, but not
where it leads. Destinations come from three places, in rising priority:

1. **Guesses** (``maps/transports/guessed.npz``, built by
   ``utility/map_data.py build``): a matching staircase / ladder on the
   floor above / below, or about 6400 tiles north underground, within a few
   tiles. Each has a confidence; where no partner is found there is no
   guess at all (a wrong guess is worse than none).
2. **Recordings**: real outcomes of using a transport in game
   (``scripts/transport_test.py``, or any bot via :meth:`Transports.record`).
   New ones go to ``maps/transports/recorded.local.json``, which is never
   published so the asset sync can't overwrite it; ``utility/map_data.py
   transports merge`` folds them into the curated file.
3. **Curated** (``maps/transports/transports.json``, published): merged
   recordings plus entries written by hand (``"source": "manual"``), which
   can also say where a transport lands at random (``to_radius``), whether
   it's one way, and what it needs (``requirements``).

Entries are keyed by object tile + option (object ids change with game
state, e.g. doors and quest objects), see :func:`transport_key`.
"""

import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy

from ..file_path_utils import ensure_local, get_root
from .collision_map import CollisionMap, Tile
from .object_map import MapObject, ObjectMap

GUESSED_PATH = 'data/maps/transports/guessed.npz'
CURATED_PATH = 'data/maps/transports/transports.json'
RECORDED_PATH = 'data/maps/transports/recorded.local.json'

SOURCES = ('guess', 'observed', 'manual')
"""Lowest to highest priority."""

UNDERGROUND = 6400
"""Underground areas are about this many tiles north of the surface."""

TRANSPORT_COST = 3
"""Pathfinding cost of using a transport, in walking steps (click, a
couple of ticks to climb)."""


def transport_key(object_tile: Iterable[int], action: str) -> str:
    x, y, z = object_tile
    return f'{x},{y},{z}:{action.lower()}'


@dataclass
class Transport:
    object_tile: Tile
    action: str
    to: Tile
    source: str = 'guess'
    object_id: int = -1
    name: str = ''
    confidence: float = 1.0
    """Guesses: how likely the destination is right (rough)."""
    from_tile: Optional[Tile] = None
    """Where it was used from, if recorded (else any usable tile)."""
    to_radius: int = 0
    """Landing is random within this many tiles of ``to``."""
    one_way: bool = False
    requirements: Dict[str, object] = field(default_factory=dict)
    count: int = 1
    """Recordings: how many times this outcome was seen."""
    seconds: Optional[float] = None
    notes: str = ''
    updated_at: str = ''

    @property
    def key(self) -> str:
        return transport_key(self.object_tile, self.action)

    def to_json(self) -> dict:
        d = asdict(self)
        for k in ('object_tile', 'to', 'from_tile'):
            if d[k] is not None:
                d[k] = list(d[k])
        return d

    @classmethod
    def from_json(cls, d: dict) -> 'Transport':
        d = dict(d)
        for k in ('object_tile', 'to', 'from_tile'):
            if d.get(k) is not None:
                d[k] = tuple(d[k])
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})


# -- guessing ---------------------------------------------------------------

UP_ACTIONS = {'climb-up', 'climb up', 'go-up', 'walk-up'}
DOWN_ACTIONS = {'climb-down', 'climb down', 'go-down', 'walk-down'}

FLOOR_RADIUS = 3
"""Floors: partner staircase / ladder within this many tiles."""
UNDERGROUND_RADIUS = 8
"""Surface <-> underground: partner within this many tiles of y +- 6400."""


def _direction(actions: Iterable[str]) -> Tuple[bool, bool]:
    acts = {a.lower() for a in actions if a}
    return bool(acts & UP_ACTIONS), bool(acts & DOWN_ACTIONS)


def guess_transports(om: ObjectMap, cm: CollisionMap) -> List[Transport]:
    """Guess where every staircase / ladder style option leads.

    For "up" options: a partner object one plane up within
    ``FLOOR_RADIUS`` tiles that can go down (or plain "Climb"); from
    underground (y > 6400, plane 0): a partner on the surface about 6400
    tiles south. "Down" the other way round. The landing tile is the
    partner's usable tile closest to the object; confidence falls with the
    partner's distance and is lower underground.
    """
    # candidate objects: anything with an up / down / plain climb option
    up_down: List[Tuple[MapObject, bool, bool]] = []
    grid: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    ids = set()
    for oid in om.definitions:
        acts = {a.lower() for f in om.forms(oid) for a in f.actions if a}
        if acts & (UP_ACTIONS | DOWN_ACTIONS | {'climb'}):
            ids.add(oid)
    idx = numpy.flatnonzero(numpy.isin(om.ids, numpy.array(sorted(ids))))
    for i in idx:
        obj = om._object(int(i))
        up, down = _direction(obj.actions)
        plain = any(a.lower() == 'climb' for a in obj.actions)
        up_down.append((obj, up or plain, down or plain))
        grid[(obj.plane, obj.x >> 3, obj.y >> 3)].append(len(up_down) - 1)

    def partners(x, y, z, radius, need_up, need_down):
        found = []
        for gx in range((x - radius) >> 3, ((x + radius) >> 3) + 1):
            for gy in range((y - radius) >> 3, ((y + radius) >> 3) + 1):
                for j in grid.get((z, gx, gy), ()):
                    other, up, down = up_down[j]
                    if (need_up and not up) or (need_down and not down):
                        continue
                    d = other.distance((x, y, z))
                    if d <= radius:
                        found.append((d, other))
        found.sort(key=lambda p: p[0])
        return found

    guesses = []
    for obj, can_up, can_down in up_down:
        for action in obj.actions:
            a = action.lower()
            if a in UP_ACTIONS:
                if obj.plane == 0 and obj.y >= UNDERGROUND:
                    spot = (obj.x, obj.y - UNDERGROUND, 0)
                    radius, base = UNDERGROUND_RADIUS, 0.5
                elif obj.plane < 3:
                    spot = (obj.x, obj.y, obj.plane + 1)
                    radius, base = FLOOR_RADIUS, 0.9
                else:
                    continue
                cands = partners(*spot, radius, False, True)
            elif a in DOWN_ACTIONS:
                if obj.plane > 0:
                    spot = (obj.x, obj.y, obj.plane - 1)
                    radius, base = FLOOR_RADIUS, 0.9
                elif obj.y < UNDERGROUND:
                    spot = (obj.x, obj.y + UNDERGROUND, 0)
                    radius, base = UNDERGROUND_RADIUS, 0.5
                else:
                    continue
                cands = partners(*spot, radius, True, False)
            else:
                continue
            if not cands:
                continue
            dist, partner = cands[0]
            tiles = om.interaction_tiles(partner, cm)
            if not tiles:
                continue
            # land next to the partner, on the side nearest the object
            to = min(tiles, key=lambda t: max(abs(t[0] - spot[0]),
                                              abs(t[1] - spot[1])))
            ambiguous = len(cands) > 1 and cands[1][0] == dist
            confidence = base * (1 - dist / (radius + 1))
            if ambiguous:
                confidence *= 0.5
            # which side of the partner we land on isn't known: anywhere
            # around it (e.g. a 2x2 staircase lands on one fixed side)
            guesses.append(Transport(
                object_tile=obj.tile, action=action, to=to, source='guess',
                object_id=obj.id, name=obj.name,
                confidence=round(confidence, 2),
                to_radius=max(partner.size)))
    return guesses


def save_guesses(path: str, guesses: List[Transport], meta: dict):
    tmp = path + '.tmp.npz'
    numpy.savez_compressed(
        tmp,
        transports=numpy.array(json.dumps([g.to_json() for g in guesses])),
        meta=numpy.array(json.dumps(meta)))
    os.replace(tmp, path)


# -- runtime ----------------------------------------------------------------

def _load_json(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, encoding='utf-8') as f:
        return json.load(f)


class Transports:
    """All known transports, merged by priority (manual > observed >
    guess; among recordings the most recent wins)."""

    DEFAULT_MIN_CONFIDENCE = 0.5
    """Guesses below this are left out by default (e.g. underground
    ladders whose nearest surface partner is several tiles off)."""

    def __init__(self, guessed: Union[str, Path, None] = None,
                 curated: Union[str, Path, None] = None,
                 recorded: Union[str, Path, None] = None,
                 min_confidence: float = DEFAULT_MIN_CONFIDENCE):
        root = Path(get_root())
        if guessed is None:
            guessed = root / GUESSED_PATH
            ensure_local(str(guessed))
        if curated is None:
            curated = root / CURATED_PATH
            ensure_local(str(curated))
        self.recorded_path = Path(recorded or root / RECORDED_PATH)
        self.min_confidence = min_confidence
        self.meta = {}
        entries: List[Transport] = []
        guessed = Path(guessed)
        if guessed.exists():
            with numpy.load(guessed) as npz:
                self.meta = json.loads(str(npz['meta']))
                entries += [Transport.from_json(d) for d in
                            json.loads(str(npz['transports']))]
        entries += [Transport.from_json(d) for d in _load_json(Path(curated))]
        entries += [Transport.from_json(d)
                    for d in _load_json(self.recorded_path)]
        self.by_key: Dict[str, Transport] = {}
        for t in entries:
            self._add(t)
        self._by_object: Dict[Tile, List[Transport]] = defaultdict(list)
        for t in self.by_key.values():
            self._by_object[t.object_tile].append(t)

    def _add(self, t: Transport):
        if t.source == 'guess' and t.confidence < self.min_confidence:
            return
        old = self.by_key.get(t.key)
        if old is None or self._rank(t) >= self._rank(old):
            self.by_key[t.key] = t

    @staticmethod
    def _rank(t: Transport):
        return SOURCES.index(t.source), t.updated_at

    def __len__(self):
        return len(self.by_key)

    def __iter__(self):
        return iter(self.by_key.values())

    def get(self, object_tile: Tile, action: str) -> Optional[Transport]:
        return self.by_key.get(transport_key(object_tile, action))

    def at(self, object_tile: Tile) -> List[Transport]:
        """Transports of the object at a tile (one per option)."""
        return list(self._by_object.get(tuple(object_tile), ()))

    def for_object(self, obj: MapObject) -> List[Transport]:
        return self.at(obj.tile)

    def edges(self, om: ObjectMap, cm: CollisionMap,
              ) -> Dict[Tile, List[Tuple[Tile, int, Transport]]]:
        """For the pathfinder: from each tile a transport can be used
        from, (destination, cost, transport). One-way and random-landing
        transports are included; a planner should check ``requirements``."""
        out: Dict[Tile, List[Tuple[Tile, int, Transport]]] = defaultdict(list)
        for t in self:
            if t.from_tile is not None and t.source != 'guess':
                starts = [t.from_tile]
                # the object can be used from its other tiles too
                obj = self._object(om, t)
                if obj is not None:
                    starts += [s for s in om.interaction_tiles(obj, cm)
                               if s != t.from_tile]
            else:
                obj = self._object(om, t)
                starts = om.interaction_tiles(obj, cm) if obj else []
            for s in starts:
                out[s].append((t.to, TRANSPORT_COST, t))
        return out

    @staticmethod
    def _object(om: ObjectMap, t: Transport) -> Optional[MapObject]:
        for obj in om.at(t.object_tile):
            if obj.tile == t.object_tile:
                return obj
        return None

    def record(self, obj: MapObject, action: str, from_tile: Tile,
               to: Tile, seconds: Optional[float] = None) -> Transport:
        """Save a real outcome to the local recordings file (and use it
        from now on)."""
        now = datetime.now(timezone.utc).isoformat(timespec='seconds')
        old = self.get(obj.tile, action)
        count = 1
        if old is not None and old.source == 'observed' and old.to == to:
            count = old.count + 1
        t = Transport(
            object_tile=obj.tile, action=action, to=tuple(to),
            source='observed', object_id=obj.id, name=obj.name,
            from_tile=tuple(from_tile), count=count, seconds=seconds,
            updated_at=now)
        recorded = _load_json(self.recorded_path)
        recorded = [d for d in recorded
                    if transport_key(d['object_tile'], d['action']) != t.key]
        recorded.append(t.to_json())
        self.recorded_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.recorded_path.with_suffix('.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(recorded, f, indent=1)
        os.replace(tmp, self.recorded_path)
        self.by_key[t.key] = t
        self._by_object[t.object_tile] = [
            x for x in self._by_object[t.object_tile] if x.key != t.key
        ] + [t]
        return t


def merge_recordings(curated_path: str, recorded_path: str,
                     ) -> Tuple[int, int, List[str]]:
    """Fold local recordings into the curated file. Manual entries are
    kept as they are; a recording that disagrees with an earlier one
    replaces it and is reported. Returns (added, updated, warnings)."""
    curated = {transport_key(d['object_tile'], d['action']): d
               for d in _load_json(Path(curated_path))}
    added = updated = 0
    warnings = []
    for d in _load_json(Path(recorded_path)):
        key = transport_key(d['object_tile'], d['action'])
        old = curated.get(key)
        if old is None:
            curated[key] = d
            added += 1
            continue
        if old.get('source') == 'manual':
            if list(old['to']) != list(d['to']):
                warnings.append(f'{key}: recorded {d["to"]} differs from '
                                f'manual entry {old["to"]} (kept manual)')
            continue
        if list(old['to']) != list(d['to']):
            warnings.append(f'{key}: landed {d["to"]}, previously '
                            f'{old["to"]} (random landing? set to_radius)')
        elif d.get('updated_at', '') <= old.get('updated_at', ''):
            continue
        # a recording's count already includes earlier sightings (it was
        # made with the curated entry loaded), so it simply replaces it
        curated[key] = d
        updated += 1
    os.makedirs(os.path.dirname(curated_path), exist_ok=True)
    with open(curated_path, 'w', encoding='utf-8') as f:
        json.dump(sorted(curated.values(),
                         key=lambda d: (d['object_tile'][2],
                                        d['object_tile'][0],
                                        d['object_tile'][1], d['action'])),
                  f, indent=1)
    return added, updated, warnings
