"""Things on the main map by label ("bank_booth", "furnace"): objects from
the game cache (:class:`ObjectMap`) and NPCs (:class:`NPCMap`).

A label is the object's name with underscores for spaces. Banks have rules of
their own, see ``BANKS``.
"""
import re
from functools import lru_cache
from typing import List, Optional, Union

from .collision_map import Tile
from .npc_map import MapNPC, NPCMap
from .object_map import WALL_TYPES, MapObject, ObjectMap

MapThing = Union[MapObject, MapNPC]

BANKS = {
    'bank_booth': re.compile('(?i)^(bank booth|grand exchange booth)$'),
    'bank_chest': re.compile('(?i)^(bank chest|open chest)$'),
    'bank_table': re.compile('(?i)^bank table$'),
}
USE_BANKS = re.compile('(?i)^bank (chest|booth)$')

UNUSABLE_BANKS = {(3147, 3449, 0), (3148, 3449, 0)}
"""Tiles of banks never to go to: the Cooks' Guild's (entry requirements, and
it's wrongly reached through the wall: object data ignores walls)."""


@lru_cache(maxsize=None)
def object_map() -> ObjectMap:
    return ObjectMap()


@lru_cache(maxsize=None)
def npc_map() -> NPCMap:
    return NPCMap()


def is_bank(obj: MapObject) -> bool:
    """Opens the bank with a click on it. Not booths with a deposit box
    option: they change form, so their left click isn't certain."""
    if 'Deposit-box' in obj.actions or tuple(obj.tile) in UNUSABLE_BANKS:
        return False
    return 'Bank' in obj.actions or (
        'Use' in obj.actions and any(map(USE_BANKS.match, obj.names)))


def is_wall(thing: MapThing) -> bool:
    return isinstance(thing, MapObject) and thing.type in WALL_TYPES


def anchor(thing: MapThing) -> Tile:
    """The north-west tile, which game entities are anchored at."""
    x, y, z = thing.tile
    return x, y + thing.size[1] - 1, z


def find(label: str, near: Optional[Tile] = None,
         radius: Optional[int] = None, om: Optional[ObjectMap] = None,
         npcs: Optional[NPCMap] = None) -> List[MapThing]:
    """Everything with a label, nearest first (and on its plane) with
    ``near``."""
    found = (npcs or npc_map()).find(label, near, radius)
    if found:
        return found
    om = om or object_map()
    if label in BANKS:
        # counters (walls, at the Grand Exchange) need a click on their edge
        return [o for o in om.find(BANKS[label], near=near, radius=radius)
                if is_bank(o) and not is_wall(o)]
    return om.find(label.replace('_', ' '), near=near, radius=radius)


def find_banks(near: Tile, radius: int, walls: bool = False,
               om: Optional[ObjectMap] = None,
               npcs: Optional[NPCMap] = None) -> List[MapThing]:
    """Bank objects and bankers, nearest first."""
    om, npcs = om or object_map(), npcs or npc_map()
    found = [o for a in ('Bank', 'Use')
             for o in om.find(action=a, near=near, radius=radius)
             if is_bank(o) and (walls or not is_wall(o))]
    unique = {(o.tile, o.id): o for o in found}
    things = [*unique.values(), *npcs.find('banker', near, radius)]
    return sorted(things, key=lambda t: t.distance(near))
