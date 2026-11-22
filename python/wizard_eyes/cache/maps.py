"""Region map data: index 5, one group per region (group id = region id).

Each group holds five files: 0 terrain, 1 object placements ("locations"),
2 a few floats (likely lighting), 3 and 4 empty in the current cache.
Region ids are ``(region_x << 8) | region_y``; a region is 64x64 tiles and
world coordinates are ``region_x * 64 + local_x``.
"""

import struct
from dataclasses import dataclass
from typing import Iterator, List, Optional

import numpy

from .store import Buffer, CacheStore

MAP_INDEX = 5
TERRAIN_FILE = 0
LOCATIONS_FILE = 1

REGION_SIZE = 64
PLANES = 4

# tile settings bits
SETTING_BLOCKED = 0x1
SETTING_BRIDGE = 0x2


def region_id(region_x: int, region_y: int) -> int:
    return (region_x << 8) | region_y


def region_xy(rid: int):
    return rid >> 8, rid & 0xFF


@dataclass
class Location:
    """An object placed on the map."""
    id: int
    x: int
    """local x (0-63)"""
    y: int
    """local y (0-63)"""
    plane: int
    type: int
    """0-3 walls, 4-8 wall decorations, 9 diagonal wall, 10-11 objects,
    12-21 roofs, 22 ground decoration."""
    orientation: int
    """0 west, 1 north, 2 east, 3 south (rotation in 90 degree steps)."""


@dataclass
class Region:
    id: int
    settings: numpy.ndarray
    """uint8 [plane, x, y] tile settings."""
    heights: numpy.ndarray
    """int16 [plane, x, y] raw height byte (0 = not set)."""
    locations: List[Location]

    @property
    def base(self):
        rx, ry = region_xy(self.id)
        return rx * REGION_SIZE, ry * REGION_SIZE


def decode_terrain(data: bytes):
    settings = numpy.zeros((PLANES, REGION_SIZE, REGION_SIZE), numpy.uint8)
    heights = numpy.zeros((PLANES, REGION_SIZE, REGION_SIZE), numpy.int16)
    buf = Buffer(data)
    for z in range(PLANES):
        for x in range(REGION_SIZE):
            for y in range(REGION_SIZE):
                while True:
                    op = buf.u16()
                    if op == 0:
                        break
                    if op == 1:
                        heights[z, x, y] = buf.u8()
                        break
                    if op <= 49:
                        buf.u16()  # overlay id (shape/rotation in op)
                    elif op <= 81:
                        settings[z, x, y] = op - 49
                    # else underlay id = op - 81
    # current caches end the terrain with a flag byte: 0 for most regions,
    # 1 followed by an extra per-tile section in a few. Neither affects
    # tile settings, so the extra section is ignored.
    trailer = data[buf.pos:buf.pos + 1]
    if trailer not in (b'', b'\x00', b'\x01'):
        raise ValueError(
            f'unexpected terrain trailer: {data[buf.pos:buf.pos + 16].hex()}')
    return settings, heights


def decode_locations(data: bytes) -> List[Location]:
    locations = []
    buf = Buffer(data)
    oid = -1
    while True:
        delta = buf.big_smart()
        if delta == 0:
            break
        oid += delta
        packed = 0
        while True:
            pdelta = buf.smart_u16()
            if pdelta == 0:
                break
            packed += pdelta - 1
            attributes = buf.u8()
            locations.append(Location(
                id=oid,
                x=packed >> 6 & 0x3F,
                y=packed & 0x3F,
                plane=packed >> 12 & 0x3,
                type=attributes >> 2,
                orientation=attributes & 0x3,
            ))
    if buf.remaining():
        raise ValueError(f'{buf.remaining()} trailing location bytes')
    return locations


def load_region(store: CacheStore, rid: int) -> Optional[Region]:
    files = store.files(MAP_INDEX, rid)
    if files is None:
        return None
    settings, heights = decode_terrain(files[TERRAIN_FILE])
    locations = decode_locations(files.get(LOCATIONS_FILE, b'\x00'))
    return Region(rid, settings, heights, locations)


def region_ids(store: CacheStore) -> List[int]:
    return list(store.table(MAP_INDEX).groups)


def load_regions(store: CacheStore,
                 errors: Optional[dict] = None) -> Iterator[Region]:
    """All regions in the cache. Groups that fail to decode are skipped and
    recorded in ``errors`` (region id -> message) if given."""
    for rid in region_ids(store):
        try:
            region = load_region(store, rid)
        except (ValueError, IndexError, struct.error) as err:
            if errors is not None:
                errors[rid] = str(err)
            continue
        if region is not None:
            yield region
