"""Cache decoders, collision rules and pathfinding on synthetic data.

The encoders here write the same formats the game cache uses, so the
decoders are checked without needing a real cache on disk.
"""
import json
import os
import struct
import tempfile
from unittest import TestCase

import numpy

from wizard_eyes.cache.collision import (
    MOVE_E, MOVE_N, MOVE_NE, MOVE_S, MOVE_W, CollisionBuilder,
)
from wizard_eyes.cache.maps import (
    Location, Region, decode_locations, decode_terrain, region_id,
)
from wizard_eyes.cache.objects import ObjectDefinition, decode_object
from wizard_eyes.cache.store import Buffer, ReferenceTable, split_group
from wizard_eyes.navigation.collision_map import CollisionMap

RX, RY = 50, 50
BASE_X, BASE_Y = RX * 64, RY * 64


def smart(v):
    return bytes([v]) if v < 128 else struct.pack('>H', v + 0x8000)


def encode_locations(locs):
    """locs: list of (id, x, y, plane, type, orientation), sorted by id."""
    out = bytearray()
    by_id = {}
    for loc in locs:
        by_id.setdefault(loc[0], []).append(loc)
    last_id = -1
    for oid in sorted(by_id):
        out += smart(oid - last_id)
        last_id = oid
        last_packed = 0
        for _, x, y, z, t, o in sorted(by_id[oid],
                                       key=lambda l: l[3] << 12 | l[1] << 6 | l[2]):
            packed = z << 12 | x << 6 | y
            out += smart(packed - last_packed + 1)
            last_packed = packed
            out.append(t << 2 | o)
        out += b'\x00'
    out += b'\x00'
    return bytes(out)


def encode_terrain(settings, heights=None):
    out = bytearray()
    for z in range(4):
        for x in range(64):
            for y in range(64):
                if settings[z, x, y]:
                    out += struct.pack('>H', 49 + int(settings[z, x, y]))
                out += struct.pack('>H', 82)  # underlay 1
                if heights is not None and heights[z, x, y]:
                    out += struct.pack('>HB', 1, int(heights[z, x, y]))
                else:
                    out += b'\x00\x00'
    return bytes(out) + b'\x00'


class TestStore(TestCase):

    def test_smarts(self):
        buf = Buffer(smart(5) + smart(300) + struct.pack('>I', 0x80012345))
        self.assertEqual(buf.smart_u16(), 5)
        self.assertEqual(buf.smart_u16(), 300)
        self.assertEqual(buf.smart_u32(), 0x12345)

    def test_split_group(self):
        files = [b'abc', b'de', b'fghi']
        data = b''.join(files)
        # sizes are stored as deltas from the previous file's size
        sizes = [len(f) for f in files]
        deltas = [sizes[0]] + [b - a for a, b in zip(sizes, sizes[1:])]
        table = b''.join(struct.pack('>i', d) for d in deltas)
        self.assertEqual(split_group(data + table + b'\x01', 3), files)

    def test_reference_table_without_names(self):
        # protocol 7, revision, flags 4 (sizes), 2 groups (3, 5)
        data = bytearray([7]) + struct.pack('>iB', 99, 4)
        data += struct.pack('>HHH', 2, 3, 2)
        data += b'\x00' * (4 * 2)      # crc
        data += b'\x00' * (8 * 2)      # sizes
        data += b'\x00' * (4 * 2)      # revisions
        data += struct.pack('>HH', 2, 1)          # file counts
        data += struct.pack('>HHH', 0, 4, 0)      # file ids
        table = ReferenceTable.decode(bytes(data))
        self.assertEqual(table.revision, 99)
        self.assertEqual(table.groups, [3, 5])
        self.assertEqual(table.files, {3: [0, 4], 5: [0]})
        self.assertEqual(table.names, {})


class TestDecoders(TestCase):

    def test_locations_round_trip(self):
        locs = [(10, 1, 2, 0, 10, 1), (10, 63, 63, 3, 0, 3),
                (5000, 7, 8, 1, 22, 0)]
        decoded = decode_locations(encode_locations(locs))
        self.assertEqual(
            sorted((l.id, l.x, l.y, l.plane, l.type, l.orientation)
                   for l in decoded),
            sorted(locs))

    def test_terrain(self):
        settings = numpy.zeros((4, 64, 64), numpy.uint8)
        settings[0, 3, 4] = 1
        settings[1, 5, 6] = 2
        heights = numpy.zeros((4, 64, 64), numpy.int16)
        heights[2, 0, 0] = 30
        s, h = decode_terrain(encode_terrain(settings, heights))
        numpy.testing.assert_array_equal(s, settings)
        self.assertEqual(h[2, 0, 0], 30)

    def test_object_with_new_opcodes(self):
        data = (
            b'\x02Door\x00'                          # name
            + b'\x0e\x02'                            # size x 2
            + b'\x1eOpen\x00'                        # action 0
            + b'\x61\xff\xff\x00\x00\x01'            # 97: varbit, varp, n
            + b'\xff\xff\xff\x00\x3c\xec'            # 2 x 3-byte ids
            + b'\x65\x00\x01\x83\xff\xff'            # 101: conditional text
            + struct.pack('>ii', 1, 2) + b'Text\x00'
            + b'\x5d\x00\x00\x14\x00\x00\x14'        # 93
            + b'\x60\x05'                            # 96
            + b'\x00'
        )
        d = decode_object(7, data)
        self.assertEqual(d.name, 'Door')
        self.assertEqual(d.size_x, 2)
        self.assertEqual(d.actions[0], 'Open')
        self.assertEqual(d.transforms, [-1, 15596, -1])


def definitions():
    return {
        1: ObjectDefinition(1, name='Wall'),
        2: ObjectDefinition(2, name='Table', size_x=2, size_y=1),
        3: ObjectDefinition(3, name='Door', actions=['Open'] + [None] * 4),
        4: ObjectDefinition(4, name='Rug', interact_type=0),
        5: ObjectDefinition(5, name='Rock', interact_type=1),
    }


def region(locs, settings=None):
    if settings is None:
        settings = numpy.zeros((4, 64, 64), numpy.uint8)
    return Region(
        id=region_id(RX, RY),
        settings=settings,
        heights=numpy.zeros((4, 64, 64), numpy.int16),
        locations=[Location(*l) for l in locs],
    )


def build(locs, settings=None):
    builder = CollisionBuilder(definitions())
    builder.add_region(region(locs, settings))
    rid = region_id(RX, RY)
    return builder.movement(rid), builder.doors(rid)


class TestCollision(TestCase):

    def test_open_ground(self):
        m, _ = build([])
        self.assertEqual(m[0, 10, 10], 0xFF)
        # region edge with no neighbour region is blocked
        self.assertFalse(m[0, 0, 10] & MOVE_W)

    def test_straight_wall(self):
        # wall on the north edge of (10, 10)
        m, _ = build([(1, 10, 10, 0, 0, 1)])
        self.assertFalse(m[0, 10, 10] & MOVE_N)
        self.assertFalse(m[0, 10, 11] & MOVE_S)
        self.assertTrue(m[0, 10, 10] & MOVE_E)
        # diagonals across the wall are blocked too
        self.assertFalse(m[0, 9, 10] & MOVE_NE)

    def test_rotated_object_covers_tiles(self):
        # 2x1 table rotated a quarter turn covers 1x2 tiles
        m, _ = build([(2, 10, 10, 0, 10, 1)])
        self.assertEqual(m[0, 10, 10], 0)
        self.assertEqual(m[0, 10, 11], 0)
        self.assertTrue(m[0, 11, 10])
        self.assertFalse(m[0, 11, 10] & MOVE_W)

    def test_non_blocking_and_ground_decoration(self):
        m, _ = build([(4, 10, 10, 0, 10, 0), (5, 12, 12, 0, 22, 0),
                      (1, 14, 14, 0, 22, 0)])
        self.assertTrue(m[0, 10, 10])          # interact type 0
        self.assertEqual(m[0, 12, 12], 0)      # ground deco, type 1
        self.assertTrue(m[0, 14, 14])          # ground deco, default type

    def test_blocked_floor_and_bridge(self):
        settings = numpy.zeros((4, 64, 64), numpy.uint8)
        settings[0, 5, 5] = 1                  # blocked
        settings[1, 20, 20] = 2                # bridge column
        settings[0, 20, 20] = 1                # water under the bridge
        m, _ = build([(5, 20, 21, 1, 10, 0)], settings)
        self.assertEqual(m[0, 5, 5], 0)
        # water under the bridge is dropped, bridge surface is plane 0
        self.assertTrue(m[0, 20, 20])
        # objects on plane 1 of a bridge column land on plane 0 only if
        # their own tile is a bridge tile; (20, 21) isn't
        self.assertEqual(m[1, 20, 21], 0)

    def test_door(self):
        m, doors = build([(3, 10, 10, 0, 0, 1)])
        self.assertFalse(m[0, 10, 10] & MOVE_N)
        self.assertTrue(doors[0, 10, 10] & MOVE_N)
        self.assertTrue(doors[0, 10, 11] & MOVE_S)
        self.assertFalse(doors[0, 10, 10] & MOVE_E)


class TestPathfinding(TestCase):

    def setUp(self):
        # room with walls on all sides of (20..24, 20..24) and a door on
        # the north side at x=22
        locs = []
        for i in range(20, 25):
            locs.append((1, i, 20, 0, 0, 3))   # south
            locs.append((1, 20, i, 0, 0, 0))   # west
            locs.append((1, 24, i, 0, 0, 2))   # east
            if i != 22:
                locs.append((1, i, 24, 0, 0, 1))  # north
        locs.append((3, 22, 24, 0, 0, 1))
        builder = CollisionBuilder(definitions())
        builder.add_region(region(locs))
        rid = region_id(RX, RY)
        self.tmp = tempfile.TemporaryDirectory()
        path = os.path.join(self.tmp.name, 'movement.npz')
        numpy.savez_compressed(
            path, **{f'm{rid}': builder.movement(rid),
                     f'd{rid}': builder.doors(rid),
                     'meta': numpy.array(json.dumps({'cache': {}}))})
        self.cm = CollisionMap(path)

    def tearDown(self):
        self.cm._npz.close()
        self.tmp.cleanup()

    def t(self, x, y):
        return BASE_X + x, BASE_Y + y, 0

    def test_open_path_is_shortest(self):
        path = self.cm.find_path(self.t(0, 0), self.t(10, 5))
        self.assertEqual(len(path) - 1, 10)  # diagonal steps count as one

    def test_path_into_room_uses_door(self):
        path = self.cm.find_path(self.t(22, 30), self.t(22, 22))
        self.assertIsNotNone(path)
        doors = self.cm.doors_on_path(path)
        self.assertEqual(len(doors), 1)
        self.assertEqual(path[doors[0]], self.t(22, 25))

    def test_no_path_without_doors(self):
        self.assertIsNone(
            self.cm.find_path(self.t(22, 30), self.t(22, 22), doors=False))

    def test_unwalkable_target(self):
        self.assertIsNone(self.cm.find_path(self.t(0, 0), self.t(64, 64)))

    def test_can_step(self):
        self.assertTrue(self.cm.can_step(*self.t(10, 10), 1, 1))
        self.assertFalse(self.cm.can_step(*self.t(22, 25), 0, -1))
        self.assertTrue(self.cm.can_step(*self.t(22, 25), 0, -1, doors=True))
