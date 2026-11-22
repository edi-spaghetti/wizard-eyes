"""Object index: build from synthetic regions, lookups, interaction tiles."""
import json
import os
import re
import tempfile
from unittest import TestCase

import numpy

from wizard_eyes.cache.collision import CollisionBuilder
from wizard_eyes.cache.maps import SETTING_BRIDGE, region_id
from wizard_eyes.cache.object_data import build_object_data
from wizard_eyes.cache.objects import ObjectDefinition
from wizard_eyes.navigation.collision_map import CollisionMap
from wizard_eyes.navigation.object_map import BLOCK_EAST, ObjectMap

from tests.unit.test_map_data import BASE_X, BASE_Y, RX, RY, region


def actions(*names):
    return list(names) + [None] * (5 - len(names))


def definitions():
    return {
        1: ObjectDefinition(1, name='Wall'),
        2: ObjectDefinition(2, name='Bank booth', actions=actions('Bank'),
                            size_x=1, size_y=1, access_block=BLOCK_EAST),
        3: ObjectDefinition(3, name='Door', actions=actions('Open')),
        4: ObjectDefinition(4, name='Table', size_x=3, size_y=1),
        5: ObjectDefinition(5),  # unnamed scenery: dropped
        6: ObjectDefinition(6, transforms=[7, -1]),  # varbit object
        7: ObjectDefinition(7, name='Ladder', actions=actions('Climb-up')),
        8: ObjectDefinition(8, name='Flowers', interact_type=0),
    }


def t(x, y, z=0):
    return BASE_X + x, BASE_Y + y, z


class TestObjectMap(TestCase):

    def setUp(self):
        locs = [
            # id, x, y, plane, type, orientation
            (2, 10, 10, 0, 10, 0),   # booth, east side blocked
            (2, 20, 10, 0, 10, 1),   # booth rotated: blocked side -> south
            (3, 30, 30, 0, 0, 1),    # door on the north edge of (30, 30)
            (4, 40, 40, 0, 10, 1),   # 3x1 table rotated -> 1x3
            (5, 5, 5, 0, 10, 0),
            (6, 50, 50, 0, 10, 0),
            (8, 12, 30, 0, 22, 0),
            (7, 60, 60, 1, 10, 0),   # ladder on a bridge: plane 0
        ]
        settings = numpy.zeros((4, 64, 64), numpy.uint8)
        settings[1, 60, 60] = SETTING_BRIDGE
        defs = definitions()
        reg = region(locs, settings)
        placements, used = build_object_data([reg], defs)
        builder = CollisionBuilder(defs)
        builder.add_region(reg)
        rid = region_id(RX, RY)

        self.tmp = tempfile.TemporaryDirectory()
        meta = numpy.array(json.dumps({'cache': {}}))
        opath = os.path.join(self.tmp.name, 'objects.npz')
        numpy.savez_compressed(
            opath, definitions=numpy.array(json.dumps(used)), meta=meta,
            **placements)
        cpath = os.path.join(self.tmp.name, 'movement.npz')
        numpy.savez_compressed(
            cpath, meta=meta, **{f'm{rid}': builder.movement(rid),
                                 f'd{rid}': builder.doors(rid)})
        self.om = ObjectMap(opath)
        self.cm = CollisionMap(cpath)

    def tearDown(self):
        self.cm._npz.close()
        self.tmp.cleanup()

    def test_unnamed_objects_dropped(self):
        self.assertEqual(len(self.om), 7)
        self.assertEqual(self.om.at(t(5, 5)), [])

    def test_find_by_name_and_action(self):
        booths = self.om.find('bank booth', near=t(0, 10))
        self.assertEqual([o.tile for o in booths], [t(10, 10), t(20, 10)])
        self.assertEqual(self.om.find('Bank booth', 'Collect'), [])
        self.assertEqual(len(self.om.find(action=re.compile('^Ban'))), 2)

    def test_radius(self):
        self.assertEqual(len(self.om.find('Bank booth', near=t(0, 10),
                                          radius=10)), 1)

    def test_transformed_object_matches_variant(self):
        obj = self.om.nearest(t(0, 0), 'Ladder', 'Climb-up')
        self.assertEqual(obj.id, 6)
        self.assertEqual(obj.name, 'Ladder')
        self.assertEqual(obj.actions, ['Climb-up'])

    def test_bridge_moves_object_down(self):
        self.assertEqual(self.om.find('Ladder', plane=1), [])
        obj = self.om.find('Ladder', near=t(60, 60))[0]
        self.assertEqual(obj.id, 7)
        self.assertEqual(obj.tile, t(60, 60, 0))

    def test_rotated_size_and_distance(self):
        table = self.om.find('Table')[0]
        self.assertEqual(table.size, (1, 3))
        self.assertEqual(len(table.footprint()), 3)
        self.assertEqual(table.distance(t(40, 44)), 2)
        self.assertEqual(self.om.at(t(40, 42)), [table])

    def test_access_block(self):
        booth = self.om.find('Bank booth', near=t(10, 10))[0]
        tiles = self.om.interaction_tiles(booth, self.cm)
        self.assertNotIn(t(11, 10), tiles)
        self.assertEqual(set(tiles), {t(9, 10), t(10, 9), t(10, 11)})
        rotated = self.om.find('Bank booth', near=t(20, 10))[0]
        tiles = self.om.interaction_tiles(rotated, self.cm)
        self.assertNotIn(t(20, 9), tiles)
        self.assertIn(t(21, 10), tiles)

    def test_door_tiles(self):
        door = self.om.find('Door')[0]
        self.assertEqual(set(self.om.interaction_tiles(door, self.cm)),
                         {t(30, 30), t(30, 31)})

    def test_walk_over_object(self):
        flowers = self.om.find('Flowers')[0]
        self.assertIn(t(12, 30), self.om.interaction_tiles(flowers, self.cm))

    def test_path_to_object(self):
        booth = self.om.find('Bank booth', near=t(10, 10))[0]
        path = self.om.path_to(booth, t(15, 10), self.cm)
        # east side is blocked: walk round to the north or south side
        self.assertIn(path[-1], {t(10, 9), t(10, 11)})

    def test_nearest_reachable(self):
        obj, path = self.om.nearest_reachable(t(25, 10), self.cm,
                                              'Bank booth')
        self.assertEqual(obj.tile, t(20, 10))
        self.assertEqual(path[0], t(25, 10))


    def test_path_to_closest_of_several_goals(self):
        path = self.cm.find_path(t(0, 0), [t(10, 0), t(3, 3), t(1, 30)])
        self.assertEqual(path[-1], t(3, 3))
        self.assertEqual(len(path) - 1, 3)
        self.assertEqual(self.cm.find_path(t(0, 0), [t(0, 0)]), [t(0, 0)])
        self.assertIsNone(self.cm.find_path(t(0, 0), [t(10, 10, 1)]))
