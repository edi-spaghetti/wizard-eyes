import tempfile
import unittest
from pathlib import Path

from wizard_eyes.navigation.collision_map import (MOVE_E, MOVE_N, MOVE_NE,
                                                  MOVE_S, MOVE_SE, MOVE_W,
                                                  _STEP_BITS)
from wizard_eyes.navigation.house import (CHUNK, HOUSE_ORIGIN,
                                          HouseCollisionMap, HouseLayout,
                                          Room, cell_of, in_template_area,
                                          is_empty_template,
                                          rotate_local, rotate_mask,
                                          rotate_vector, rotation_from_step,
                                          unrotate_local)

ALL = 0xFF


class FakeBase:
    """Template collision: chunk (1856, 7048) open everywhere except its
    outer edge, which is open only in the middle of the east side (a
    doorway); chunk (1864, 7048) all open."""
    path = Path('fake')
    meta = {}

    def mask(self, x, y, z):
        if 1864 <= x < 1872 and 7048 <= y < 7056:
            return ALL
        if not (1856 <= x < 1864 and 7048 <= y < 7056):
            return 0
        lx, ly = x - 1856, y - 7048
        m = ALL
        if lx == 7 and ly != 3:
            m &= ~(MOVE_E | MOVE_NE | 0x20)
        return m


class TestRotation(unittest.TestCase):

    def test_round_trip(self):
        for r in range(4):
            for x in range(CHUNK):
                for y in range(CHUNK):
                    self.assertEqual(unrotate_local(*rotate_local(x, y, r),
                                                    r), (x, y))

    def test_vector_matches_tiles(self):
        for r in range(4):
            for step in ((0, 1), (1, 0), (1, 1), (-1, 1)):
                a = rotate_local(3, 3, r)
                b = rotate_local(3 + step[0], 3 + step[1], r)
                self.assertEqual((b[0] - a[0], b[1] - a[1]),
                                 rotate_vector(*step, r))

    def test_mask_matches_vector(self):
        for r in range(4):
            for (dx, dy), bit in _STEP_BITS.items():
                self.assertEqual(rotate_mask(bit, r),
                                 _STEP_BITS[rotate_vector(dx, dy, r)])
        self.assertEqual(rotate_mask(MOVE_N, 1), MOVE_E)
        self.assertEqual(rotate_mask(MOVE_N | MOVE_E, 2), MOVE_S | MOVE_W)

    def test_rotation_from_step(self):
        for r in range(4):
            house = rotate_vector(0, 2, r)
            self.assertEqual(rotation_from_step((0, 2), house), r)
        self.assertIsNone(rotation_from_step((0, 0), (1, 0)))
        # partly blocked: moved diagonally in the template
        self.assertEqual(rotation_from_step((1, 2), (0, -3)), 2)

    def test_template_area(self):
        self.assertTrue(in_template_area((1922, 7051, 1)))
        self.assertFalse(in_template_area((3165, 3479, 0)))
        self.assertFalse(in_template_area(None))

    def test_empty_cell(self):
        # grass of an empty cell, house styles 1 and 2
        self.assertTrue(is_empty_template((1866, 7047, 1)))
        self.assertTrue(is_empty_template((1933, 7043, 1)))
        # a garden and a room next to the grass chunk
        self.assertFalse(is_empty_template((1922, 7051, 1)))
        self.assertFalse(is_empty_template((1936, 7043, 1)))
        self.assertFalse(is_empty_template((3000, 7043, 0)))


class TestLayout(unittest.TestCase):

    def layout(self):
        return HouseLayout('test', [
            Room((0, 0, 1), (1856, 7048, 1), 0),
            Room((1, 0, 1), (1864, 7048, 1), 1),
            Room((0, 1, 1), (1864, 7048, 1), 2),
        ], path=Path(tempfile.gettempdir()) / 'we_house_test.json')

    def test_to_house_and_back(self):
        lay = self.layout()
        t = (1858, 7050, 1)
        h = lay.to_house(t)
        self.assertEqual(h, (HOUSE_ORIGIN[0] + 2, HOUSE_ORIGIN[1] + 2, 1))
        self.assertEqual(lay.to_template(h), t)

    def test_shared_template_needs_near(self):
        lay = self.layout()
        t = (1864, 7048, 1)
        self.assertIsNone(lay.to_house(t))
        east = lay.to_house(t, near=(HOUSE_ORIGIN[0] + 8, HOUSE_ORIGIN[1],
                                     1))
        north = lay.to_house(t, near=(HOUSE_ORIGIN[0], HOUSE_ORIGIN[1] + 9,
                                      1))
        self.assertEqual(cell_of(east), (1, 0, 1))
        self.assertEqual(cell_of(north), (0, 1, 1))
        for h in (east, north):
            self.assertEqual(lay.to_template(h), t)

    def test_save_load(self):
        lay = self.layout()
        lay.rooms[(0, 0, 1)].furniture['Jewellery box space'] = 'Box'
        lay.portal = (2954, 3224, 0)
        lay.save()
        back = HouseLayout.load('test', lay.path)
        self.assertEqual(back.portal, (2954, 3224, 0))
        self.assertEqual(set(back.rooms), set(lay.rooms))
        self.assertEqual(back.rooms[(0, 0, 1)].furniture,
                         {'Jewellery box space': 'Box'})
        self.assertEqual(back.rooms[(1, 0, 1)].rotation, 1)
        lay.path.unlink()


class TestHouseCollision(unittest.TestCase):

    def test_doorway(self):
        lay = HouseLayout('test', [
            Room((0, 0, 1), (1856, 7048, 1), 0),
            Room((1, 0, 1), (1864, 7048, 1), 3),
        ])
        cm = HouseCollisionMap(lay, FakeBase())
        ox, oy = HOUSE_ORIGIN
        # only the doorway at local y 3 crosses east
        self.assertTrue(cm.can_step(ox + 7, oy + 3, 1, 1, 0))
        self.assertFalse(cm.can_step(ox + 7, oy + 4, 1, 1, 0))
        # no diagonals between rooms
        self.assertFalse(cm.can_step(ox + 7, oy + 3, 1, 1, 1))
        # nothing outside the rooms
        self.assertFalse(cm.can_step(ox, oy, 1, -1, 0))
        path = cm.find_path((ox + 1, oy + 6, 1), (ox + 12, oy + 6, 1))
        self.assertIsNotNone(path)
        self.assertIn((ox + 8, oy + 3, 1), path)

    def test_turned_room(self):
        # the doorway room turned a half turn: its doorway faces west
        lay = HouseLayout('test', [
            Room((0, 0, 1), (1864, 7048, 1), 0),
            Room((1, 0, 1), (1856, 7048, 1), 2),
        ])
        cm = HouseCollisionMap(lay, FakeBase())
        ox, oy = HOUSE_ORIGIN
        # template local (7, 3) -> house local (0, 4)
        self.assertTrue(cm.can_step(ox + 8, oy + 4, 1, -1, 0))
        self.assertFalse(cm.can_step(ox + 8, oy + 3, 1, -1, 0))
        self.assertTrue(cm.can_step(ox + 7, oy + 4, 1, 1, 0))
        self.assertEqual(rotate_mask(MOVE_S | MOVE_SE, 2),
                         MOVE_N | 0x80)
        self.assertEqual(rotate_mask(MOVE_NE, 2), 0x40)
        self.assertEqual(rotate_mask(MOVE_W, 2), MOVE_E)


if __name__ == '__main__':
    unittest.main()
