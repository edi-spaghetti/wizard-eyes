import random
import re
from unittest import TestCase, mock

from wizard_eyes.navigation import walker as walker_module
from wizard_eyes.navigation.banks import (BankRoute, bank_at, bank_objects,
                                          load_banks)
from wizard_eyes.navigation.collision_map import STEPS, CollisionMap
from wizard_eyes.navigation.walker import Walker


def inside(x, y):
    return 0 <= x < 20 and 0 <= y < 20


def open_field():
    """20x20 open tiles, as a collision map without a data file."""
    cm = object.__new__(CollisionMap)
    cm.mask = lambda x, y, z: sum(
        bit for bit, dx, dy in STEPS
        if inside(x, y) and inside(x + dx, y + dy))
    cm.door_mask = lambda x, y, z: 0
    return cm


class Base(TestCase):

    def setUp(self):
        self.app = mock.MagicMock()
        self.app.msg = []
        self.pos = (2, 2, 0)
        mm = self.app.client.gauges.minimap
        mm.gps.get_coordinates.side_effect = lambda **k: self.pos
        mm.get_bbox.return_value = (0, 0, 150, 150)
        mm.orb_xy, mm.safe_radius, mm.tile_size = (75, 75), 65, 4
        self.now = 1000.0
        mock.patch.object(walker_module.time, 'time',
                          side_effect=lambda: self.now).start()
        self.addCleanup(mock.patch.stopall)

    def steps(self, thing, n):
        result = None
        for _ in range(n):
            self.now += 0.5
            result = thing.step()
            if result is not None:
                break
        return result


class TestWalker(Base):

    def test_arrives(self):
        w = Walker(self.app, [(10, 10, 0), (19, 2, 0)], cm=open_field())
        self.assertIsNone(self.steps(w, 1))
        self.assertEqual(w.target, (10, 10, 0))
        self.assertEqual(w.clicks, 1)
        self.pos = (10, 10, 0)
        self.assertTrue(self.steps(w, 10))

    def test_already_there(self):
        w = Walker(self.app, [self.pos], cm=open_field())
        self.assertTrue(self.steps(w, 10))
        self.assertEqual(w.clicks, 0)

    def test_gives_up_when_not_moving(self):
        w = Walker(self.app, [(10, 10, 0)], cm=open_field())
        self.assertIs(self.steps(w, 100), False)
        self.assertLessEqual(w.clicks, Walker.MAX_FAILURES)
        self.assertTrue(w.failed)

    def test_stops_before_clicking_again_after_a_failure(self):
        w = Walker(self.app, [(12, 2, 0)], cm=open_field())
        self.steps(w, 1)
        self.pos = (2, 9, 0)  # off route, moving
        self.steps(w, 1)
        self.assertTrue(w.settling)
        for y in (8, 7, 6):
            self.pos = (2, y, 0)
            self.steps(w, 1)
        self.assertEqual(w.clicks, 1)
        self.steps(w, 5)
        self.assertFalse(w.settling)
        self.assertEqual(w.clicks, 2)

    def test_gives_up_without_path_or_position(self):
        w = Walker(self.app, [(90, 90, 0)], cm=open_field())
        self.assertIs(self.steps(w, 2), False)
        self.pos = (-1, -1, -1)
        w = Walker(self.app, [(5, 5, 0)], cm=open_field())
        self.assertIs(self.steps(w, 100), False)


class TestBankRoute(Base):

    def setUp(self):
        super().setUp()
        self.obj = mock.MagicMock(x=11, y=10, plane=0, type=10,
                                  size=(1, 1), actions=['Bank'])
        self.om = mock.MagicMock()
        self.om.find.return_value = [self.obj]
        self.om.path_to.return_value = [(2, 2, 0), (10, 10, 0)]

    def route(self, names):
        return BankRoute(self.app, names, om=self.om, cm=open_field(),
                         banks={'a': (10, 10, 0)}, rng=random.Random(1))

    def test_walks_then_has_entity(self):
        r = self.route('a')
        self.assertIsNone(self.steps(r, 3))
        self.assertIsNone(r.entity)
        self.pos = (10, 10, 0)
        self.assertTrue(self.steps(r, 10))
        self.assertIsNotNone(r.entity)
        self.assertIsNone(r.wall)

    def test_unreachable(self):
        self.om.path_to.return_value = None
        self.assertIs(self.steps(self.route('a'), 5), False)
        self.app.client.screen.click_aoi.assert_not_called()

    def test_unknown_name(self):
        with self.assertRaises(ValueError):
            self.route('nowhere')

    def test_deposit_booths_left_out(self):
        self.obj.actions = ['Bank', 'Deposit-box']
        self.assertEqual(bank_objects(self.om, (1, 1, 0)), [])

    def test_bank_at(self):
        banks = {'a': (10, 10, 0), 'b': (40, 10, 0)}
        self.assertEqual(bank_at((12, 9, 0), banks), 'a')
        self.assertIsNone(bank_at((12, 9, 1), banks))
        self.assertIsNone(bank_at((200, 9, 0), banks))

    def test_config(self):
        for name, tile in load_banks().items():
            self.assertTrue(re.match('^[a-z_]+$', name))
            self.assertEqual(len(tile), 3)
