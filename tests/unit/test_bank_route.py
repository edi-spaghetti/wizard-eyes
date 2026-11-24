import random
from unittest import TestCase, mock

from wizard_eyes.navigation import walker as walker_module
from wizard_eyes import item_teleport
from wizard_eyes.item_teleport import ItemTeleport
from wizard_eyes.navigation.banks import BankTrip
from wizard_eyes.navigation.collision_map import STEPS, CollisionMap
from wizard_eyes.navigation import map_labels
from wizard_eyes.navigation.npc_map import NPCMap
from wizard_eyes.navigation.object_map import MapObject
from wizard_eyes.navigation.walker import Walker, in_view
from wizard_eyes.teleports import Teleports


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


    def test_done_ends_the_walk_while_moving(self):
        seen = []
        w = Walker(self.app, [(18, 2, 0)], cm=open_field(),
                   done=lambda: bool(seen))
        self.steps(w, 1)
        self.pos = (6, 2, 0)
        self.assertIsNone(self.steps(w, 1))
        seen.append(1)
        self.pos = (8, 2, 0)
        self.assertTrue(self.steps(w, 1))
        self.assertEqual(w.clicks, 1)

    def test_done_not_believed_through_a_wall(self):
        cm = open_field()
        w = Walker(self.app, [(6, 2, 0)], cm=cm, done=lambda: True)
        w.target = (6, 2, 0)
        cm.find_path = lambda *a, **k: None
        self.assertFalse(w.close_enough(self.pos))
        cm.find_path = lambda *a, **k: [self.pos] * 30
        self.assertFalse(w.close_enough(self.pos))

    def test_already_close_enough_needs_no_click(self):
        w = Walker(self.app, [(6, 2, 0)], cm=open_field(),
                   done=lambda: True)
        self.assertTrue(self.steps(w, 1))
        self.assertEqual(w.clicks, 0)

    def test_spread_click_is_followed_to_its_end(self):
        w = Walker(self.app, [(12, 2, 0)], cm=open_field(), spread=4,
                   done=lambda: False)
        self.steps(w, 1)
        goal = w.expect.goal
        self.pos = (goal[0] - 2, 2, 0)
        self.steps(w, 1)
        self.assertEqual(w.clicks, 1)
        # never in view: keeps going to the exact tile
        for _ in range(30):
            self.pos = w.expect.goal if w.expect else self.pos
            if self.steps(w, 1):
                break
        self.assertTrue(w.arrived)

    def test_spread_varies_the_last_click(self):
        goals = set()
        for _ in range(60):
            w = Walker(self.app, [(12, 2, 0)], cm=open_field(), spread=4)
            self.steps(w, 1)
            goals.add(w.expect.goal[0])
        self.assertGreater(len(goals), 2)
        self.assertTrue(goals <= {8, 9, 10, 11, 12})


class TestInView(Base):

    def test_in_view(self):
        client = self.app.client
        client.get_bbox.return_value = (0, 0, 1000, 1000)
        client.game_screen.is_clickable.return_value = True
        entity = mock.MagicMock()
        for bbox, expected in (((450, 450, 530, 530), True),
                               ((150, 450, 230, 530), True),
                               ((450, 750, 530, 830), True),
                               ((150, 750, 230, 830), False),
                               ((750, 150, 830, 230), False),
                               (None, False)):
            entity.get_bbox.return_value = bbox
            self.assertIs(in_view(client, entity), expected)
        entity.get_bbox.return_value = (450, 450, 530, 530)
        client.game_screen.is_clickable.return_value = False
        self.assertFalse(in_view(client, entity))


class TestBankTrip(Base):

    def setUp(self):
        super().setUp()
        self.obj = mock.MagicMock(x=11, y=10, plane=0, type=10,
                                  size=(1, 1), actions=['Bank'],
                                  spec=MapObject, id=1,
                                  names=['Bank booth'])
        self.obj.tile = (11, 10, 0)
        self.obj.distance.return_value = 1
        self.app.client.get_bbox.return_value = (0, 0, 1000, 1000)
        self.entity = \
            self.app.client.game_screen.create_game_entity.return_value
        self.entity.get_bbox.return_value = (900, 900, 980, 980)
        self.om = mock.MagicMock()
        self.om.find.return_value = [self.obj]
        self.om.interaction_tiles.return_value = [(10, 10, 0)]
        self.om.path_to.return_value = [(2, 2, 0), (10, 10, 0)]

    def route(self, teleports=()):
        return BankTrip(self.app, teleports, om=self.om, cm=open_field(),
                        rng=random.Random(1))

    def test_walks_then_has_entity(self):
        r = self.route()
        self.assertIsNone(self.steps(r, 3))
        self.assertIsNone(r.entity)
        self.pos = (10, 10, 0)
        self.assertTrue(self.steps(r, 10))
        self.assertIsNotNone(r.entity)
        self.assertIsNone(r.wall)

    def test_booth_in_view_is_enough(self):
        r = self.route()
        self.steps(r, 3)
        self.pos = (6, 6, 0)
        self.entity.get_bbox.return_value = (500, 500, 580, 580)
        self.assertTrue(self.steps(r, 1))
        self.assertIs(r.entity, self.entity)
        self.assertGreater(r.seconds_left, 0)

    def test_unreachable(self):
        self.om.interaction_tiles.return_value = [(90, 90, 0)]
        self.assertIs(self.steps(self.route(), 5), False)
        self.app.client.screen.click_aoi.assert_not_called()

    def test_unknown_or_unsupported_teleports(self):
        for name in ('nowhere', 'minigame_castle_wars'):
            with self.assertRaises(ValueError):
                self.route([name])

    def test_close_bank_is_walked_to(self):
        r = self.route(['strength_cape'])
        self.steps(r, 1)
        self.assertIsNone(r.tele)
        self.assertIsNotNone(r.walker)

    def test_teleports_then_walks(self):
        self.om.interaction_tiles.return_value = [(90, 90, 0)]
        r = self.route(['strength_cape'])
        self.steps(r, 1)
        self.assertIsInstance(r.tele, ItemTeleport)
        r.tele = mock.MagicMock()
        r.tele.step.return_value = True
        self.om.interaction_tiles.return_value = [(10, 10, 0)]
        self.steps(r, 2)
        self.assertIsNotNone(r.walker)

    def test_each_teleport_tried_once(self):
        self.om.interaction_tiles.return_value = [(90, 90, 0)]
        r = self.route(['strength_cape', 'varrock_tablet'])
        tried = []
        for _ in range(2):
            self.steps(r, 1)
            tried.append(r.tele.method.key)
            r.tele.failed = 'no'
            self.steps(r, 1)
        self.assertEqual(sorted(tried), ['strength_cape', 'varrock_tablet'])
        self.assertIs(self.steps(r, 2), False)

    def test_bankers(self):
        npcs = NPCMap()
        self.assertTrue(npcs.find('banker'))
        self.assertEqual({n.wander for n in npcs.find('banker')}, {0})
        self.om.find.return_value = []
        home = npcs.find('banker')[0].tile
        found = map_labels.find_banks(home, 12, om=self.om, npcs=npcs)
        self.assertEqual(found[0].tile, home)
        upstairs = (home[0], home[1], 3)
        self.assertEqual(
            map_labels.find_banks(upstairs, 12, om=self.om, npcs=npcs), [])

    def test_counters_only_when_asked(self):
        self.obj.type = 0
        for walls in (True, False):
            found = map_labels.find_banks((11, 11, 0), 12, walls=walls,
                                          om=self.om, npcs=NPCMap())
            self.assertEqual(bool(found), walls)


class TestItemTeleport(Base):

    def setUp(self):
        super().setUp()
        self.inv = self.app.client.tabs.inventory
        self.inv.state = 'inventory_selected'
        self.icons = {str(i): mock.MagicMock(state='something')
                      for i in range(3)}
        self.inv.interface.icons = self.icons
        self.app.client.mouse_options.state = ''
        self.use = mock.patch.object(item_teleport, 'UseOption').start()
        self.use.return_value.step.return_value = True
        self.tele = ItemTeleport(
            self.app, Teleports().methods['strength_cape'])

    def run_to(self, state):
        for _ in range(10):
            self.steps(self.tele, 1)
            if self.tele.state == state:
                return
        self.fail(self.tele.state)

    def test_lands(self):
        self.app.client.mouse_options.state = 'Teleport | Strength cape(t)'
        self.run_to('leaving')
        self.pos = (2866, 3545, 0)
        self.assertTrue(self.steps(self.tele, 10))

    def test_wrong_landing(self):
        self.app.client.mouse_options.state = 'Teleport | Strength cape(t)'
        self.run_to('leaving')
        self.pos = (3000, 3000, 0)
        self.assertIs(self.steps(self.tele, 10), False)

    def test_does_not_move(self):
        self.app.client.mouse_options.state = 'Teleport | Strength cape(t)'
        self.run_to('leaving')
        self.assertIs(self.steps(self.tele, 100), False)

    def test_not_in_inventory(self):
        self.assertIs(self.steps(self.tele, 100), False)
        self.use.assert_not_called()
