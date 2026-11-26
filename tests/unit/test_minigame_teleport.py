import tempfile
from pathlib import Path
from unittest import TestCase, mock

import numpy

from wizard_eyes import minigame_teleport as mt
from wizard_eyes.minigame_teleport import (Cooldowns, MinigameTeleport,
                                           is_locked)
from wizard_eyes.teleports import Teleports


class TestMinigameTeleport(TestCase):

    def setUp(self):
        rng = numpy.random.RandomState(0)
        self.templates = {n: rng.randint(0, 255, s, numpy.uint8) for n, s in
                          (('title', (16, 78)), ('spell', (21, 21)),
                           ('soul_wars', (64, 62)), ('continue', (16, 149)))}
        mock.patch.object(mt, 'load_template',
                          side_effect=self.templates.get).start()
        self.scroll = mock.patch.object(mt.pyautogui, 'scroll').start()
        self.now = 1000.0
        mock.patch.object(mt.time, 'time',
                          side_effect=lambda: self.now).start()
        self.addCleanup(mock.patch.stopall)
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.cooldowns = Cooldowns('x', Path(self.dir.name) / 'x.local.json')
        self.app = mock.MagicMock()
        self.app.msg = []
        c = self.app.client
        c.get_bbox.return_value = (0, 0, 1400, 1000)
        c.tabs.inventory.located = True
        c.tabs.inventory.get_bbox.return_value = (1039, 966, 1071, 1001)
        self.pos = (2536, 3573, 0)
        c.gauges.minimap.gps.get_coordinates.side_effect = lambda: self.pos
        self.img = numpy.full((1000, 1400), 60, numpy.uint8)
        self.put('spell', 1200, 700)
        self.show()
        self.tele = MinigameTeleport(
            self.app, Teleports().methods['minigame_soul_wars'],
            self.cooldowns)

    def put(self, name, x, y):
        t = self.templates[name]
        self.img[y:y + t.shape[0], x:x + t.shape[1]] = t

    def show(self):
        self.app.client.img = self.img
        self.app.client.original_img = numpy.dstack([self.img] * 4)

    def steps(self, n):
        for _ in range(n):
            self.now += 0.5
            result = self.tele.step()
            if result is not None:
                return result

    def open_menu(self, emblem=True):
        self.steps(2)
        self.assertEqual(self.tele.state, 'menu')
        self.put('title', 540, 274)
        if emblem:
            self.put('soul_wars', 581, 400)
        self.show()

    def test_casts_and_lands(self):
        self.open_menu()
        self.steps(3)
        self.assertEqual(self.tele.state, 'leaving')
        x1, y1, x2, y2 = self.app.client.screen.click_aoi.call_args[0]
        self.assertTrue(581 < x1 < x2 < 800 and 400 < y1 < y2 < 464)
        self.steps(20)
        self.pos = (2211, 2858, 0)
        self.assertTrue(self.steps(10))
        self.assertGreater(self.cooldowns.left('minigame'), 1100)

    def test_interrupted(self):
        self.open_menu()
        self.assertIs(self.steps(100), False)
        self.assertEqual(self.cooldowns.left('minigame'), 0)

    def test_continues_the_dialogue(self):
        self.open_menu()
        self.steps(3)
        self.assertEqual(self.tele.state, 'leaving')
        self.put('continue', 186, 966)      # (chat box, bottom left)
        self.show()
        self.steps(1)
        keys = self.app.client.screen.press_key
        keys.assert_called_once_with('space')
        self.steps(3)                       # still showing: not again yet
        self.assertEqual(keys.call_count, 1)
        self.steps(10)                      # at most twice
        self.assertEqual(keys.call_count, 2)
        self.steps(30)
        self.assertEqual(keys.call_count, 2)
        self.img[:] = 60
        self.show()
        self.steps(10)                      # the 25 s count from the last one
        self.pos = (2211, 2858, 0)
        self.assertTrue(self.steps(10))

    def test_no_dialogue_no_keys(self):
        self.open_menu()
        self.steps(3)
        self.steps(20)
        self.app.client.screen.press_key.assert_not_called()

    def test_scrolls_until_the_list_ends(self):
        self.open_menu(emblem=False)
        self.assertIs(self.steps(20), False)
        self.assertEqual(self.scroll.call_count, 2)
        self.app.client.screen.press_key.assert_called_once_with('esc')

    def test_scrolls_to_it(self):
        self.open_menu(emblem=False)
        self.steps(5)
        self.put('soul_wars', 581, 450)
        self.show()
        self.steps(3)
        self.assertEqual(self.tele.state, 'leaving')

    def test_locked(self):
        self.open_menu()
        with mock.patch.object(mt, 'is_locked', return_value=True):
            self.assertIs(self.steps(5), False)
        self.assertEqual(
            self.app.client.screen.click_aoi.call_count, 1)  # the spell

    def test_opens_the_magic_tab_twice_at_most(self):
        self.img[:] = 60
        self.show()
        self.assertIs(self.steps(60), False)
        calls = self.app.client.screen.click_aoi.call_args_list
        self.assertEqual(len(calls), 2)
        x1, y1, x2, y2 = calls[0][0]
        self.assertTrue(1138 < x1 < 1154 < x2 < 1171)

    def test_cooldown(self):
        self.cooldowns.start('minigame', 600)
        self.assertIs(self.steps(1), False)
        self.app.client.screen.click_aoi.assert_not_called()

    def test_menu_does_not_open(self):
        self.assertIs(self.steps(30), False)

    def test_red_text(self):
        tile = numpy.full((70, 220, 4), 60, numpy.uint8)
        self.assertFalse(is_locked(tile))
        tile[30:34, 100:110] = (0, 0, 255, 255)
        self.assertTrue(is_locked(tile))

    def test_every_minigame_has_a_template_name(self):
        methods = Teleports().methods
        for key, m in methods.items():
            if key.startswith('minigame_'):
                self.assertEqual(m.minigame, key[len('minigame_'):])
