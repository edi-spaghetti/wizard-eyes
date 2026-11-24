import unittest
from pathlib import Path

import numpy

from wizard_eyes.screen_text import menu_line
from wizard_eyes.teleports import Teleports, name_matches
from wizard_eyes.file_path_utils import get_root
from wizard_eyes.worn_teleport import (SLOTS, STATES, WornTeleport,
                                       worn_state)


class TestWornTeleport(unittest.TestCase):

    def test_ring_of_wealth(self):
        method = Teleports().methods['ring_of_wealth_ge']
        self.assertIn(method.slot, SLOTS)
        # as read off the game
        menu = ['Remove', '', 'Grand Exchange', 'Falador', 'Dondakan',
                'Boss Log', 'Coin Collection', 'Examine', 'Cancel']
        self.assertEqual(menu_line(menu, method.worn_option), 2)
        self.assertTrue(name_matches('Ring of wealth (3)', method.item))
        self.assertIsNone(menu_line(['Remove', 'Boss Log', 'Cancel'],
                                    method.worn_option))

    def test_ring_states(self):
        for name, state in STATES['ring'].items():
            self.assertNotEqual(worn_state(numpy.zeros((37, 37), 'uint8'),
                                           'ring'), state)
            tpl = numpy.load(str(Path(get_root()) / 'data/teleport_items/worn'
                                 / f'{name}.npy'))
            img = numpy.full((37, 37), 60, numpy.uint8)
            img[5:5 + tpl.shape[0], 6:6 + tpl.shape[1]] = tpl
            self.assertEqual(worn_state(img, 'ring'), state)

    def test_other_methods_refused(self):
        class App:
            client = None
        with self.assertRaises(ValueError):
            WornTeleport(App, Teleports().methods['varrock_tablet'])


if __name__ == '__main__':
    unittest.main()
