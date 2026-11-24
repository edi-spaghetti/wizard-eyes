import unittest

import numpy

from wizard_eyes.icon_teleport import (MENU_BGR, icon_difference, load_icon,
                                       submenu_left)
from wizard_eyes.teleports import Teleports


class TestIconTeleport(unittest.TestCase):

    def test_icons_told_apart(self):
        icons = {m.icon for m in Teleports().methods.values() if m.icon}
        for a in icons:
            icon = load_icon(a)
            img = numpy.full((icon.shape[0] + 6, icon.shape[1] + 6, 3), 40,
                             numpy.uint8)
            img[3:3 + icon.shape[0], 2:2 + icon.shape[1]] = icon[..., :3]
            self.assertEqual(icon_difference(img, a), 0)
            for b in icons - {a}:
                self.assertGreater(icon_difference(img, b), 18, (a, b))

    def test_submenu_left(self):
        row = numpy.zeros((800, 3), numpy.uint8)
        self.assertIsNone(submenu_left(row, 700))
        row[560:697] = MENU_BGR
        self.assertEqual(submenu_left(row, 700), 560)

    def test_usable(self):
        usable = Teleports().usable()
        self.assertIn('royal_seed_pod', usable)
        self.assertIn('house_jewellery_box_edgeville', usable)
        self.assertNotIn('sailors_amulet_red_rock', usable)
        self.assertNotIn('construction_cape_house', usable)


if __name__ == '__main__':
    unittest.main()
