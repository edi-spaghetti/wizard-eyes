import unittest

from wizard_eyes.navigation.still_npc import middle_of_tile


class TestMiddle(unittest.TestCase):

    def test_neighbours(self):
        tile = (900, 500, 80, 80)
        self.assertEqual(middle_of_tile(tile, 0, 0), (920, 520, 960, 560))
        # one west, two north
        self.assertEqual(middle_of_tile(tile, -1, 2), (840, 360, 880, 400))


if __name__ == '__main__':
    unittest.main()
