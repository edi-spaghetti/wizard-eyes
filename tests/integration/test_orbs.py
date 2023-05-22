import sys

import os.path
import unittest

from wizard_eyes.client import Client


class TestPrayerOrb(unittest.TestCase):

    def setUp(self):

        old_argv = sys.argv

        # path = os.path.join(
        #     os.path.dirname(__file__), '..', 'fixtures',
        #     '0_1684674291.2899742.png'
        # )
        # sys.argv = ['test.py', '--static-img', path]

        self.client = Client('RuneLite')
        self.client.post_init()

    def test_prayer_points(self):

        self.client.update()
        self.client.minimap.orb.prayer.update()
        self.assertEqual(self.client.minimap.orb.prayer.state, 90)
