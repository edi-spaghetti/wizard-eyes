"""Matching options in mouse text and OCR'd right-click menu lines."""
import re
from unittest import TestCase

from wizard_eyes.interactions import (
    best_menu_line, menu_line_matches, mouse_text_pattern,
)


class TestMenuMatching(TestCase):

    def test_menu_lines(self):
        # real OCR output from the game
        self.assertTrue(menu_line_matches('collectbbankbookth', 'Collect',
                                          ['Bank booth']))
        self.assertTrue(menu_line_matches('bankbankbookh', 'Bank',
                                          ['Bank booth']))
        self.assertTrue(menu_line_matches('climb-downstaircase',
                                          'Climb-down', ['Staircase']))
        self.assertTrue(menu_line_matches('climbstaircase', 'Climb',
                                          ['Staircase']))
        # wrong option, or a longer option starting the same way
        self.assertFalse(menu_line_matches('collectbbankbookth', 'Bank',
                                           ['Bank booth']))
        self.assertFalse(menu_line_matches('climb-upstaircase', 'Climb',
                                           ['Staircase']))
        self.assertFalse(menu_line_matches('examinebankbookh', 'Collect',
                                           ['Bank booth']))
        # right option, different thing
        self.assertFalse(menu_line_matches('collectgrandexchangeclerk',
                                           'Collect', ['Bank booth']))

    def test_mouse_text(self):
        p = mouse_text_pattern('Climb-up')
        self.assertTrue(re.search(p, '12moreoptionsClimb-up | Staircase'))
        self.assertFalse(re.search(mouse_text_pattern('Climb'),
                                   'Climb-up 2moreoptions | Staircase'))
        self.assertTrue(re.search(mouse_text_pattern('Bank'),
                                  'Bank 13moreoptions | Bankbooth'))
        # with names: the thing hovered must be one of them
        self.assertTrue(re.search(
            mouse_text_pattern('Tele to POH', ['Construct. cape(t)']),
            'TeletoPOH 15moreoptions | Constructcape(t) | '))
        booth = mouse_text_pattern('Collect', ['Bank booth'])
        self.assertFalse(re.search(booth, 'Collect to bank | '))
        self.assertFalse(re.search(booth, 'Collecttobank | '))
        self.assertTrue(re.search(booth, 'Collect 2moreoptions | Bankbooth'))
        self.assertTrue(re.search(
            mouse_text_pattern('Bank', ['Bank booth']),
            'Bank 13moreoptions | Bank booth'))


class TestBestMenuLine(TestCase):
    """Real right-click menus read by OCR in game (first letters wrong)."""

    RING = ['wearkingofwealthiz)', 'featureskingofiwealthiz)121',
            'usekingofswealthliz)', 'kubkingofswealthilizz)2',
            'dropkingofswealthliz)', 'examinekingofwealthliz)', 'cancel']
    CAPE = ['leletopunhconstroctcapelt)', 'leleportconstroctcapelt)2',
            'wearconstructcapelt)', 'uselonstroctcapelt)',
            'uropconstroctcapelt)', 'examinelonstroctcapelt)', 'cancel']
    TABLET = ['breakyarrockteleport', 'yarrockyarrockteleport',
              'brandexchangeyarrockteleport', 'useyarrockteleport',
              'loggleyarrockteleport', 'uropyarrockteleport',
              'examineyarrockteleport', 'cancel']

    def test_picks_the_option(self):
        ring = ['Ring of wealth (i2)']
        self.assertEqual(best_menu_line(self.RING, 'Rub', ring), 3)
        self.assertEqual(best_menu_line(self.RING, 'Wear', ring), 0)
        cape = ['Construct. cape(t)']
        self.assertEqual(best_menu_line(self.CAPE, 'Tele to POH', cape), 0)
        self.assertEqual(best_menu_line(self.CAPE, 'Teleport', cape), 1)
        tab = ['Varrock teleport']
        self.assertEqual(best_menu_line(self.TABLET, 'Grand Exchange', tab),
                         2)
        self.assertEqual(best_menu_line(self.TABLET, 'Break', tab), 0)

    def test_nothing_close_enough(self):
        self.assertIsNone(best_menu_line(self.RING, 'Grand Exchange',
                                         ['Ring of wealth (i2)']))
        self.assertIsNone(best_menu_line(self.TABLET, 'Rub',
                                         ['Varrock teleport']))
