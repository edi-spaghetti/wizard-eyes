"""Matching options in mouse text and OCR'd right-click menu lines."""
import re
from unittest import TestCase

from wizard_eyes.interactions import menu_line_matches, mouse_text_pattern


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
        booth = mouse_text_pattern('Collect', ['Bank booth'])
        self.assertFalse(re.search(booth, 'Collect to bank | '))
        self.assertFalse(re.search(booth, 'Collecttobank | '))
        self.assertTrue(re.search(booth, 'Collect 2moreoptions | Bankbooth'))
        self.assertTrue(re.search(
            mouse_text_pattern('Bank', ['Bank booth']),
            'Bank 13moreoptions | Bank booth'))
