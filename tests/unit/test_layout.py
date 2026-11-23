"""Layout detection on synthetic frames built from the real tab templates.

Positions are taken from real captures of each layout, then the whole UI is
shifted around to check only relative positions matter.
"""
from unittest import TestCase, mock

import numpy

from wizard_eyes import login
from wizard_eyes.startup_checks import check_layout


def frame(parts, offset=(0, 0), size=(1168, 1936)):
    rng = numpy.random.default_rng(0)
    img = rng.integers(0, 40, size=size, dtype=numpy.uint8)  # dark noise
    ox, oy = offset
    for name, (x, y) in parts.items():
        t = login.template(name)
        img[oy + y:oy + y + t.shape[0], ox + x:ox + x + t.shape[1]] = t
    return img


# positions measured from real screenshots of each layout
MODERN = {'ingame_inventory_tab': (1598, 1124), 'ingame_prayer_tab': (1664, 1124),
          'ingame_emotes_tab': (1862, 1124), 'ingame_music_tab': (1895, 1124),
          'ingame_chat_all_selected': (13, 1137)}
CLASSIC = {'ingame_inventory_tab': (1791, 825), 'ingame_prayer_tab': (1857, 825),
           'ingame_emotes_tab': (1857, 1123), 'ingame_chat_all_selected': (13, 1137)}
FIXED = {'ingame_inventory_tab': (1211, 199), 'ingame_prayer_tab': (1277, 199),
         'ingame_emotes_tab': (1277, 497), 'ingame_chat_all_selected': (590, 511)}


class TestDetectLayout(TestCase):

    def test_layouts(self):
        self.assertEqual(login.detect_layout(frame(MODERN)), login.MODERN)
        self.assertEqual(login.detect_layout(frame(CLASSIC)), login.RESIZABLE_CLASSIC)
        self.assertEqual(login.detect_layout(frame(FIXED)), login.FIXED_CLASSIC)

    def test_position_independent(self):
        moved = {k: (x - 700, y - 300) for k, (x, y) in MODERN.items()}
        self.assertEqual(login.detect_layout(frame(moved)), login.MODERN)
        self.assertEqual(login.detect_layout(frame(FIXED, offset=(-500, 400))),
                         login.FIXED_CLASSIC)

    def test_unknown_without_tabs(self):
        only_chat = {'ingame_chat_all_selected': (13, 1137)}
        self.assertEqual(login.detect_layout(frame(only_chat)), login.UNKNOWN)

    def test_classify_in_game_anywhere(self):
        # chat button mid-window (fixed layout) is found by the fallback
        state, _ = login.classify(frame({'ingame_chat_all_selected': (590, 511)}))
        self.assertEqual(state, 'in_game')
        self.assertEqual(login.classify(frame({}))[0], 'loading')


class TestLayoutCache(TestCase):

    def run_check(self, cache, result, recheck=False, fixed=False):
        with mock.patch('wizard_eyes.login.check_layout',
                        return_value=result) as m, \
                mock.patch('wizard_eyes.display_settings.set_modern_layout',
                           return_value=fixed):
            ok = check_layout(cache, recheck, mock.Mock(), 1, 'RuneLite - X')
        return ok, m.called

    def test_pass_is_cached(self):
        cache = {}
        self.assertEqual(self.run_check(cache, login.MODERN), (True, True))
        self.assertEqual(self.run_check(cache, login.MODERN), (True, False))
        self.assertEqual(self.run_check(cache, login.MODERN, recheck=True),
                         (True, True))

    def test_failure_not_cached(self):
        cache = {}
        self.assertEqual(self.run_check(cache, login.FIXED_CLASSIC), (False, True))
        self.assertEqual(self.run_check(cache, login.FIXED_CLASSIC), (False, True))

    def test_switched_layout_is_cached(self):
        cache = {}
        self.assertEqual(self.run_check(cache, login.FIXED_CLASSIC, fixed=True),
                         (True, True))
        self.assertEqual(self.run_check(cache, login.FIXED_CLASSIC),
                         (True, False))

    def test_expiry(self):
        cache = {'layout': {'RuneLite - X': {'layout': login.MODERN,
                                             'checked_at': 0}}}
        self.assertEqual(self.run_check(cache, login.MODERN), (True, True))
