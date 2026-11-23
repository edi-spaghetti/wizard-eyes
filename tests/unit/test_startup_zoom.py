import unittest

from wizard_eyes.startup_checks import zoom_needs_reset


class ZoomResetTest(unittest.TestCase):

    def test_wrong_zoom_is_always_reset(self):
        self.assertTrue(zoom_needs_reset(False, True))
        self.assertTrue(zoom_needs_reset(False, False))

    def test_right_zoom_is_left_alone(self):
        self.assertFalse(zoom_needs_reset(True, True))
        self.assertFalse(zoom_needs_reset(True, False))

    def test_unknown_zoom_is_reset_only_without_a_pass(self):
        self.assertFalse(zoom_needs_reset(None, True))
        self.assertTrue(zoom_needs_reset(None, False))


if __name__ == '__main__':
    unittest.main()
