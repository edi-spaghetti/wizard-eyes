from unittest import TestCase, mock

import numpy

from wizard_eyes.screen_tools import Screen

BLACK = numpy.zeros((10, 10, 4), numpy.uint8)
IMAGE = BLACK + 50


class TestWindowGrab(TestCase):

    def setUp(self):
        self.screen = Screen(None)
        self.window = mock.patch.object(Screen, 'grab_window').start()
        self.desktop = mock.patch.object(Screen, 'grab_screen').start()
        self.addCleanup(mock.patch.stopall)
        self.desktop.return_value = IMAGE

    def test_uses_window(self):
        self.window.return_value = IMAGE
        self.screen.grab_client(1, 0, 0, 9, 9)
        # once it has worked, a black frame (loading) doesn't switch it
        self.window.return_value = BLACK
        self.screen.grab_client(1, 0, 0, 9, 9)
        self.desktop.assert_not_called()

    def test_black_falls_back(self):
        self.window.return_value = BLACK
        with self.assertLogs('wizard_eyes.screen_tools', 'WARNING') as logs:
            for _ in range(2):
                self.assertIs(self.screen.grab_client(1, 0, 0, 9, 9), IMAGE)
        self.assertEqual(len(logs.output), 1)
        self.assertEqual(self.window.call_count, 1)

    def test_window_coordinates(self):
        mock.patch.stopall()
        with mock.patch('wizard_eyes.screen_tools.win32gui.GetWindowRect',
                        return_value=(-8, -8, 100, 100)), \
                mock.patch.object(Screen, '_grab') as grab:
            self.screen.grab_window(7, -8, 12, 92, 92)
        grab.assert_called_once_with(7, 0, 20, 100, 100)
