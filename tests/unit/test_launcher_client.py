import json
import os
import tempfile
import unittest
from unittest import mock

from wizard_eyes import login


class LauncherClientTest(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, 'settings.json')
        patch = mock.patch.object(
            login, 'launcher_settings_path', return_value=self.path)
        patch.start()
        self.addCleanup(patch.stop)

    def write(self, client):
        with open(self.path, 'w') as f:
            json.dump({'language': 'en', 'gamePreferences': {
                'lastGamePlayed': 'osrs',
                'lastGameClientByGameId': {'osrs': client, 'runescape': 'x'}}}, f)

    def test_already_runelite_is_not_touched(self):
        self.write('osrs_runelite')
        with mock.patch.object(login, 'close_launcher') as close, \
                mock.patch.object(login, 'set_launcher_client') as change:
            login.ensure_runelite_selected()
        close.assert_not_called()
        change.assert_not_called()

    def test_other_client_is_switched_and_rest_kept(self):
        self.write('osrs_ehc')
        with mock.patch.object(login, 'close_launcher') as close:
            login.ensure_runelite_selected()
        close.assert_called_once()
        with open(self.path) as f:
            s = json.load(f)
        self.assertEqual(s['gamePreferences']['lastGameClientByGameId'],
                         {'osrs': 'osrs_runelite', 'runescape': 'x'})
        self.assertEqual(s['language'], 'en')

    def test_launcher_that_wont_close_is_not_edited(self):
        self.write('osrs_ehc')
        with mock.patch.object(login, 'close_launcher',
                               side_effect=login.LoginError('open')):
            with self.assertRaises(login.LoginError):
                login.ensure_runelite_selected()
        self.assertEqual(login.launcher_selected_client(), 'osrs_ehc')

    def test_no_settings_yet(self):
        login.ensure_runelite_selected()
        self.assertFalse(os.path.exists(self.path))


if __name__ == '__main__':
    unittest.main()
