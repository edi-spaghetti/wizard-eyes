import json
import os
import shutil
import tempfile
from unittest import TestCase, mock

from wizard_eyes import login


class TestLauncherSettings(TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        os.makedirs(os.path.join(self.dir, 'Jagex Launcher'))

    def write_settings(self, data):
        with open(os.path.join(self.dir, 'Jagex Launcher', 'settings.json'), 'w') as f:
            json.dump(data, f)

    def test_selected_character_id(self):
        self.write_settings({'gamePreferences': {
            'lastGameAccountByGameId': {'osrs': '123', 'runescape': None}}})
        with mock.patch.dict(os.environ, {'LOCALAPPDATA': self.dir}):
            self.assertEqual(login.launcher_selected_character_id(), '123')

    def test_missing_settings(self):
        with mock.patch.dict(os.environ, {'LOCALAPPDATA': self.dir}):
            self.assertIsNone(login.launcher_selected_character_id())


class TestAccountCache(TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        p = mock.patch.object(login, 'accounts_cache_path',
                              return_value=os.path.join(self.dir, 'a.json'))
        p.start()
        self.addCleanup(p.stop)

    def test_remember_is_case_insensitive(self):
        self.assertEqual(login.load_account_cache(), {})
        login.remember_account('Afantii', '42')
        self.assertEqual(login.load_account_cache(), {'afantii': '42'})
        login.remember_account('AFANTII', '43')
        self.assertEqual(login.load_account_cache(), {'afantii': '43'})
