from unittest import TestCase, mock

from wizard_eyes import application, login
from wizard_eyes.application import Application


class TestRelogin(TestCase):

    def setUp(self):
        cls = type('App', (Application,), {})
        cls.__abstractmethods__ = frozenset()
        self.app = app = object.__new__(cls)
        app.client = mock.MagicMock()
        app.client.args.username = 'someone'
        app.client.get_bbox.return_value = (0, 0, 10, 10)
        app.afk_timer = mock.MagicMock()
        app.continue_ = True
        app.on_relogin = mock.MagicMock()
        self.now = 1000.0
        for target, kw in (
                ('win32gui.IsWindow', dict(return_value=True)),
                ('startup_checks.ensure_maximised', {}),
                ('startup_checks.ensure_camera', {}),
                ('time.sleep', {}),
                ('time.time', dict(side_effect=lambda: self.now))):
            mock.patch(f'wizard_eyes.application.{target}', **kw).start()
        self.state = mock.patch.object(
            login, 'classify', return_value=('in_game', None)).start()
        self.ensure = mock.patch.object(login, 'ensure_logged_in').start()
        self.addCleanup(mock.patch.stopall)

    def check(self, **kw):
        self.now += 5
        return self.app.check_login(**kw)

    def test_in_game_or_loading_does_nothing(self):
        for state in ('in_game', 'loading', 'connecting'):
            self.state.return_value = (state, None)
            self.assertFalse(self.check())
            self.assertFalse(self.check())
        self.ensure.assert_not_called()

    def test_needs_two_checks(self):
        self.state.return_value = ('login', None)
        self.assertFalse(self.check())
        self.ensure.assert_not_called()
        self.assertTrue(self.check())
        self.ensure.assert_called_once()
        self.app.on_relogin.assert_called_once()

    def test_not_checked_every_frame(self):
        self.check()
        self.now += 1
        self.app.check_login()
        self.assertEqual(self.state.call_count, 1)

    def test_stopping_checks_at_once(self):
        self.state.return_value = ('dialog', None)
        self.assertTrue(self.check(stopping=True))

    def test_stopping_in_game_stops(self):
        self.assertFalse(self.check(stopping=True))
        self.ensure.assert_not_called()

    def test_no_username(self):
        self.app.client.args.username = None
        self.state.return_value = ('login', None)
        for _ in range(3):
            self.assertFalse(self.check())
        self.ensure.assert_not_called()

    def test_limit(self):
        self.state.return_value = ('login', None)
        done = [self.check() for _ in range(8)]
        self.assertEqual(sum(done), 3)
        self.assertFalse(self.app.continue_)

    def test_failed_login_stops(self):
        self.state.return_value = ('login', None)
        self.ensure.side_effect = login.LoginError('nope')
        self.check()
        self.assertFalse(self.check())
        self.assertFalse(self.app.continue_)

    def test_moved_window_stops(self):
        self.state.return_value = ('login', None)
        self.app.client.get_bbox.side_effect = [
            (0, 0, 10, 10), (5, 5, 10, 10)]
        self.check()
        self.assertFalse(self.check())
        self.assertFalse(self.app.continue_)

    def test_window_gone(self):
        application.win32gui.IsWindow.return_value = False
        self.check()
        self.assertTrue(self.check())
