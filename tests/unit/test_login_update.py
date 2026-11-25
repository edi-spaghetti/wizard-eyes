from unittest import TestCase, mock

from wizard_eyes import login
from wizard_eyes.window import ClientWindow


class Base(TestCase):

    def setUp(self):
        self.now = 1000.0
        self.window = mock.Mock(spec=ClientWindow, pid=1, handle=2,
                                title='RuneLite - someone')

        def sleep(seconds):
            self.now += seconds

        for name, kw in (
                ('time.time', dict(side_effect=lambda: self.now)),
                ('time.sleep', dict(side_effect=sleep)),
                ('current_window', dict(return_value=self.window)),
                ('find_runelite_for', dict(return_value=self.window)),
                ('focus_window', {}),
                ('grab_window', dict(return_value=(None, (0, 0))))):
            mock.patch(f'wizard_eyes.login.{name}', **kw).start()
        self.click = mock.patch.object(login, 'click_box').start()
        self.launch = mock.patch.object(
            login, 'launch_from_launcher', return_value=self.window).start()
        self.close = mock.patch.object(login, 'close_client').start()
        self.addCleanup(mock.patch.stopall)


class TestServersUpdating(Base):

    def run_states(self, state_at, updating=True):
        mock.patch.object(login, 'classify', side_effect=lambda img: (
            state_at(self.now - 1000), mock.Mock(
                **{'screen_box.return_value': (0, 0, 1, 1)}))).start()
        mock.patch.object(login, 'find', side_effect=lambda img, name: (
            True if name == 'login_being_updated' and updating else None)
        ).start()
        return login.log_in(self.window, timeout=300)

    def test_waits_before_trying_again(self):
        def state(t):
            return 'in_game' if self.click.call_count else 'dialog'
        self.run_states(state)
        self.assertGreaterEqual(self.now - 1000, login.UPDATE_RETRY[0])

    def test_outlasts_the_timeout(self):
        def state(t):
            if t > 1500:
                return 'in_game'
            return 'login' if self.click.call_count % 2 else 'dialog'
        self.run_states(state)
        self.assertGreater(self.now - 1000, 1500)

    def test_other_dialogs_keep_the_timeout(self):
        with self.assertRaises(login.ClientStuck):
            self.run_states(lambda t: 'dialog', updating=False)
        self.assertLess(self.now - 1000, 310)

    def test_gives_up_in_the_end(self):
        with self.assertRaises(login.ClientStuck):
            self.run_states(lambda t: 'login' if int(t) % 30 < 10
                            else 'dialog')
        self.assertLess(self.now - 1000, login.UPDATE_PATIENCE + 600)

    def test_client_updated(self):
        with self.assertRaises(login.ClientOutdated):
            self.run_states(lambda t: 'client_updated')
        self.click.assert_not_called()


class TestRestarts(Base):

    def test_outdated_client_restarted_until_it_works(self):
        out = login.ClientOutdated('updated')
        with mock.patch.object(login, 'log_in', side_effect=[
                out, login.LoginError('RuneLite closed'), out, self.window]):
            login.ensure_logged_in('someone')
        self.assertEqual(self.close.call_count, 3)
        self.assertEqual(self.launch.call_count, 3)
        # at once the first time, then with pauses
        self.assertGreaterEqual(self.now - 1000, 2 * login.UPDATE_RETRY[0])

    def test_outdated_gives_up_in_the_end(self):
        with mock.patch.object(login, 'log_in',
                               side_effect=login.ClientOutdated('updated')):
            with self.assertRaises(login.ClientOutdated):
                login.ensure_logged_in('someone')

    def test_stuck_client_restarted_once(self):
        with mock.patch.object(login, 'log_in',
                               side_effect=login.ClientStuck('stuck')):
            with self.assertRaises(login.ClientStuck):
                login.ensure_logged_in('someone')
        self.assertEqual(self.launch.call_count, 1)

    def test_other_errors_are_not_retried(self):
        with mock.patch.object(login, 'log_in',
                               side_effect=login.LoginError('no')):
            with self.assertRaises(login.LoginError):
                login.ensure_logged_in('someone')
        self.launch.assert_not_called()
