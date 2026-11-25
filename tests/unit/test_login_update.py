from unittest import TestCase, mock

from wizard_eyes import game_status, login
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
        self.status = mock.patch.object(
            game_status, 'fetch', return_value=None).start()
        game_status._last = (-float('inf'), None)
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
        self.assertGreaterEqual(self.now - 1000, game_status.RETRY[0])

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

    def forever(self, t):
        return 'login' if self.click.call_count % 2 else 'dialog'

    def test_gives_up_in_the_end(self):
        with self.assertRaises(login.ClientStuck):
            self.run_states(self.forever)
        self.assertLess(self.now - 1000, game_status.PATIENCE + 900)

    def test_maintenance_is_waited_out(self):
        self.status.side_effect = lambda: \
            'offline' if self.now < 1000 + 3 * 3600 else 'online'
        self.run_states(lambda t: 'in_game' if t > 3 * 3600 + 200
                        else self.forever(t))
        self.assertLess(self.now - 1000, 3 * 3600 + 800)
        # a try every five minutes or so, two clicks each
        self.assertLess(self.click.call_count, 100)

    def test_offline_forever_gives_up(self):
        self.status.return_value = 'offline'
        with self.assertRaises(login.ClientStuck):
            self.run_states(self.forever)
        self.assertLess(self.now - 1000, game_status.OFFLINE_PATIENCE + 2000)

    def test_frozen_login_screen(self):
        with self.assertRaises(login.ClientStuck):
            self.run_states(lambda t: 'login')
        self.assertEqual(self.click.call_count, login.MAX_SAME_CLICKS)
        self.assertLess(self.now - 1000, 60)

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
        self.assertGreaterEqual(self.now - 1000, 2 * game_status.RETRY[0])

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


class TestPage(TestCase):
    PAGE = """<div class="osrs-title" id="Status">Current Status:
       <!--<span style="color: #F75F5F;">OFFLINE</span>-->
        <span style="color: #04B648;">ONLINE</span>
    </div><p>Game goes OFFLINE at 11:30</p>"""

    def test_online(self):
        self.assertEqual(game_status.parse(self.PAGE), 'online')

    def test_offline(self):
        page = self.PAGE.replace('<!--', '').replace('-->', '').replace(
            '<span style="color: #04B648;">ONLINE</span>',
            '<!--<span>ONLINE</span>-->')
        self.assertEqual(game_status.parse(page), 'offline')

    def test_unclear(self):
        both = self.PAGE.replace('<!--', '').replace('-->', '')
        for page in ('', 'blocked', both):
            self.assertIsNone(game_status.parse(page))

    def test_unreachable(self):
        with mock.patch.object(game_status, 'urlopen', side_effect=OSError):
            self.assertIsNone(game_status.fetch())

    def test_not_read_all_the_time(self):
        game_status._last = (-float('inf'), None)
        with mock.patch.object(game_status, 'fetch',
                               return_value='online') as f:
            for _ in range(5):
                self.assertEqual(game_status.current(), 'online')
        f.assert_called_once()
