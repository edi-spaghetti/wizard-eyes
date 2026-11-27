"""Rests in a bot's day (wizard_eyes.rest) and how the application and the
farm's watchdog treat them, plus a banned account."""
import random
import re
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

from wizard_eyes import login, rest
from wizard_eyes.application import Application
from wizard_eyes.botfarm import watchdog as w


def at(text: str) -> float:
    return datetime.strptime(text, '%Y-%m-%d %H:%M').timestamp()


class TestParse(unittest.TestCase):

    def test_sleep(self):
        self.assertEqual(rest.parse_sleep('23:00-07:30'), (1380, 450))
        self.assertEqual(rest.parse_sleep(' 1:05 - 9:00 '), (65, 540))
        for bad in ('23-07', '25:00-07:00', '07:00-07:00', ''):
            with self.assertRaises(ValueError):
                rest.parse_sleep(bad)

    def test_breaks(self):
        self.assertEqual(rest.parse_breaks('50-130:3-15'),
                         ((50, 130), (3, 15)))
        self.assertEqual(rest.parse_breaks('60:5'), ((60, 60), (5, 5)))
        for bad in ('50-130', '130-50:3', '0:5', 'x:y'):
            with self.assertRaises(ValueError):
                rest.parse_breaks(bad)

    def test_from_args(self):
        self.assertIsNone(rest.from_args(['--username', 'x']))
        s = rest.from_args(['--username', 'x', '--sleep', '23:00-07:00',
                            '--breaks', '50-130:3-15', '--rest-jitter', '30'])
        self.assertEqual((s.sleep, s.breaks, s.jitter),
                         ((1380, 420), ((50, 130), (3, 15)), 30))
        self.assertIn('sleep 23:00-07:00', str(s))


class TestSchedule(unittest.TestCase):

    def sched(self, sleep='23:00-07:00', breaks=None, jitter=45, seed=1):
        return rest.RestSchedule(
            rest.parse_sleep(sleep) if sleep else None,
            rest.parse_breaks(breaks) if breaks else None,
            jitter, random.Random(seed))

    def test_next_sleep_is_tonight(self):
        for seed in range(50):
            r = self.sched(seed=seed).next_sleep(at('2026-11-27 12:00'))
            self.assertEqual(r.kind, 'sleep')
            self.assertLessEqual(abs(r.start - at('2026-11-27 23:00')),
                                 45 * 60)
            self.assertLessEqual(abs(r.end - at('2026-11-28 07:00')),
                                 45 * 60)

    def test_inside_the_night_sleeps_now(self):
        now = at('2026-11-28 03:00')
        r = self.sched().next_sleep(now)
        self.assertEqual(r.start, now)
        self.assertLessEqual(abs(r.end - at('2026-11-28 07:00')), 45 * 60)

    def test_daytime_sleep(self):
        r = self.sched('13:00-14:00', jitter=0).next_sleep(
            at('2026-11-27 13:30'))
        self.assertEqual((r.start, r.end), (at('2026-11-27 13:30'),
                                            at('2026-11-27 14:00')))

    def test_breaks_come_before_the_sleep(self):
        s = self.sched(breaks='50-130:3-15')
        now = at('2026-11-27 09:00')
        r = s.plan(now)
        self.assertEqual(r.kind, 'break')
        self.assertTrue(50 * 60 <= r.start - now <= 130 * 60)
        self.assertTrue(3 <= r.minutes <= 15)
        # late in the evening there's no room for one before the sleep
        r = s.plan(at('2026-11-27 21:30'))
        self.assertEqual(r.kind, 'sleep')

    def test_breaks_only(self):
        s = self.sched(sleep=None, breaks='60:5')
        now = at('2026-11-27 09:00')
        self.assertIsNone(s.due(now))
        r = s.due(now + 3600)
        self.assertEqual((r.kind, r.minutes), ('break', 5))
        s.taken(now + 3900)
        self.assertEqual(s.next.start, now + 3900 + 3600)

    def test_missed_rest_is_replanned(self):
        s = self.sched(sleep=None, breaks='60:5')
        now = at('2026-11-27 09:00')
        s.due(now)
        r = s.due(now + 5 * 3600)      # the PC slept through it
        self.assertIsNone(r)
        self.assertGreater(s.next.start, now + 5 * 3600)


class TestApplicationRests(unittest.TestCase):

    def setUp(self):
        cls = type('App', (Application,), {})
        cls.__abstractmethods__ = frozenset()
        self.app = app = object.__new__(cls)
        app.client = mock.MagicMock()
        app.client.args.username = 'someone'
        app.client.continue_ = True
        app.client.get_bbox.return_value = (0, 0, 10, 10)
        app.afk_timer = mock.MagicMock()
        app.continue_ = True
        app.msg_length = 100
        app.on_relogin = mock.MagicMock()
        self.now = at('2026-11-27 09:00')
        app.start_time = self.now
        self.state = mock.patch.object(
            login, 'classify', return_value=('login', None)).start()
        self.ensure = mock.patch.object(login, 'ensure_logged_in').start()

        def sleep(s):
            self.now += s
        for target, kw in (
                ('win32gui.IsWindow', dict(return_value=True)),
                ('startup_checks.ensure_maximised', {}),
                ('startup_checks.ensure_camera', {}),
                ('time.sleep', dict(side_effect=sleep)),
                ('time.time', dict(side_effect=lambda: self.now))):
            mock.patch(f'wizard_eyes.application.{target}', **kw).start()
        self.addCleanup(mock.patch.stopall)

    def test_no_options_no_rests(self):
        self.app.start_rests(['--username', 'x'])
        self.assertFalse(self.app.take_rest())

    def test_break_then_log_back_in(self):
        self.app.start_rests(['--breaks', '60:10'])
        self.assertFalse(self.app.take_rest())
        self.now += 3600
        self.app.can_rest = lambda: False
        self.assertFalse(self.app.take_rest())       # not a good moment
        self.app.can_rest = lambda: True
        self.assertTrue(self.app.take_rest())
        self.assertGreaterEqual(self.now, at('2026-11-27 10:10'))
        self.assertIsNone(self.app.resting)
        # the game logged us out: back in, not counted as a relogin
        self.ensure.assert_called_once()
        self.assertEqual(self.app._relogins, ())
        self.app.on_relogin.assert_called_once()

    def test_short_break_still_logged_in(self):
        self.state.return_value = ('in_game', None)
        self.app.start_rests(['--breaks', '60:2'])
        self.now += 3600
        self.assertTrue(self.app.take_rest())
        self.ensure.assert_not_called()

    def test_banned_after_rest(self):
        self.state.return_value = ('banned', None)
        self.app.farm_heartbeat = mock.MagicMock()
        self.app.start_rests(['--breaks', '60:2'])
        self.now += 3600
        self.app.take_rest()
        self.assertTrue(self.app.banned)
        self.assertFalse(self.app.continue_)
        self.app.farm_heartbeat.send.assert_called()

    def test_check_login_stops_on_ban(self):
        self.state.return_value = ('banned', None)
        self.app._next_login_check = 0
        self.assertFalse(self.app.check_login())
        self.assertTrue(self.app.banned)
        self.assertFalse(self.app.continue_)
        self.ensure.assert_not_called()


class TestWatchdog(unittest.TestCase):

    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.folder = Path(d.name)
        (self.folder / 'wanted-bot.local').write_text('run')
        self.now = 1_000_000.0
        self.reports = {}
        self.control = mock.MagicMock()
        self.control.busy.return_value = False
        self.said = []
        watch = mock.MagicMock()
        watch.bot.side_effect = lambda *a: self.said.append(a)
        self.dog = w.Watchdog(self.control, self.folder, ['bot'],
                              lambda: self.reports,
                              lambda r, s: {'vm': True, 'running': True,
                                            'log_age': 5, 'lines': ['x']},
                              clock=lambda: self.now, watch=watch)

    def minutes(self, n, beat):
        for _ in range(n):
            self.now += 60
            self.reports['bot'] = {'alive': self.now, 'heartbeat': beat()}
            self.dog.tick()

    def test_resting_all_night_is_fine(self):
        self.minutes(5, lambda: {'progress': self.now})
        until = self.now + 9 * 3600
        self.minutes(8 * 60, lambda: {'progress': 1, 'resting': {
            'kind': 'sleep', 'until': until}})
        self.assertEqual(self.dog.status()['bot']['state'], w.OK)
        self.assertIn('resting (sleep)', self.dog.status()['bot']['text'])
        self.control.run.assert_not_called()
        self.minutes(20, lambda: {'progress': self.now})
        self.assertEqual(self.dog.status()['bot']['text'], 'working')
        self.control.run.assert_not_called()

    def test_banned_alerts_at_once(self):
        self.minutes(1, lambda: {'banned': True})
        self.assertEqual(self.dog.status()['bot']['state'], w.ALERT)
        self.assertEqual(self.said, [('bot', w.ALERT, w.BANNED)])
        self.minutes(30, lambda: {'banned': True})
        self.control.run.assert_not_called()
        self.assertEqual(len(self.said), 1)        # one message

    def test_banned_in_the_log(self):
        lines = ['Traceback (most recent call last):',
                 'wizard_eyes.login.AccountBanned: the account is banned '
                 '(login screen: ...)']
        self.assertEqual(w.judge({'vm': True, 'running': False,
                                  'log_age': 60, 'lines': lines}, 600),
                         (w.ALERT, w.BANNED))
        self.assertEqual(w.judge({'vm': True, 'running': True, 'log_age': 60,
                                  'lines': ['Time 1h - Resting (break) 3 min '
                                            'to go']}, 3000)[0], w.WAIT)


if __name__ == '__main__':
    unittest.main()
