import tempfile
import unittest
from pathlib import Path

from wizard_eyes.coordination import (Coordinator, CoordinationError,
                                      CoordinatorUnavailable, Store, serve)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class TestStore(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.store = Store(clock=self.clock)

    def meeting(self, **kw):
        return self.store.propose('a', 'b', {'items': {'x': 1}}, 600, **kw)

    def test_full_meeting(self):
        m = self.meeting()
        self.assertEqual(m['state'], 'proposed')
        self.assertEqual(self.store.act(m['id'], 'b', 'accept')['state'],
                         'accepted')
        self.assertEqual(self.store.act(m['id'], 'a', 'arrive')['state'],
                         'accepted')
        self.assertEqual(self.store.act(m['id'], 'b', 'arrive')['state'], 'met')
        self.assertEqual(self.store.act(m['id'], 'a', 'done', {'ok': 1})
                         ['state'], 'met')
        done = self.store.act(m['id'], 'b', 'done', {'ok': 1})
        self.assertEqual(done['state'], 'done')
        self.assertEqual(done['results'], {'a': {'ok': 1}, 'b': {'ok': 1}})

    def test_only_invitee_accepts(self):
        m = self.meeting()
        with self.assertRaises(CoordinationError):
            self.store.act(m['id'], 'a', 'accept')
        with self.assertRaises(CoordinationError):
            self.store.act(m['id'], 'c', 'accept')

    def test_no_arrive_before_accept_and_no_done_before_met(self):
        m = self.meeting()
        with self.assertRaises(CoordinationError):
            self.store.act(m['id'], 'a', 'arrive')
        self.store.act(m['id'], 'b', 'accept')
        with self.assertRaises(CoordinationError):
            self.store.act(m['id'], 'a', 'done')

    def test_decline_and_abort(self):
        m = self.meeting()
        self.assertEqual(self.store.act(m['id'], 'b', 'decline')['state'],
                         'declined')
        with self.assertRaises(CoordinationError):
            self.store.act(m['id'], 'a', 'abort', 'late')
        m = self.meeting()
        self.store.act(m['id'], 'b', 'accept')
        out = self.store.act(m['id'], 'a', 'abort', 'no stock')
        self.assertEqual(out['state'], 'aborted')
        self.assertEqual(out['reasons'], {'a': 'no stock'})

    def test_expiry(self):
        m = self.meeting()
        self.clock.t += 601
        self.assertEqual(self.store.meetings('a', open_only=True), [])
        (gone,) = self.store.meetings('a')
        self.assertEqual(gone['state'], 'aborted')
        self.assertEqual(gone['reasons'], {'_server': 'expired'})
        with self.assertRaises(CoordinationError):
            self.store.act(m['id'], 'b', 'accept')

    def test_cannot_meet_self(self):
        with self.assertRaises(CoordinationError):
            self.store.propose('a', 'a', {}, 60)

    def test_persists(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / 'state.local.json')
            s = Store(path, clock=self.clock)
            s.set_status('a', {'coins': 5})
            s.set_shared('prices', {'x': 1}, 'a')
            m = s.propose('a', 'b', {}, 60)
            s2 = Store(path, clock=self.clock)
            self.assertEqual(s2.bots()['a']['status'], {'coins': 5})
            self.assertEqual(s2.shared()['prices']['value'], {'x': 1})
            self.assertEqual(s2.meetings('b')[0]['id'], m['id'])


class TestHttp(unittest.TestCase):
    def setUp(self):
        self.server = serve(Store(), port=0, token='t')
        url = f'http://127.0.0.1:{self.server.server_address[1]}'
        self.a = Coordinator(url, 'a', token='t')
        self.b = Coordinator(url, 'b', token='t')
        self.url = url

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_round_trip(self):
        self.a.post_status({'clean': {'Irit leaf': 100}})
        self.assertEqual(self.b.bots()['a']['status'],
                         {'clean': {'Irit leaf': 100}})
        self.a.set_shared('prices', {'Grimy irit leaf': 930})
        self.assertEqual(self.b.shared()['prices']['by'], 'a')

        m = self.a.propose('b', {'place': 'ge'}, expires_in=60)
        (seen,) = self.b.meetings()
        self.assertEqual(seen['id'], m['id'])
        self.b.accept(m['id'])
        self.a.arrive(m['id'])
        self.assertEqual(self.b.arrive(m['id'])['state'], 'met')
        self.a.done(m['id'], {'traded': True})
        self.assertEqual(self.b.done(m['id'], {'traded': True})['state'],
                         'done')
        self.assertEqual(self.a.meetings(), [])          # open only
        self.assertEqual(len(self.a.meetings(open_only=False)), 1)

    def test_refusals_are_unavailable(self):
        m = self.a.propose('b', {})
        with self.assertRaises(CoordinatorUnavailable) as e:
            self.a.accept(m['id'])
        self.assertIn('409', str(e.exception))
        with self.assertRaises(CoordinatorUnavailable):
            Coordinator(self.url, 'a', token='wrong').bots()

    def test_unreachable(self):
        c = Coordinator('http://127.0.0.1:9', 'a', timeout=0.5)
        with self.assertRaises(CoordinatorUnavailable):
            c.bots()


if __name__ == '__main__':
    unittest.main()
