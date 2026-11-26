import unittest

from wizard_eyes import heartbeat
from wizard_eyes.coordination import (Coordinator, CoordinatorUnavailable,
                                      Store, last_seen, serve)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class TestStoreHeartbeat(unittest.TestCase):

    def test_status_and_heartbeat_keep_each_other(self):
        clock = Clock()
        store = Store(clock=clock)
        store.set_status('mixer', {'stock': {'a': 1}})
        clock.t = 1060
        store.beat('mixer', {'frames': 5})
        clock.t = 1100
        store.set_status('mixer', {'stock': {'a': 2}})
        b = store.bots()['mixer']
        self.assertEqual(b['status'], {'stock': {'a': 2}})
        self.assertEqual(b['heartbeat'], {'frames': 5})
        self.assertEqual((b['updated'], b['alive']), (1100, 1060))
        self.assertEqual(last_seen(b), 1100)
        # a bot that only beats
        store.beat('other', {})
        self.assertNotIn('status', store.bots()['other'])
        self.assertEqual(last_seen(store.bots()['other']), 1100)
        self.assertEqual(last_seen(None), 0.0)


class FakeApp:
    start_time = 0.0
    frame_number = 12
    sleeping = False
    msg_buffer = ['one', 'Time 0h 1m 2s 0.2 - AFK: 0.3']
    progress = 4


class TestHeartbeat(unittest.TestCase):

    def test_env(self):
        self.assertIsNone(heartbeat.from_env({}))
        self.assertIsNone(heartbeat.from_env(
            {heartbeat.ENV_ROLE: 'mixer'}))
        self.assertEqual(heartbeat.from_env(
            {heartbeat.ENV_ROLE: 'mixer', heartbeat.ENV_URL: 'http://x'}),
            {'role': 'mixer', 'url': 'http://x'})
        self.assertIsNone(heartbeat.start_for(FakeApp(), {}))

    def test_snapshot(self):
        s = heartbeat.snapshot(FakeApp())
        self.assertEqual(s['frames'], 12)
        self.assertEqual(s['line'], 'Time 0h 1m 2s 0.2 - AFK: 0.3')
        self.assertEqual(s['progress'], 4)
        self.assertFalse(s['sleeping'])
        app = FakeApp()
        app.start_time = float('inf')     # not running yet
        app.msg_buffer = []
        app.progress = None
        s = heartbeat.snapshot(app)
        self.assertEqual((s['running_for'], s['line']), (0.0, ''))
        self.assertNotIn('progress', s)

    def test_sends_and_survives_no_service(self):
        server = serve(Store(), port=0)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        url = f'http://127.0.0.1:{server.server_address[1]}'
        h = heartbeat.Heartbeat(url, 'mixer', lambda: {'frames': 1})
        self.assertTrue(h.send())
        self.assertEqual(Coordinator(url, 'x').bots()['mixer']['heartbeat'],
                         {'frames': 1})
        gone = heartbeat.Heartbeat('http://127.0.0.1:9', 'mixer',
                                   lambda: {}, client=Coordinator(
                                       'http://127.0.0.1:9', 'mixer',
                                       timeout=0.3))
        self.assertFalse(gone.send())
        self.assertTrue(gone.failing)

        def broken():
            raise RuntimeError('bad snapshot')
        self.assertFalse(heartbeat.Heartbeat(url, 'm', broken).send())

    def test_thread_stops(self):
        sent = []

        class Client:
            def heartbeat(self, beat):
                sent.append(beat)
        h = heartbeat.Heartbeat('', 'mixer', lambda: {}, interval=0.01,
                                client=Client()).start()
        h.stop()
        h.thread.join(1)
        self.assertFalse(h.thread.is_alive())
        self.assertGreaterEqual(len(sent), 1)


if __name__ == '__main__':
    unittest.main()
