"""The engine's farm page and watchdog, for a farm with no code of its own:
bots known only by the engine's heartbeat."""
import json
import re
import tempfile
import unittest
from pathlib import Path
from urllib import error, request

from wizard_eyes.botfarm import watchdog as w
from wizard_eyes.botfarm.control import Control
from wizard_eyes.botfarm.page import FarmPage, serve
from wizard_eyes.coordination import Coordinator, Store
from wizard_eyes.coordination import serve as serve_service


def look(*lines, vm=True, running=True, log_age=2):
    return {'vm': vm, 'running': running, 'log_age': log_age,
            'lines': list(lines)}


class FakeControl:
    def __init__(self):
        self.ran = []

    def busy(self, role):
        return False

    def run(self, role, action):
        self.ran.append((role, action))


class TestGenericWatchdog(unittest.TestCase):

    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.folder = Path(d.name)
        self.now = 1_000_000.0
        self.reports = {}
        self.control = FakeControl()
        (self.folder / 'wanted-miner.local').write_text('run')

    def make(self, **kw):
        return w.Watchdog(self.control, self.folder, ['miner'],
                          lambda: self.reports, lambda r, s: look('stuck'),
                          clock=lambda: self.now, **kw)

    def run_minutes(self, dog, n, beat=None):
        for _ in range(n):
            self.now += 60
            if beat is not None:
                self.reports['miner'] = {'alive': self.now,
                                         'heartbeat': beat()}
            dog.tick()

    def test_heartbeat_alone_keeps_it_ok(self):
        dog = self.make()
        self.run_minutes(dog, 120, beat=lambda: {'frames': 1})
        self.assertEqual(dog.status()['miner']['state'], w.OK)
        self.assertEqual(self.control.ran, [])

    def test_silence_restarts(self):
        dog = self.make()
        self.run_minutes(dog, 5, beat=lambda: {})
        self.run_minutes(dog, 12)
        self.assertEqual(self.control.ran, [('miner', 'restart')])

    def test_heartbeat_progress_stuck_restarts(self):
        dog = self.make()
        count = iter(range(1000))
        self.run_minutes(dog, 30, beat=lambda: {'progress': next(count)})
        self.run_minutes(dog, 60, beat=lambda: {'progress': 'same'})
        self.assertEqual(self.control.ran, [('miner', 'restart')])

    def test_no_cure(self):
        self.assertEqual(w.judge(look('Out of ore; stopping', running=False),
                                 600, [(re.compile('Out of ore'), 'ore')]),
                         (w.ALERT, 'ore'))
        self.assertEqual(w.judge(look('The Bank PIN was wrong; stopping',
                                      running=False), 600)[0], w.ALERT)


class Job:
    def __init__(self, *a, **k):
        pass

    def poll(self):
        return None


class TestPage(unittest.TestCase):

    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.folder = Path(d.name)
        service = serve_service(Store(), port=0)
        self.addCleanup(service.server_close)
        self.addCleanup(service.shutdown)
        self.service = f'http://127.0.0.1:{service.server_address[1]}'
        Coordinator(self.service, 'miner').heartbeat(
            {'script': 'miner.py', 'line': 'mining', 'running_for': 60})
        self.control = Control('x.ps1', self.folder, ['miner', 'smith'],
                               popen=Job)
        page = FarmPage('Test farm', ['miner', 'smith'], self.service,
                        self.control, folder=self.folder,
                        get={'/extra': lambda: (b'{"x": 1}',
                                                'application/json')},
                        post={'/echo': lambda body: body})
        server = serve(page, 0)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.url = f'http://127.0.0.1:{server.server_address[1]}'

    def post(self, path, body, **headers):
        return request.urlopen(request.Request(
            self.url + path, json.dumps(body).encode(),
            {'Content-Type': 'application/json', **headers}))

    def test_page_and_control(self):
        html = request.urlopen(self.url + '/').read().decode()
        self.assertIn('<title>Test farm</title>', html)
        got = json.load(request.urlopen(self.url + '/control'))
        self.assertEqual(got['farm'], 'Test farm')
        self.assertTrue(got['bots']['miner']['working'])
        self.assertEqual(got['bots']['miner']['heartbeat']['line'], 'mining')
        self.assertFalse(got['bots']['smith']['working'])
        self.assertIsNone(got['bots']['smith']['heartbeat'])
        self.post('/control', {'role': 'smith', 'action': 'start'})
        self.assertTrue(self.control.busy('smith'))
        with self.assertRaises(error.HTTPError) as e:
            self.post('/control', {'role': 'nobody', 'action': 'start'})
        self.assertEqual(e.exception.code, 400)
        with self.assertRaises(error.HTTPError) as e:
            self.post('/control', {'role': 'miner', 'action': 'start'},
                      Origin='http://evil.example')
        self.assertEqual(e.exception.code, 403)

    def test_farm_routes(self):
        self.assertEqual(json.load(request.urlopen(self.url + '/extra')),
                         {'x': 1})
        self.assertEqual(json.load(self.post('/echo', {'a': 2})), {'a': 2})
        for path in ('/nothing', '/live/miner'):
            with self.assertRaises(error.HTTPError) as e:
                request.urlopen(self.url + path)
            self.assertEqual(e.exception.code, 404)


if __name__ == '__main__':
    unittest.main()
