import tempfile
import unittest
from pathlib import Path

from wizard_eyes.market import PriceHistory, volume


class Wiki:
    def __init__(self):
        self.calls, self.now, self.down = [], 0, False

    def item(self, name):
        if 'mapping' not in self.calls:
            self.calls.append('mapping')
        return {'id': {'A': 1, 'B': 2}[name], 'limit': 10}

    def get(self, endpoint, **params):
        self.calls.append(endpoint)
        if self.down:
            raise OSError('down')
        if endpoint == 'timeseries':
            return {'data': [{
                'timestamp': self.now - 300 * i, 'avgHighPrice': 5,
                'avgLowPrice': 4, 'highPriceVolume': 10,
                'lowPriceVolume': 10} for i in range(288, 0, -1)]}
        return {'timestamp': self.now - 300 + self.now % 300, 'data': {
            '1': {'avgHighPrice': 6, 'avgLowPrice': None,
                  'highPriceVolume': 10, 'lowPriceVolume': 0}}}


class TestPriceHistory(unittest.TestCase):

    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.path = str(Path(d.name) / 'p.json')
        self.wiki = Wiki()
        self.wiki.now = 1_800_000_000
        self.h = self.history()

    def history(self):
        return PriceHistory(['A', 'B'], self.path, 900, self.wiki,
                            lambda: self.wiki.now)

    def test_one_request_per_interval(self):
        self.h.refresh()
        self.assertEqual(self.wiki.calls,
                         ['mapping', 'timeseries', 'timeseries', '5m'])
        self.wiki.calls.clear()
        for _ in range(12):         # an hour, looked at every 5 minutes
            self.wiki.now += 300
            self.h.refresh()
        self.assertEqual(self.wiki.calls, ['5m'] * 4)
        # a restart doesn't ask again
        self.wiki.calls.clear()
        h = self.history()
        h.refresh()
        self.assertEqual(self.wiki.calls, [])
        self.assertEqual(h.limit('A'), 10)
        self.assertEqual(len(h.series('A')), len(self.h.series('A')))

    def test_volumes_stay_comparable(self):
        self.h.refresh()
        day = volume(self.h.series('A'))
        for _ in range(8):
            self.wiki.now += 900
            self.h.refresh()
        new = self.h.series('A')[-1]
        self.assertEqual((new['avgHighPrice'], new['highPriceVolume']),
                         (6, 30))
        self.assertGreater(volume(self.h.series('A')), day * 0.9)
        # an item nobody traded still gets its time stamp
        self.assertEqual(self.h.series('B')[-1]['timestamp'],
                         new['timestamp'])
        self.assertIsNone(self.h.series('B')[-1]['avgHighPrice'])

    def test_failure_waits_too(self):
        self.h.refresh()
        arrived = self.h.fetched
        self.wiki.down, self.wiki.now = True, self.wiki.now + 900
        with self.assertRaises(OSError):
            self.h.refresh()
        self.wiki.calls.clear()
        self.wiki.now += 300
        self.h.refresh()
        self.assertEqual(self.wiki.calls, [])
        self.assertEqual(self.h.fetched, arrived)

    def test_seeds_again_after_a_long_break(self):
        self.h.refresh()
        self.wiki.calls.clear()
        self.wiki.now += 20 * 3600
        self.h.refresh()
        self.assertEqual(self.wiki.calls.count('timeseries'), 2)


if __name__ == '__main__':
    unittest.main()
