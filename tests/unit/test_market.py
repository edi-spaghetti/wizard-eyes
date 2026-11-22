import random
import unittest

from wizard_eyes.market import (BASE_OFFSET, MIN_MARGIN, Conversion,
                                Evaluation, Fill, OfferBook, evaluate,
                                margin, next_trip_seconds, plan_quantities,
                                reference, select, step_size,
                                supplies_needed)


def series(prices, key, vol=500):
    side = 'lowPriceVolume' if key == 'avgLowPrice' else 'highPriceVolume'
    return [{'timestamp': i * 300, key: p, side: vol}
            for i, p in enumerate(prices)]


def ev(buy=1000, sell=1200, ok=True, source='Grimy x', product='X'):
    e = Evaluation(Conversion(source, product), buy, sell, 150, 1.0, 0.2,
                   10 ** 6, 10 ** 6)
    if not ok:
        e.reasons.append('bad')
    return e


class TestPrices(unittest.TestCase):

    def test_margin(self):
        self.assertEqual(margin(100, 200), 200 - 4 - 100)

    def test_reference_median_of_last_hour(self):
        s = series([1] * 20 + [100] * 11 + [10000], 'avgLowPrice')
        s.insert(25, {'timestamp': 1, 'avgLowPrice': None})
        self.assertEqual(reference(s, 'avgLowPrice'), 100)
        self.assertIsNone(reference([], 'avgLowPrice'))

    def test_evaluate_good_and_bad(self):
        good = evaluate(Conversion('Grimy a', 'A'),
                        series([1000 + i % 3 for i in range(288)],
                               'avgLowPrice'),
                        series([1250 + i % 5 for i in range(288)],
                               'avgHighPrice'))
        self.assertTrue(good.ok, good.reasons)
        # ranarr-like: big price, thin margin share
        thin = evaluate(Conversion('Grimy r', 'R'),
                        series([4650] * 288, 'avgLowPrice'),
                        series([4900] * 288, 'avgHighPrice'))
        self.assertFalse(thin.ok)
        self.assertTrue(any('of price' in r for r in thin.reasons))
        # huasca-like: no volume
        quiet = evaluate(Conversion('Grimy h', 'H'),
                         series([300] * 288, 'avgLowPrice', vol=1),
                         series([600] * 288, 'avgHighPrice', vol=1))
        self.assertTrue(any('volume' in r for r in quiet.reasons))

    def test_select_ranks_by_limit_earnings(self):
        a, b, c = ev(source='a'), ev(source='b'), ev(source='c', ok=False)
        b.median_margin = 300
        out = select([a, b, c], {'a': 13000, 'b': 1000, 'c': 99999})
        self.assertEqual([e.conversion.source for e in out], ['a', 'b'])


class TestOfferBook(unittest.TestCase):

    def test_base_prices(self):
        buy, sell = OfferBook().prices(ev())
        self.assertEqual(buy.price, 1000 + BASE_OFFSET)
        self.assertEqual(sell.price, 1200 - BASE_OFFSET)

    def test_slow_buy_steps_up_complete_backs_off(self):
        book = OfferBook()
        slow = Fill('Grimy x', 'buy', 1005, 100, 10, hours=5)
        self.assertEqual(book.learn(slow, 1000), step_size(1000))
        self.assertEqual(book.prices(ev())[0].price,
                         1000 + BASE_OFFSET + step_size(1000))
        done = Fill('Grimy x', 'buy', 1015, 100, 100, hours=1)
        self.assertEqual(book.learn(done, 1000), 10 - 5)

    def test_young_partial_offer_unchanged(self):
        book = OfferBook()
        self.assertEqual(book.learn(Fill('X', 'sell', 1, 100, 10, 0.5), 100),
                         0)

    def test_limits(self):
        book = OfferBook()
        for _ in range(50):
            book.learn(Fill('X', 'sell', 1, 10, 0, 9), 1000)
        self.assertEqual(book.get('sell', 'X'), 6 * step_size(1000))
        for _ in range(50):
            book.learn(Fill('X', 'sell', 1, 10, 10, 9), 1000)
        self.assertEqual(book.get('sell', 'X'), -BASE_OFFSET)

    def test_margin_floor(self):
        book = OfferBook(extra={'buy:Grimy x': 500})
        buy, _ = book.prices(ev())
        ask = 1200 - BASE_OFFSET
        self.assertLessEqual(margin(buy.price, ask), MIN_MARGIN + 1)
        self.assertIn('floor', buy.why)
        # market failed: not even the base bid keeps the floor
        buy, sell = OfferBook().prices(ev(buy=1150, sell=1200))
        self.assertIsNone(buy.price)
        self.assertIsNotNone(sell.price)

    def test_save_load(self):
        import os
        import tempfile
        path = os.path.join(tempfile.mkdtemp(), 'm.local.json')
        OfferBook(extra={'buy:a': 7}, path=path).save()
        self.assertEqual(OfferBook.load(path).get('buy', 'a'), 7)


class TestTrips(unittest.TestCase):

    def test_trip_gap(self):
        rng = random.Random(1)
        gaps = [next_trip_seconds(rng) / 3600 for _ in range(500)]
        self.assertTrue(all(4 <= g <= 12 for g in gaps))
        self.assertLess(min(gaps), 4.5)
        self.assertGreater(max(gaps), 11.5)

    def test_quantities(self):
        ranked = [ev(source='a'), ev(source='b'), ev(source='c')]
        limits = {'a': 13000, 'b': 11000, 'c': 13000}
        self.assertEqual(plan_quantities(ranked, limits, 20000, 8),
                         {'a': 13000, 'b': 7000})
        self.assertEqual(plan_quantities(ranked, limits, 99999, 2),
                         {'a': 13000, 'b': 11000})
        self.assertEqual(supplies_needed(16000, 6, 10000), 86000)


if __name__ == '__main__':
    unittest.main()
