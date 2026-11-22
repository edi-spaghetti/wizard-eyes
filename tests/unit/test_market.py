import os
import tempfile
import unittest

from wizard_eyes.market import (Conversion, Evaluation, Fill, OfferBook,
                                Rules, evaluate, margin, plan_quantities,
                                reference, select)

RULES = Rules(offset=5, min_margin=50, min_margin_share=0.05,
              min_positive=0.95, max_noise=0.5, min_volume=50_000)


def series(prices, key, vol=500):
    side = 'lowPriceVolume' if key == 'avgLowPrice' else 'highPriceVolume'
    return [{'timestamp': i * 300, key: p, side: vol}
            for i, p in enumerate(prices)]


def ev(buy=1000, sell=1200, ok=True, source='Src', product='Prod'):
    e = Evaluation(Conversion(source, product), buy, sell, 150, 1.0, 0.2,
                   10 ** 6, 10 ** 6)
    if not ok:
        e.reasons.append('bad')
    return e


class TestPrices(unittest.TestCase):

    def test_margin(self):
        self.assertEqual(margin(100, 200), 200 - 4 - 100)
        self.assertEqual(margin(100, 200, ratio=0.5), 98 - 100)

    def test_reference_median_of_last_hour(self):
        s = series([1] * 20 + [100] * 11 + [10000], 'avgLowPrice')
        s.insert(25, {'timestamp': 1, 'avgLowPrice': None})
        self.assertEqual(reference(s, 'avgLowPrice'), 100)
        self.assertIsNone(reference([], 'avgLowPrice'))

    def test_evaluate(self):
        good = evaluate(Conversion('a', 'A'),
                        series([1000 + i % 3 for i in range(288)],
                               'avgLowPrice'),
                        series([1250 + i % 5 for i in range(288)],
                               'avgHighPrice'), RULES)
        self.assertTrue(good.ok, good.reasons)
        thin = evaluate(Conversion('r', 'R'),
                        series([4650] * 288, 'avgLowPrice'),
                        series([4900] * 288, 'avgHighPrice'), RULES)
        self.assertTrue(any('of price' in r for r in thin.reasons))
        quiet = evaluate(Conversion('h', 'H'),
                         series([300] * 288, 'avgLowPrice', vol=1),
                         series([600] * 288, 'avgHighPrice', vol=1), RULES)
        self.assertTrue(any('volume' in r for r in quiet.reasons))
        none = evaluate(Conversion('n', 'N'), [], [], RULES)
        self.assertEqual(none.reasons, ['no prices'])

    def test_select_and_quantities(self):
        a, b, c = ev(source='a'), ev(source='b'), ev(source='c', ok=False)
        b.median_margin = 300
        limits = {'a': 13000, 'b': 1000, 'c': 99999}
        ranked = select([a, b, c], limits)
        self.assertEqual([e.conversion.source for e in ranked], ['a', 'b'])
        self.assertEqual(plan_quantities(ranked, limits, 13500, 8),
                         {'a': 13000, 'b': 500})
        self.assertEqual(plan_quantities(ranked, limits, 99999, 1),
                         {'a': 13000})


class TestOfferBook(unittest.TestCase):

    def test_base_prices(self):
        buy, sell = OfferBook().prices(ev(), RULES)
        self.assertEqual((buy.price, sell.price), (1005, 1195))

    def test_slow_steps_up_complete_backs_off(self):
        book = OfferBook()
        step = book.step(1000)
        slow = Fill('Src', 'buy', 1005, 100, 10, hours=5)
        self.assertEqual(book.learn(slow, 1000, 5), step)
        self.assertEqual(book.prices(ev(), RULES)[0].price, 1005 + step)
        done = Fill('Src', 'buy', 1015, 100, 100, hours=1)
        self.assertEqual(book.learn(done, 1000, 5), step - step // 2)

    def test_young_partial_offer_unchanged(self):
        self.assertEqual(OfferBook().learn(Fill('X', 'sell', 1, 100, 10,
                                                0.5), 100, 5), 0)

    def test_limits(self):
        book = OfferBook()
        for _ in range(50):
            book.learn(Fill('X', 'sell', 1, 10, 0, 9), 1000, 5)
        self.assertEqual(book.get('sell', 'X'), 6 * book.step(1000))
        for _ in range(50):
            book.learn(Fill('X', 'sell', 1, 10, 10, 9), 1000, 5)
        self.assertEqual(book.get('sell', 'X'), -5)

    def test_margin_floor(self):
        buy, _ = OfferBook(extra={'buy:Src': 500}).prices(ev(), RULES)
        self.assertLessEqual(margin(buy.price, 1195), RULES.min_margin + 1)
        self.assertIn('floor', buy.why)
        buy, sell = OfferBook().prices(ev(buy=1150, sell=1200), RULES)
        self.assertIsNone(buy.price)
        self.assertIsNotNone(sell.price)

    def test_save_load(self):
        path = os.path.join(tempfile.mkdtemp(), 'm.local.json')
        OfferBook(extra={'buy:a': 7}, path=path).save()
        self.assertEqual(OfferBook.load(path).get('buy', 'a'), 7)


if __name__ == '__main__':
    unittest.main()
