import unittest

from wizard_eyes.orders import (Order, Slot, check_order, is_stale, reconcile)


class TestReconcile(unittest.TestCase):
    def test_keep_abort_collect_leave_alone(self):
        slots = [Slot('A', 'buy', 1000, 10),          # 2% off: kept
                 Slot('B', 'buy', 1000, 10),          # 3% off: re-listed
                 Slot('C', 'sell', 1000, 10, 10),     # complete
                 Slot('D', 'buy', 1000, 10),          # priced, not wanted
                 Slot('E', 'sell', 1000, 10)]         # not priced
        orders = [Order('A', 'buy', 1020, 99), Order('B', 'buy', 1031, 5)]
        d = reconcile(slots, orders, 'ABCD')
        self.assertEqual([s.item for s in d.keep], ['A', 'E'])
        self.assertEqual([s.item for s in d.abort], ['B', 'D'])
        self.assertEqual([s.item for s in d.collect], ['C'])
        self.assertEqual([o.item for o in d.place], ['B'])

    def test_other_side_is_a_different_offer(self):
        d = reconcile([Slot('A', 'sell', 100, 5)],
                      [Order('A', 'buy', 100, 5)], ['A'])
        self.assertEqual(len(d.abort), 1)
        self.assertEqual(len(d.place), 1)

    def test_slots_run_out(self):
        d = reconcile([Slot('X', 'sell', 1, 5)],
                      [Order(i, 'buy', 1, 1) for i in 'ABC'], 'ABC',
                      total_slots=3)
        self.assertEqual([o.item for o in d.place], ['A', 'B'])
        self.assertEqual([o.item for o in d.no_slot], ['C'])

    def test_duplicate_orders_refused(self):
        with self.assertRaises(ValueError):
            reconcile([], [Order('A', 'buy', 1, 1)] * 2, ['A'])


class TestChecks(unittest.TestCase):
    def test_price_range(self):
        self.assertIsNone(check_order(Order('A', 'buy', 900, 1), 1000, 1100))
        self.assertIsNone(check_order(Order('A', 'buy', 1210, 1), 1000, 1100))
        for price in (899, 1211):
            self.assertIn('outside', check_order(
                Order('A', 'buy', price, 1), 1000, 1100))

    def test_refusals(self):
        self.assertIn('quantity', check_order(
            Order('A', 'buy', 100, 201), 100, 100, 200))
        self.assertIn('no market', check_order(
            Order('A', 'buy', 100, 1), None, 100))
        self.assertIn('side', check_order(Order('A', 'x', 100, 1), 100, 100))
        self.assertIn('least 1', check_order(
            Order('A', 'buy', 100, 0), 100, 100))

    def test_stale(self):
        self.assertFalse(is_stale(1000, 1500, 600))
        self.assertTrue(is_stale(1000, 1601, 600))
        self.assertTrue(is_stale(1000, 900, 600))


if __name__ == '__main__':
    unittest.main()
