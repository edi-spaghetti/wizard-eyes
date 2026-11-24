import logging
import types
import unittest
from unittest import mock

import numpy

from wizard_eyes import player_trade as pt
from wizard_eyes.coordination import CoordinatorUnavailable
from wizard_eyes.player_trade import Give, PlayerTrade, TradeSignal
from wizard_eyes.screen_text import alike, lines, menu_line, pick


class FakeService:
    """The coordination service, seen by one bot."""

    def __init__(self, bot, data):
        self.bot, self.data, self.down = bot, data, False

    def set_shared(self, key, value):
        if self.down:
            raise CoordinatorUnavailable('down')
        self.data[key] = {'value': value}

    def shared(self):
        if self.down:
            raise CoordinatorUnavailable('down')
        return self.data


class TestText(unittest.TestCase):

    def test_menu(self):
        texts = ['Messaqe', 'Delete"', 'Add Note', 'Hop-to',
                 'Hide notifications', 'Cancel']
        self.assertEqual(menu_line(texts, 'Hop-to'), 3)
        self.assertIsNone(menu_line(texts + ['Hop to'], 'Hop-to'))
        self.assertIsNone(menu_line(texts[:3], 'Hop-to'))
        offers = ['Offer-All', 'Offer-10', 'Offer', 'Offer-X']
        self.assertEqual(menu_line(offers, 'Offer-X'), 3)

    def test_pick(self):
        names = ['LootGablin', 'Robthenobby', 'mrg splash']
        self.assertEqual(pick(names, 'LootGoblin50'), 0)
        self.assertIsNone(pick(names[1:], 'LootGoblin50'))
        menu = ['Walk here LootGoblin3@', 'Follow LootGoblin3@',
                'Trade with LootGoblin3@', 'Lookup LootGoblin3@']
        self.assertEqual(pick(menu, 'Trade with LootGoblin50', 0.65), 2)
        self.assertGreater(alike(menu[2][:10], 'Trade with'), 0.7)

    def test_lines(self):
        m = numpy.zeros((40, 10), bool)
        m[3:12, 2] = True
        m[20:29, 2] = True
        m[35, 1] = True
        self.assertEqual(lines(m), [(3, 12), (20, 29)])


class TestChat(unittest.TestCase):

    def test_short_amount(self):
        self.assertEqual([pt.short_amount(n) for n in (
            10_000_000, 54_500_000, 41_000, 14_527, 100)],
            ['10m', '54500k', '41k', '14527', '100'])

    def test_stacked_lines(self):
        m = numpy.zeros((120, 10), bool)
        m[10:66, 2] = True      # four lines without a gap
        m[100:113, 2] = True
        self.assertEqual(pt.chat_lines(m), [
            (100, 113), (52, 66), (38, 52), (24, 38), (10, 24)])


class TestSignal(unittest.TestCase):

    def test_both_ways(self):
        data, now = {}, [0.0]
        a = TradeSignal(FakeService('a', data), 't1', 'b', lambda: now[0])
        b = TradeSignal(FakeService('b', data), 't1', 'a', lambda: now[0])
        self.assertEqual(a.partner(), {})
        b.post('here', tile=[1, 2, 0])
        self.assertEqual(a.partner(), {})       # not polled again yet
        now[0] = 2
        self.assertEqual(a.partner()['tile'], [1, 2, 0])

    def test_another_trade(self):
        data = {}
        a = TradeSignal(FakeService('a', data), 't2', 'b')
        TradeSignal(FakeService('b', data), 't1', 'a').post('offered')
        self.assertEqual(a.partner(), {})

    def test_service_down(self):
        service = FakeService('a', {})
        service.down = True
        a = TradeSignal(service, 't', 'b')
        a.post('offered')
        self.assertEqual(a.partner(), {})
        self.assertEqual(a.error, 'down')


class TestTrade(unittest.TestCase):

    def setUp(self):
        self.now, self.data = 1000.0, {}
        p = mock.patch.object(pt, 'time', types.SimpleNamespace(
            time=lambda: self.now, sleep=lambda s: None))
        p.start()
        self.addCleanup(p.stop)
        log = logging.getLogger('test_player_trade')
        log.disabled = True
        app = types.SimpleNamespace(client=types.SimpleNamespace(
            logger=log, ocr=object()))
        self.partner = TradeSignal(FakeService('b', self.data), 't', 'a',
                                   lambda: self.now)
        self.t = PlayerTrade(app, 'Partner', TradeSignal(
            FakeService('a', self.data), 't', 'b', lambda: self.now))
        self.name = 'first'
        self.inventory = numpy.zeros((4, 4))
        self.clicks = []
        self.t.screen = lambda: (self.name, (0, 0) if self.name else None)
        self.t._click = lambda at, rel, pad=4: self.clicks.append(rel)
        self.t._inventory = lambda: self.inventory.copy()
        self.assertTrue(self.t._opened())

    def run_for(self, seconds):
        result, end = None, self.now + seconds
        while result is None and self.now < end:
            self.now += 0.3
            result = self.t.step()
        return result

    def test_waits_for_the_signal(self):
        self.run_for(20)
        self.assertEqual(self.data['trade-a']['value']['stage'], 'offered')
        self.assertEqual(self.clicks, [])
        self.partner.post('offered')
        self.run_for(8)
        self.assertEqual(set(self.clicks), {pt.FIRST_ACCEPT})

    def test_done(self):
        self.partner.post('offered')
        self.run_for(8)
        self.name = 'second'
        self.run_for(8)
        self.assertEqual(self.clicks[-1], pt.SECOND_ACCEPT)
        self.name = None
        self.inventory = numpy.ones((4, 4)) * 50
        self.assertTrue(self.run_for(10))
        self.assertEqual(self.data['trade-a']['value']['stage'], 'done')

    def test_screens_changing(self):
        self.partner.post('offered')
        self.run_for(8)
        self.name = None
        self.assertIsNone(self.run_for(1))
        self.name = 'second'
        self.assertIsNone(self.run_for(8))

    def test_declined(self):
        self.partner.post('offered')
        self.run_for(8)
        self.name = None
        self.assertIs(self.run_for(15), False)
        self.assertEqual(self.data['trade-a']['value']['stage'], 'failed')

    def test_partner_failed(self):
        self.t.todo = [Give('Cadantine', 5, None)]
        self.partner.post('failed', why='no menu')
        self.assertIs(self.run_for(5), False)
        self.assertIn('no menu', self.t.failed)

    def test_partner_never_offers(self):
        self.assertIs(self.run_for(200), False)
        self.assertNotIn(pt.FIRST_ACCEPT, self.clicks)


if __name__ == '__main__':
    unittest.main()
