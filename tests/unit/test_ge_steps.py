"""The GE steps (sell by name, abort, collect) against a scripted GE."""
import logging
import types
import unittest
from unittest import mock

from wizard_eyes import grand_exchange as gx
from wizard_eyes.grand_exchange import (AbortOffer, CollectOffers, OfferView,
                                        PlaceOffer)

NAMES = [f'{h} potion (unf)' for h in (
    'Guam', 'Marrentill', 'Tarromin', 'Harralander', 'Ranarr', 'Toadflax',
    'Irit', 'Avantoe', 'Kwuarm', 'Huasca', 'Snapdragon', 'Cadantine',
    'Lantadyme', 'Dwarf weed', 'Torstol')]


class FakeGE:
    """Boxes are their names, so a click is the name of what was clicked."""

    def __init__(self):
        self.screen = 'main'
        self.clicks = []
        self.on_click = {}
        self.text = {}
        self.numbers = {}
        self.view = None
        self.bar = ('active', 0.0)
        self.has_abort = True
        self.collectable = False
        self.prompt = None
        self.empty = [0, 1, 2]

    def update(self):
        return self.screen

    def click(self, box, pad=4):
        self.clicks.append(box)
        self.on_click.get(box, lambda: None)()

    setup_box = status_box = main_box = lambda self, part: part

    def slot_box(self, i, part=None):
        return f'slot{i}_{part}'

    def read_text(self, part):
        return self.text.get(part, '')

    read_status_text = read_text

    def read_number(self, part):
        return self.numbers.get(part)

    def find_prompt(self, kind, what):
        return (0, 0) if self.prompt == (kind, what) else None

    def read_slot(self, i, names):
        return self.view

    def status_progress(self):
        return self.bar

    def abort_button(self):
        return self.has_abort

    def collect_button(self):
        return self.collectable

    def empty_slots(self):
        return self.empty

    def slot_empty(self, i):
        return i in self.empty


class Base(unittest.TestCase):

    def setUp(self):
        self.now = 1000.0
        self.ge = FakeGE()
        self.typed = []
        fake_time = types.SimpleNamespace(time=lambda: self.now,
                                          sleep=lambda s: None)
        for target, value in (
                ('time', fake_time), ('pause_seconds', lambda: 1.0),
                ('GrandExchange', lambda client: self.ge),
                ('type_slowly', self.typed.append),
                ('keyboard', types.SimpleNamespace(
                    press_and_release=self.enter))):
            p = mock.patch.object(gx, target, value)
            p.start()
            self.addCleanup(p.stop)
        log = logging.getLogger('test_ge_steps')
        log.disabled = True
        self.app = types.SimpleNamespace(client=types.SimpleNamespace(
            logger=log, screen=types.SimpleNamespace(
                click_aoi=lambda *a, **k: self.inventory_click())))
        self.icon = types.SimpleNamespace(get_bbox=lambda: (0, 0, 9, 9))

    def inventory_click(self):
        self.ge.screen = 'setup'

    def enter(self, key):
        self.ge.numbers[self.ge.prompt[1]] = int(self.typed[-1])
        self.ge.prompt = None

    def run_steps(self, steps, seconds=90):
        end = self.now + seconds
        while self.now < end:
            self.now += 0.3
            result = steps.step()
            if result is not None:
                return result
        self.fail(f'still at {steps.state}')


class TestSell(Base):

    def setUp(self):
        super().setUp()
        ge = self.ge
        ge.text = {'type': 'Sell offer', 'name': 'Irit potion (unf)'}
        ge.numbers = {'quantity': 1, 'price': 2000}

        def prompt(what):
            ge.prompt = ('sell', what)

        ge.on_click = {
            'quantity_all': lambda: ge.numbers.update(quantity=7),
            'quantity_custom': lambda: prompt('quantity'),
            'price_custom': lambda: prompt('price'),
            'back': lambda: setattr(ge, 'screen', 'main'),
            'confirm': self.confirm,
        }

    def confirm(self):
        self.ge.screen = 'main'
        self.ge.empty = [1, 2]

    def offer(self, orders, **kw):
        return PlaceOffer.sell_by_name(self.app, self.icon, orders, NAMES,
                                       confirm=True, **kw)

    def test_all_at_its_price(self):
        o = self.offer({'Irit potion (unf)': (None, 1500)})
        self.assertTrue(self.run_steps(o))
        self.assertEqual((o.item, o.quantity, o.price, o.skipped),
                         ('Irit potion (unf)', 7, 1500, False))
        self.assertEqual(self.typed, ['1500'])
        self.assertEqual(self.ge.clicks,
                         ['quantity_all', 'price_custom', 'confirm'])

    def test_some(self):
        o = self.offer({'Irit potion (unf)': (3, 2000)})
        self.assertTrue(self.run_steps(o))
        self.assertEqual(self.ge.clicks, ['quantity_custom', 'confirm'])

    def test_no_order(self):
        o = self.offer({'Guam potion (unf)': (None, 5)})
        self.assertTrue(self.run_steps(o))
        self.assertTrue(o.skipped)
        self.assertEqual(self.ge.clicks, ['back'])

    def test_unreadable_name(self):
        self.ge.text['name'] = 'lrlt'
        self.assertFalse(self.run_steps(
            self.offer({'Irit potion (unf)': (None, 5)})))
        self.assertEqual(self.ge.clicks, [])

    def test_too_many(self):
        self.assertFalse(self.run_steps(
            self.offer({'Irit potion (unf)': (None, 5)}, most=6)))
        self.assertNotIn('confirm', self.ge.clicks)

    def test_dry_run(self):
        o = PlaceOffer.sell_by_name(
            self.app, self.icon, {'Irit potion (unf)': (None, 1500)}, NAMES)
        self.assertTrue(self.run_steps(o))
        self.assertEqual(self.ge.clicks[-1], 'back')
        self.assertFalse(o.skipped)

    def test_buy_needs_everything(self):
        with self.assertRaises(ValueError):
            PlaceOffer(self.app, 'buy', 'Guam leaf', None, 5)

    def test_names_told_apart(self):
        for a in NAMES:
            self.assertEqual(gx.best_name(a, NAMES, 0.85), a)
        for text, item in (('Fuuarm potion [unfl', 'Kwuarm'),
                           ('fivantoe potion [unfl', 'Avantoe'),
                           ('Fanarr potion [unfl', 'Ranarr'),
                           ('Torstal potion [unfl', 'Torstol')):
            self.assertEqual(gx.best_name(text, NAMES),
                             f'{item} potion (unf)')


class TestAbort(Base):

    def setUp(self):
        super().setUp()
        ge = self.ge
        ge.view = OfferView(1, 'buy', 'Grimy tarromin', 200, 205, 'active', 0)
        ge.text = {'type': 'Buy offer', 'name': 'Grimy tarromin'}
        ge.on_click = {
            'slot1_name': lambda: setattr(ge, 'screen', 'status'),
            'abort': lambda: setattr(ge, 'bar', ('cancelled', 1.0)),
            'back': lambda: setattr(ge, 'screen', 'main'),
        }
        self.steps = AbortOffer(self.app, 1, 'buy', 'Grimy tarromin',
                                ['Tarromin'])

    def test_abort(self):
        self.assertTrue(self.run_steps(self.steps))
        self.assertEqual(self.ge.clicks, ['slot1_name', 'abort', 'back'])
        self.assertEqual(self.steps.status, 'cancelled')

    def test_filled_meanwhile(self):
        self.ge.bar = ('complete', 1.0)
        self.assertTrue(self.run_steps(self.steps))
        self.assertEqual(self.ge.clicks, ['slot1_name', 'back'])
        self.assertEqual(self.steps.status, 'complete')

    def test_other_offer_in_slot(self):
        self.ge.view.item = 'Tarromin'
        self.assertFalse(self.run_steps(self.steps))
        self.assertEqual(self.ge.clicks, [])

    def test_other_offer_on_its_screen(self):
        self.ge.text['name'] = 'Tarromin'
        self.assertFalse(self.run_steps(self.steps))
        self.assertNotIn('abort', self.ge.clicks)

    def test_gives_up(self):
        self.ge.on_click['abort'] = lambda: None
        self.assertFalse(self.run_steps(self.steps))
        self.assertEqual(self.ge.clicks.count('abort'), 2)


class TestCollect(Base):

    def test_collect(self):
        self.ge.collectable = True
        self.ge.on_click = {
            'collect': lambda: setattr(self.ge, 'collectable', False)}
        self.assertTrue(self.run_steps(CollectOffers(self.app)))
        self.assertEqual(self.ge.clicks, ['collect'])

    def test_nothing(self):
        self.assertTrue(self.run_steps(CollectOffers(self.app)))
        self.assertEqual(self.ge.clicks, [])

    def test_gives_up(self):
        self.ge.collectable = True
        self.assertFalse(self.run_steps(CollectOffers(self.app)))
        self.assertEqual(self.ge.clicks, ['collect'] * 2)


if __name__ == '__main__':
    unittest.main()
