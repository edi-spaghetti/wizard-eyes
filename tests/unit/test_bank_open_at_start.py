from unittest import TestCase, mock

from wizard_eyes.application import Application


def tab(name='inventory', auto=True, located=False):
    t = mock.MagicMock(auto_locate=auto, located=located)
    t.name = name
    return t


class TestBankOpenAtStart(TestCase):

    def app(self, *tabs, bank_open=True):
        cls = type('App', (Application,), {})
        cls.__abstractmethods__ = frozenset()
        app = object.__new__(cls)
        app.client = mock.MagicMock()
        app.client.bank.open = bank_open
        app.client.bank.clicked = False
        app.client.tabs.widgets = list(tabs)
        app.afk_timer = mock.MagicMock()
        app.continue_ = True
        return app

    def test_closes_when_tab_not_located(self):
        app = self.app(tab())
        self.assertTrue(app.close_bank_to_locate_tabs())
        app.client.bank.close.assert_called_once()

    def test_no_input_when_not_needed(self):
        for app in (self.app(tab(located=True)),
                    self.app(tab(auto=False)),
                    self.app(tab(name='logout')),
                    self.app(tab(), bank_open=False)):
            self.assertFalse(app.close_bank_to_locate_tabs())
            app.client.bank.close.assert_not_called()

    def test_gives_up(self):
        app = self.app(tab())
        for _ in range(3):
            self.assertTrue(app.close_bank_to_locate_tabs())
        self.assertEqual(app.client.bank.close.call_count, 2)
        self.assertFalse(app.continue_)
