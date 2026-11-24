from unittest import TestCase, mock

from wizard_eyes.application import Application
from wizard_eyes.dynamic_menus.widget import AbstractWidget


def widget(auto_locate, located):
    # skip __init__ (loads templates): only the click guard is under test
    cls = type('Widget', (AbstractWidget,), {
        'interface_class': None, '__abstractmethods__': frozenset()})
    w = object.__new__(cls)
    w.name, w.auto_locate, w.located = 'inventory', auto_locate, located
    w.client, w._clicked = mock.MagicMock(), []
    w._bbox = (0, 0, 10, 10)
    w.client.screen.click_aoi.return_value = (5, 5)
    w.client.screen.map_between.return_value = 1
    return w


class TestWidgetClickable(TestCase):

    def test_unlocated_is_not_clicked(self):
        w = widget(auto_locate=True, located=False)
        self.assertEqual(w.click(), (-1, -1))
        w.client.screen.click_aoi.assert_not_called()

    def test_located_is_clicked(self):
        w = widget(auto_locate=True, located=True)
        self.assertEqual(w.click(), (5, 5))

    def test_hand_positioned_is_clicked(self):
        w = widget(auto_locate=False, located=False)
        self.assertEqual(w.click(), (5, 5))

    def test_click_tab_waits(self):
        app, w = mock.MagicMock(msg=[]), widget(True, False)
        Application._click_tab(app, w)
        w.client.screen.click_aoi.assert_not_called()
        self.assertIn('not located yet', app.msg[0])
