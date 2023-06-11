from .interface import BankInterface
from .widget import BankWidget
from ...dynamic_menus.container import AbstractContainer


class Bank(AbstractContainer):

    CLOSE_HOTKEY = 'esc'

    STATIC_TABS = ['tabINF', 'tab0', 'tab1', 'tab2', 'tab3']
    PERMUTATIONS = ['selected']

    def __init__(self, client):
        super().__init__(client, client)

        # this is the 'all items' tab we can assume is always present.
        self.tabINF = self.create_widget('tabINF')
        # bank tabs are fairly arbitrary, as they depend on the templates we
        # provide them. Can't really see a situation where we'd need more than
        # 4 tabs though.
        self.tab0 = self.create_widget('tab0')
        self.tab1 = self.create_widget('tab1')
        self.tab2 = self.create_widget('tab2')
        self.tab3 = self.create_widget('tab3')

    @property
    def interface_class(self):
        return BankInterface

    @property
    def widget_class(self):
        return BankWidget

    def close(self):

        self.client.screen.press_key(self.CLOSE_HOTKEY)
        offset = self.client.screen.map_between(start=0.6, stop=1.2)
        self.add_timeout(offset)

        return self.CLOSE_HOTKEY
