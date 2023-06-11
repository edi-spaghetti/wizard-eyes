from .widget import TabWidget
from .interface import TabInterface
from ...dynamic_menus.container import AbstractContainer


class Tabs(AbstractContainer):
    """Container for the main screen tabs."""

    STATIC_TABS = [
        'combat',
        'stats',
        'inventory',
        'equipment',
        'prayer',
        'logout',
    ]

    MUTABLE_TABS = {
        'spellbook': ['standard', 'ancient', 'lunar', 'arceuus'],
        'influence': ['quests', 'diary']
    }
    PERMUTATIONS = ['selected']

    def __init__(self, client):
        super().__init__(client, client)

        self.combat: TabWidget = self.create_widget('combat')
        self.stats: TabWidget = self.create_widget('stats')
        self.inventory: TabWidget = self.create_widget('inventory')
        self.equipment: TabWidget = self.create_widget('equipment')
        self.prayer: TabWidget = self.create_widget('prayer')
        self.spellbook: TabWidget = self.create_widget('spellbook')
        self.influence: TabWidget = self.create_widget('influence')
        self.logout: TabWidget = self.create_widget('logout')  # TODO: thresh

    @property
    def interface_class(self):
        """Interface object used for layout only."""
        return TabInterface

    @property
    def widget_class(self):
        """
        Define the specific class for tab items on the control panel tabs.
        """
        return TabWidget
