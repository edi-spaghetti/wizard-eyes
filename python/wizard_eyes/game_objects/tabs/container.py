from typing import Union

from .widget import TabWidget
from .interface import TabInterface
from ...dynamic_menus.container import AbstractContainer


class Tabs(AbstractContainer):
    """Container for the main screen tabs."""

    PATH_TEMPLATE = '{root}/data/tabs/{name}.npy'
    STATIC_TABS = [
        'combat',
        'stats',
        'inventory',
        'equipment',
        'prayer',
    ]

    MUTABLE_TABS = {
        'spellbook': ['standard', 'ancient', 'lunar', 'arceuus'],
        'influence': ['quests', 'diary']
    }
    PERMUTATIONS = ['selected']

    DISABLED = 'disabled'

    def __init__(self, client):

        super().__init__(
            client, client, config_path='tabs',
            container_name='personal_menu',
        )

        # add in placeholders for the tabs we expect to find (this will
        # helper the linter)
        # TODO: handle mutable tabs e.g. quests/achievement diary or spellbooks
        self.combat: Union[TabWidget, None] = None
        self.stats: Union[TabWidget, None] = None
        self.inventory: Union[TabWidget, None] = None
        self.equipment: Union[TabWidget, None] = None
        self.prayer: Union[TabWidget, None] = None
        self.spellbook: Union[TabWidget, None] = None
        self.influence: Union[TabWidget, None] = None

    @property
    def width(self):
        # TODO: double tab stack if client width below threshold
        return self.config['width'] * 13

    @property
    def height(self):
        # TODO: double tab stack if client width below threshold
        return self.config['height'] * 1

    @property
    def interface_class(self):
        """Interface object used for layout only."""
        return TabInterface

    @property
    def interface_init_params(self):
        return (self.client, self.client), dict()

    def load_widget_masks(self, names=None):
        """Set every template to use the same tab mask."""

        self.load_masks(['tab'], cache=True)
        tab_mask = self.masks.get('tab')
        for name in names:
            self._masks[name] = tab_mask

        self._masks[self.DISABLED] = tab_mask

    @property
    def widget_class(self):
        """
        Define the specific class for tab items on the control panel tabs.
        """
        return TabWidget

    def widget_masks(self, tab):
        """All tabs use the same mask, so return static value."""
        return ['tab']

    def widget_templates(self, all_widget_names, cur_widget_name):
        """"""
        templates = super().widget_templates(all_widget_names, cur_widget_name)
        templates.append(self.DISABLED)

        return templates
