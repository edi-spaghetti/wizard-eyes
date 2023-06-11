from .interface import TabInterface
from ...dynamic_menus.widget import AbstractWidget


class TabWidget(AbstractWidget):

    def __init__(self, *args, **kwargs):
        super(TabWidget, self).__init__(*args, **kwargs)

        if self.name != 'logout':
            self.load_masks(['tab'])
            for template in self.templates:
                self.alias_mask('tab', template)

    @property
    def interface_class(self):
        """Interface class associated with current widget."""
        return TabInterface

    def locate(self):
        """Attempt to find the widget within the client. Once the widget has
        been located, the disabled template will be loaded and aliased. We do
        this after location, because if for example we're in the bank, the
        tabs are all disabled, so we'd end up with every tab being located
        at the same place."""

        result = super().locate()

        # now that we've located the widget we can add the disabled template
        # without worrying about false matches.
        if result:
            self.load_templates(['disabled'])
            self.alias_mask('tab', 'disabled')

        return result
