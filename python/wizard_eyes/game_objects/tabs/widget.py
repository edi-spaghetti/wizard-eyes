from .interface import TabInterface, LogoutInterface
from ...dynamic_menus.widget import AbstractWidget

import  numpy


class TabWidget(AbstractWidget):

    def __init__(self, *args, **kwargs):
        super(TabWidget, self).__init__(*args, **kwargs)

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
        if result and self.name != 'logout':
            self.load_templates(['disabled'])
            self.alias_mask('tab', 'disabled')

        return result


class LogoutWidget(AbstractWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_templates(['logout', 'logout_selected'])
        self.load_masks(['logout', 'logout_selected'])

    @property
    def interface_class(self):
        """Interface class associated with current widget."""
        return LogoutInterface

    def locate(self) -> bool:
        """Locate the logout widget per the gauges' alpha.

        The logout widget is actually a static object linked to the
        gauges/minimap, but we still categorise it as a tab widget because
        it opens an interface the same as e.g. inventory.

        """

        if not self.client.gauges.located:
            return False

        # gauges is located, so we can assume logout stub widget has been
        # allocated a bounding box per the alpha.
        self.set_aoi(*self.client.gauges.logout.get_bbox())
        return True

    def identify(self, threshold=None):
        """Identify by a mere pixel.

        The logout widget is (mostly) transparent, which causes real issues
        with template matching. Luckily, the botton left pixel is overlapping
        the minimap gauge, which gives us a solid colour.

        Ideally we'd still use the templates here, but they get converted to
        greyscale and lose the vital info we need.

        """

        img = self.client.get_img_at(self.get_bbox(), mode=self.client.BGRA)
        if numpy.array_equal(img[22, 0], numpy.array([10, 16, 19, 255])):
            return 'logout'
        elif numpy.array_equal(img[22, 0], numpy.array([1, 0, 0, 255])):
            return 'logout_selected'
        return ''
