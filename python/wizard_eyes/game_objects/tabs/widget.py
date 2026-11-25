from .interface import TabInterface, LogoutInterface
from . import logout_button
from ...dynamic_menus.widget import AbstractWidget
from ...file_path_utils import ensure_local, get_root

import numpy


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
        self._reference = None

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

    @property
    def reference(self) -> numpy.ndarray:
        """The open tab's X (BGRA, transparent around it)."""
        if self._reference is None:
            path = self.resolve_path(name='logout_reference', root=get_root())
            ensure_local(path)
            self._reference = numpy.load(path)
        return self._reference

    def identify(self, threshold=None):
        """Identify by how solid the X is (see ``logout_button``): it's
        transparent when closed, so normal template matching won't do."""
        img = self.client.get_img_at(self.get_bbox(), mode=self.client.BGRA)
        return logout_button.state(img, self.reference)
