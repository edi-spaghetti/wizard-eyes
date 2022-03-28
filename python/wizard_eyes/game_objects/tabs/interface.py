from collections import defaultdict

import cv2
import numpy

from .icon import InterfaceIcon
from ..game_objects import GameObject


class TabInterface(GameObject):
    """
    The interface area that becomes available on clicking on the tabs on
    the main screen. For example, inventory, prayer etc.
    Note, these interfaces, and the icons they contain, are intended to be
    generated dynamically, rather than the rigid structure enforced by e.g.
    :class:`Inventory`.
    """

    PATH_TEMPLATE = '{root}/data/tabs/{name}.npy'

    def __init__(self, client, parent):
        super(TabInterface, self).__init__(
            client, client, config_path='tabs.interface',
            container_name='dynamic_tabs')

        # don't save as self.parent because that implies the parent contains
        # the child, which in this case is not true.
        self.parent_tab = parent

        self.icons = dict()

        # convenience variables for counting icons as they're updated
        self.icon_count = None
        self.state_count = None

    def locate_icons(self, template_mapping):
        """
        Attempt to locate icons within the interface, and generate a game
        object for them if found. Icon objects are added to a dictionary
        available at :attr:`TabInterface.icons` and also an instance attribute
        of the same name. Note, this means icons must have unique names per
        interface!

        :param template_mapping: A mapping of the name to assign to the icon,
            and a list of template names that apply to that icon. For example,
            you may have different templates for one gp, two gp etc. but they
            all represent the inventory icon 'gold_pieces'.

        """

        # we'll need these vectors to convert matches icons to global later
        px1, py1, _, _ = self.get_bbox()

        for icon_name, data in template_mapping.items():

            threshold = data.get('threshold', 0.99)
            quantity = data.get('quantity', 1)
            templates = data.get('templates', [])
            count = 0

            for template_name in templates:
                template = self.templates.get(template_name)
                mask = self.masks.get(template_name)

                matches = cv2.matchTemplate(
                    self.img, template, cv2.TM_CCOEFF_NORMED,
                    mask=mask,
                )
                (my, mx) = numpy.where(matches >= threshold)

                h, w = template.shape

                for y, x in zip(my, mx):
                    x1 = x + px1
                    y1 = y + py1
                    x2 = x1 + w - 1
                    y2 = y1 + h - 1

                    # if we're only going to have one of the icons, don't
                    # append a number to keep the namespace clean
                    if quantity == 1:
                        name = icon_name
                    else:
                        name = f'{icon_name}{count}'

                    icon = InterfaceIcon(
                        name, self.client, self,
                        threshold=threshold, type_=icon_name)
                    icon.set_aoi(x1, y1, x2, y2)
                    icon.load_templates(templates)
                    icon.load_masks(templates)

                    self.icons[name] = icon
                    setattr(self, name, icon)
                    self.logger.debug(f'{name} from template: {template_name}')

                    # increase the counter to ensure we only create as many
                    # as we need
                    count += 1

                    if count >= quantity:
                        break
                if count >= quantity:
                    break

    def _click(self, *args, **kwargs):
        self.logger.warning('Do not click container, click the icons.')

    def sum_states(self, *states):
        """Calculate sum total of icons with the given state(s)."""

        # just in case we try to call this before updating at least once
        if self.state_count is None:
            return 0

        count = 0
        for state in states:
            count += self.state_count.get(state, 0)

        return count

    def update(self):
        """
        Run update on each of the icons (if the tab is selected - and the
        interface therefore open)
        Note, it does not update click timeouts, as this class should not be
        clicked directly (attempting to do so throws a warning).
        """

        if self.parent_tab.selected:

            # init counter variables at zero
            self.icon_count = 0
            self.state_count = defaultdict(int)

            for icon in self.icons.values():
                icon.update()

                # update our counters
                self.icon_count += 1
                self.state_count[icon.state] += 1
