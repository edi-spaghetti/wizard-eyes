from typing import Tuple, List, Union, Dict
from copy import deepcopy
from abc import ABC, abstractmethod
from collections import defaultdict

import cv2
import numpy

from ..game_objects.game_objects import GameObject
from ..script_utils import weighted_random
from .icon import AbstractIcon

import wizard_eyes.client


class IconTracker(object):

    def __init__(self):
        self._group_mapping = dict()
        self._tracker = dict()

    def add_grouping(self, names: Tuple[str]):
        """

        :param tuple names: One or more icon names that will be grouped
            together when checking for icon changes.
        """
        self._tracker[names] = dict(newest_at=-float('inf'),
                                    new_this_frame=False)
        for name in names:
            self._group_mapping[name] = names

    def get_grouping(self, name):

        try:
            key = self._group_mapping[name]
        except KeyError:
            key = name

        try:
            data = self._tracker[key]
        except KeyError:
            data = {}

        return data

    def update(self, icons: List):
        """
        Update tracking data to check what icons changed state and when.
        """

        prev_tracker = deepcopy(self._tracker)

        for group, data in self._tracker.items():
            # reset values so we can find the newest as of *this frame*
            data['newest_at'] = -float('inf')
            data['new_this_frame'] = False
            prev_newest = prev_tracker[group]['newest_at']

            for icon in icons:
                if icon.state not in group:
                    continue

                # if this icon is newer then it becomes the current newest
                if (icon.state_changed_at or float('-inf')) > data['newest_at']:
                    data['newest_at'] = icon.state_changed_at

                    # if the icon is newer than the newest found last frame,
                    # it means the icon is newly detected, which we can use to
                    # reset timers etc.
                    if icon.state_changed_at > prev_newest:
                        data['new_this_frame'] = True


class AbstractInterface(GameObject, ABC):
    """
    The interface area that becomes available on clicking on the tabs on
    the main screen. For example, inventory, prayer etc.
    Note, these interfaces, and the icons they contain, are intended to be
    generated dynamically, rather than the rigid structure enforced by e.g.
    :class:`Inventory`.
    """

    def __init__(self, client: "wizard_eyes.client.Client", widget,
                 *args, **kwargs):
        super().__init__(
            client, client, *args, **kwargs)

        # don't save as self.parent because that implies the parent contains
        # the child, which in this case is not true.
        self.widget = widget

        self.icons: Dict[str, AbstractIcon] = dict()

        # convenience variables for counting icons as they're updated
        self.icon_count = None
        self.state_count = None
        self.icon_tracker = None

    @property
    @abstractmethod
    def icon_class(self):
        """"""

    def locate_icons(self, template_mapping, update=False):
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
        :param bool update: If true the located icons will be updated on the
            existing icons dict, otherwise the icons dict will be overwritten.

        """

        # we'll need these vectors to convert matches icons to global later
        px1, py1, _, _ = self.get_bbox()

        for icon_name, data in template_mapping.items():

            threshold = data.get('threshold', 0.99)
            quantity = data.get('quantity', 1)
            templates = data.get('templates', [])
            method = data.get('method', cv2.TM_CCOEFF_NORMED)
            invert = data.get('invert', False)
            count = len(self.icons)

            img = self.img
            if invert:
                img = cv2.bitwise_not(img)

            for template_name in templates:
                template = self.templates.get(template_name)
                mask = self.masks.get(template_name)

                if invert:
                    template = cv2.bitwise_not(template)

                matches = cv2.matchTemplate(
                    img, template, method,
                    mask=mask,
                )

                if method == cv2.TM_CCOEFF_NORMED:
                    (my, mx) = numpy.where(matches >= threshold)
                elif method == cv2.TM_SQDIFF_NORMED:
                    (my, mx) = numpy.where(matches <= threshold)
                else:
                    (my, mx) = [], []

                h, w = template.shape

                for y, x in zip(my, mx):
                    x1 = x + px1
                    y1 = y + py1
                    x2 = x1 + w - 1
                    y2 = y1 + h - 1

                    # check if we already found an icon at this location
                    found_existing = False
                    for icon in self.icons.values():
                        if icon.get_bbox() == (x1, y1, x2, y2):
                            # update the icon in case it has changed its
                            # contents since the last time we tried to locate
                            icon.update()
                            found_existing = True
                            break
                    # if we already found an icon here, nothing left to do
                    if found_existing:
                        continue

                    # if we're only going to have one of the icons, don't
                    # append a number to keep the namespace clean
                    if quantity == 1 and not update:
                        name = icon_name
                    else:
                        name = f'{icon_name}{count}'

                    icon = self.icon_class(
                        name, self.client, self,
                        threshold=threshold, type_=icon_name)
                    icon.TEMPLATE_THRESHOLD = threshold
                    icon.TEMPLATE_METHOD = method
                    icon.TEMPLATE_INVERT = invert

                    icon.set_aoi(x1, y1, x2, y2)
                    icon.load_templates(templates)
                    icon.load_masks(templates)
                    icon.update()

                    self.icons[name] = icon
                    setattr(self, name, icon)
                    self.logger.debug(f'{name} from template: {template_name}')

                    # update the counter to ensure we only create as many
                    # as we need
                    count += 1

                    if count >= quantity:
                        break
                if count >= quantity:
                    break

    def add_icon_tracker_grouping(self, names):
        """
        Use a data structure to keep track of when groupings of icon
        have detected changes. Once initialised, it will be updated every
        time :meth:`TabInterface.update` is called.

        :param tuple names: One or more icon names that will be grouped
            together when checking for new items
        """

        # init data structure if we don't have one already
        if self.icon_tracker is None:
            self.icon_tracker = IconTracker()

        self.icon_tracker.add_grouping(names)

    def update_icon_tracker(self):
        """
        Update the icon tracker (if there is one).
        """

        if self.icon_tracker is None:
            return

        self.icon_tracker.update(self.icons.values())

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

    def icons_by_state(self, *states):
        """Return a list of icons with the desired state."""

        icons = list()
        states = set(states)

        for icon in self.icons.values():
            if icon.state in states:
                icons.append(icon)

        return icons

    def choose_target_icon(self, *names,
                           clicked=None) -> Union[AbstractIcon, None]:

        candidates = self.icons_by_state(*names)
        mx, my = self.client.screen.mouse_xy
        distances = list()
        for candidate in candidates:

            if clicked is not None:
                if bool(candidate.clicked) != clicked:
                    continue

            cx, cy = self.client.screen.distribute_normally(
                *candidate.get_bbox())

            # don't square root, so we get a stronger weight on closer items
            distance = (cx - mx) ** 2 + (cy - my) ** 2 or 0.01
            distances.append(distance)

        return weighted_random(candidates, distances)

    def update(self, selected=False):
        """
        Run update on each of the icons (if the tab is selected - and the
        interface therefore open)
        Note, it does not update click timeouts, as this class should not be
        clicked directly (attempting to do so throws a warning).
        """

        super().update()

        if selected:

            # init counter variables at zero
            self.icon_count = 0
            self.state_count = defaultdict(int)

            for icon in self.icons.values():
                icon.update()

                # update our counters
                self.icon_count += 1
                self.state_count[icon.state] += 1

            # once all icons have been updated, then update the tracker
            self.update_icon_tracker()
