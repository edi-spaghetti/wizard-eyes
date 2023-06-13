from typing import Tuple, List, Union, Dict, Type
from copy import deepcopy
from abc import ABC, abstractmethod
from collections import defaultdict
from os.path import exists

import cv2
import numpy

from ..game_objects.game_objects import GameObject
from ..game_objects.template import TemplateGroup, Template
from ..script_utils import weighted_random
from ..file_path_utils import get_root
from ..constants import REDA
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
    generated dynamically, either by a pre-defined alpha or by template
    matching.

    Note that templates loaded to this class are expected to be loaded into
    the icons it generates. The interface itself is located using `self.frame`.

    """

    PATH_TEMPLATE = '{root}/data/{container}/{widget}/{name}.npy'

    ALPHA_MAPPING: Dict[str,
        Dict[Tuple[int, int, int, int], Tuple[str, int]]] = {}
    """Mapping of alpha colours for the icons they represent. Alpha colours
    must be unique within the interface to find the bounding box. 

    Mapping of should be in the form:
    ```
        {
            <widget name>: {
                <bgra colour>: (<template group name>, <quantity>)
            }
        }
    ```
    
    """

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f'{self.__class__.__name__}<{self.widget.name}>'

    def __init__(
            self, client: 'wizard_eyes.client.Client',
            widget: Type['AbstractWidget'],  # noqa: 262
            *args, **kwargs):
        super().__init__(client, client, *args, **kwargs)

        # don't save as self.parent because that implies the parent contains
        # the child, which in this case is not true.
        self.widget: Type['AbstractWidget'] = widget  # noqa: 262
        self.located: bool = False
        self.alpha = self.get_alpha()
        self.frame = self.load_templates(['frame'], cache=False).get('frame')
        self.frame_mask = self.load_masks(['frame'], cache=False).get('frame')
        self.match_threshold = .99

        self.icons: Dict[str, AbstractIcon] = dict()

        # convenience variables for counting icons as they're updated
        self.icon_count = None
        self.state_count = None
        self.icon_tracker = None

        self.colour = REDA

    def get_alpha(self):
        path = self.resolve_path(
            name='alpha', root=get_root()).replace('.npy', '.png')
        if exists(path):
            return cv2.imread(path)
        return numpy.uint8([[[0, 0, 0]]])

    def resolve_path(self, **kwargs):
        kwargs['container'] = self.widget.parent.name
        kwargs['widget'] = self.widget.name
        return super().resolve_path(**kwargs)

    @property
    @abstractmethod
    def icon_class(self):
        """"""

    def sort_unique_alpha(self, unique):
        """Optionally override the sort order for unique alpha colours."""
        return unique

    def create_template_groups_from_alpha_mapping(
            self, templates, aliases=None):
        """This should be run when setting up an interface before any widgets
        or interfaces have been located."""

        mapping = self.ALPHA_MAPPING.get(self.widget.name)
        if not mapping:
            return

        aliases = aliases or [None] * len(templates)
        for colour, (name, quantity) in mapping.items():
            group = self.create_template_group(name, colour, quantity)

            for template, alias in zip(templates, aliases):
                self.add_template_to_group(
                    group, Template(template, alias=alias))

    def template_group_from_colour(self, colour):
        for group in self.template_groups:
            if group.colour == colour:
                return group

        group = TemplateGroup(
            name=self.widget.name, templates=[], colour=colour)
        return group

    def locate_icons_by_alpha(self):
        """Locate icons from a pre-defined alpha data image."""

        if not self.located:
            self.logger.warning(
                'Cannot locate icons before interface is located!')
            return

        if not self.alpha.any():
            self.logger.warning(
                f'No alpha data found for {self.widget.name} interface!')
            return

        unique = numpy.unique(
            self.alpha.reshape(-1, self.alpha.shape[2]), axis=0)
        unique = self.sort_unique_alpha(unique)

        i = 0
        for bgr in unique:
            if not bgr.any():
                continue

            bgra = cv2.cvtColor(bgr.reshape(1, 1, 3), cv2.COLOR_BGR2BGRA)
            colour = tuple(int(c) for c in bgra.reshape(4))

            group = self.template_group_from_colour(colour)

            if group.quantity == 1:
                name = group.name
            else:
                name = f'{group.name}{i}'

            icon = self.icon_class(
                name, self.client, self, type_=group.name)
            icon.DEFAULT_COLOUR = colour

            # set up bounding box
            ay, ax = numpy.where(
                numpy.all(
                    self.alpha == colour[:3], axis=-1
                )
            )
            x1, y1, x2, y2 = min(ax), min(ay), max(ax), max(ay)
            x1, y1, x2, y2 = self.globalise(x1, y1, x2, y2)
            icon.set_aoi(x1, y1, x2, y2)

            # set up templates
            templates = [t.name for t in group.templates]
            icon.load_templates(templates)
            icon.load_masks(templates)
            for template in group.templates:
                if template.alias:
                    icon.load_masks([template.alias])
                    icon.alias_mask(template.name, template.alias)

            i += 1
            icon.DETECT_ANYTHING = True
            self.icons[icon.name] = icon
            setattr(self, icon.name, icon)

        return True

    def locate_icons_by_template(self, update=False):
        """
        Attempt to locate icons within the interface, and generate a game
        object for them if found. Icon objects are added to a dictionary
        available at :attr:`TabInterface.icons` and also an instance attribute
        of the same name. Note, this means icons must have unique names per
        interface!

        :param bool update: If true the located icons will be updated on the
            existing icons dict, otherwise the icons dict will be overwritten.

        """

        if not self.located:
            self.logger.warning(
                'Cannot locate icons before interface is located!')
            return

        # we'll need these vectors to convert matches icons to global later
        px1, py1, _, _ = self.get_bbox()

        # if we need an inverted image for any of the templates,
        # make the inversion once at the top here
        img = self.img
        inverted_img = None
        for group in self.template_groups:
            for template in group.templates:
                if template.invert:
                    inverted_img = cv2.bitwise_not(img)

        for group in self.template_groups:
            count = 0

            for template in group.templates:

                img = self.img
                if template.invert:
                    img = inverted_img

                template_image = template.image
                if template.invert:
                    # TODO: invert template image once of init
                    template_image = cv2.bitwise_not(template)

                try:
                    matches = cv2.matchTemplate(
                        img, template_image, template.method,
                        mask=template.mask,
                    )
                except cv2.error:
                    continue

                if template.method == cv2.TM_CCOEFF_NORMED:
                    (my, mx) = numpy.where(matches >= template.threshold)
                elif template.method == cv2.TM_SQDIFF_NORMED:
                    (my, mx) = numpy.where(matches <= template.threshold)
                else:
                    (my, mx) = [], []

                h, w = template.image.shape

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
                    if group.quantity == 1 and not update:
                        name = group.name
                    else:
                        name = f'{group.name}{count}'

                    icon = self.icon_class(
                        name, self.client, self, type_=group.name)
                    # TODO: set up icon with template groups instead of
                    #       straight templates - this will break if any of the
                    #       templates have different settings.
                    icon.match_threshold = template.threshold
                    icon.match_method = template.method
                    icon.match_invert = template.invert
                    icon.DEFAULT_COLOUR = group.colour

                    icon.set_aoi(x1, y1, x2, y2)
                    templates = [t.name for t in group.templates]
                    icon.load_templates(templates)
                    icon.load_masks(templates)
                    for template_ in group.templates:
                        if template_.alias:
                            icon.load_masks([template_.alias])
                            icon.alias_mask(template_.alias, template_.name)
                    icon.update()

                    self.icons[name] = icon
                    setattr(self, name, icon)
                    self.logger.debug(f'{name} from template: {template.name}')

                    # update the counter to ensure we only create as many
                    # as we need
                    count += 1

                    if count >= group.quantity:
                        break
                if count >= group.quantity:
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

    def locate(self):
        """Locate the interface itself. This method can also be used to check
        if the interface is currently open."""

        if self.frame is None:
            return False

        mask = self.frame_mask
        matches = cv2.matchTemplate(
            # must be client img, because we don't know where
            # the widget is yet
            self.client.img,
            self.frame,
            cv2.TM_CCOEFF_NORMED,
            mask=mask
        )

        (my, mx) = numpy.where(matches >= self.match_threshold)
        if len(mx) > 1:
            self.logger.warning(
                f'Found {len(mx)} matches for {self.widget.name}, '
                'assuming the first one is correct.')

        # assume interfaces are unique and we only get one match
        for y, x in zip(my, mx):
            h, w = self.frame.shape
            x1, y1, x2, y2 = self.client.globalise(x, y, x + w - 1, y + h - 1)
            self.set_aoi(x1, y1, x2, y2)
            return True

        return False

    def draw(self):
        bboxes = {
            '*bbox',
            f'{self.widget.name}_bbox',
            f'{self.widget.name}_interface_bbox'
        }
        if self.client.args.show.intersection(bboxes) and self.located:
            self.draw_bbox()

    def update(self, selected=False):
        """
        Run update on each of the icons (if the tab is selected - and the
        interface therefore open)
        Note, it does not update click timeouts, as this class should not be
        clicked directly (attempting to do so throws a warning).
        """

        super().update()
        if self.covered_by_right_click_menu():
            return

        if selected:

            if not self.located:
                if self.widget.auto_locate:
                    if not self.locate():
                        return
                    self.located = True
                    result = self.locate_icons_by_alpha()
                    if not result:
                        self.locate_icons_by_template()
                else:
                    return

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
