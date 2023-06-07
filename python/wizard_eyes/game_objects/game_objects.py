import random
import ctypes
import logging
from os.path import exists
from typing import Union, Dict, Tuple

import numpy
import cv2
import pyautogui
from PIL import Image

from .timeout import Timeout
from ..file_path_utils import get_root
from ..constants import COLOUR_DICT_HSV

import wizard_eyes.client

# TODO: use scale factor and determine current screen to apply to any config
#       values. For the time being I'm setting system scaling factor to 100%
scale_factor = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100


class GameObject(object):

    PATH_TEMPLATE = '{root}/data/{name}.npy'

    # default colour for showing client image (note: BGRA)
    DEFAULT_COLOUR = (0, 0, 0, 255)

    def __init__(self,
                 client: "wizard_eyes.client.Client",
                 parent: "GameObject",
                 config_path=None, container_name=None,
                 template_names=None, logging_level=None, data=None):
        self._logging_level = logging_level
        self.logger = self.setup_logger()

        self.client: "wizard_eyes.client.Client" = client
        self.parent = parent
        self.context_menu: Union["ContextMenu", None] = None
        self.data = data  # can be whatever you need
        self._bbox = None
        self.config = self._get_config(config_path)
        self.container_name = container_name
        self.containers = self.setup_containers()
        self.default_bbox = self.get_bbox
        self.x1_offset = 0
        self.y1_offset = 0
        self.x2_offset = 0
        self.y2_offset = 0

        self.colour = self.DEFAULT_COLOUR

        self._templates = dict()
        self._masks = dict()
        self.load_templates(names=template_names)
        self.load_masks(names=template_names)
        self.single_match = True
        self.match_invert = False
        self.match_method = cv2.TM_CCOEFF_NORMED

        # audit fields
        self._clicked = list()

    def setup_logger(self):
        # TODO: convert to singleton
        logger = logging.getLogger(self.__class__.__name__)
        handler = logging.StreamHandler()
        level = self._logging_level or logging.INFO

        handler.setLevel(level)
        if not logger.handlers:
            logger.addHandler(handler)
        logger.setLevel(level)

        return logger

    def _get_config(self, path, config=None):
        """
        Attempt to retrieve object config by recursive item access on
        client config. All objects associated with client should be nested
        inside client config, ideally in hierarchy order.
        :param str path: Dot notated path to object config
        :param None/dict config: Object config at current level
        :return: Object config or None if not found/specified
        """
        if path is None:
            return config

        config = config or self.client.config
        try:
            key, path = str(path).split('.', 1)
        except ValueError:
            key = path
            path = None

        config = config.get(key, {})
        return self._get_config(path, config)

    def _eval_config_value(self, value):
        return eval(str(value))

    @property
    def img(self):
        """
        Slice the current client image on current object's bbox.
        """
        cx1, cy1, cx2, cy2 = self.client.get_bbox()

        x1, y1, x2, y2 = self.default_bbox()
        img = self.client.img
        i_img = img[y1 - cy1:y2 - cy1 + 1, x1 - cx1:x2 - cx1 + 1]

        return i_img

    @property
    def width(self):
        val = self.config.get('width', 0)
        return self._eval_config_value(val)

    @property
    def height(self):
        val = self.config.get('height', 0)
        return self._eval_config_value(val)

    @property
    def margin_top(self):
        val = self.config.get('margins', {}).get('top', 0)
        return self._eval_config_value(val)

    @property
    def margin_bottom(self):
        val = self.config.get('margins', {}).get('bottom', 0)
        return self._eval_config_value(val)

    @property
    def margin_left(self):
        val = self.config.get('margins', {}).get('left', 0)
        return self._eval_config_value(val)

    @property
    def margin_right(self):
        val = self.config.get('margins', {}).get('right', 0)
        return self._eval_config_value(val)

    @property
    def padding_top(self):
        val = self.config.get('padding', {}).get('top', 0)
        return self._eval_config_value(val)

    @property
    def padding_bottom(self):
        val = self.config.get('padding', {}).get('bottom', 0)
        return self._eval_config_value(val)

    @property
    def padding_left(self):
        val = self.config.get('padding', {}).get('left', 0)
        return self._eval_config_value(val)

    @property
    def padding_right(self):
        val = self.config.get('padding', {}).get('right', 0)
        return self._eval_config_value(val)

    @property
    def alignment(self):
        """
        Get alignment of current object within container.
        Assume top left alignment if not defined.
        :return: list[str] of alignment keywords
        """
        return self.config.get('alignment', ['top', 'left'])

    @property
    def container(self):
        """
        Query the parent object's containers to find the current instance's
        container. That is, a list of objects aligned with current object
        within the parent's bounding box.
        :return: list[GameObject] which includes current instance
        """
        return self.parent.containers.get(
            self.container_name, dict())

    @property
    def centre(self):

        if self.get_bbox() is None:
            return

        x1, y1, x2, y2 = self.get_bbox()
        x = int((x1 + x2) / 2)
        y = int((y1 + y2) / 2)

        return x, y

    def setup_containers(self):
        """
        Containers should be defined x: left to right, y: top to bottom
        :return: Dictionary of container configuration
        """
        return dict()

    def set_aoi(self, x1, y1, x2, y2):
        self._bbox = x1, y1, x2, y2

    def get_bbox(self):

        # return cached value, if any
        # note, this makes game objects static once the client has started,
        # but is being kept for backwards compatibility with some scripts
        # creating generic game objects and setting the aoi manually.
        if self._bbox:
            return self._bbox
        # if no config is defined any bbox we get will just be the client bbox,
        # which is more than likely not what we want. just return here.
        if self.config is None:
            return

        if self.client.name != 'RuneLite':
            raise NotImplementedError

        # get parent bounding box
        px1, py1, px2, py2 = self.parent.get_bbox()

        # determine outermost x coord based on alignment
        if 'left' in self.alignment:
            x1 = px1

            x1 += self.parent.padding_left

            container = self.container.get('x', [])
            for game_object in container:
                if game_object is self:
                    break
                true_width = (
                        game_object.margin_right +
                        game_object.width +
                        game_object.margin_left
                )
                x1 += true_width

            x1 += self.margin_left

            x2 = x1 + self.width
        else:
            x2 = px2

            x2 -= self.parent.padding_right

            # cycle items backwards, because containers are defined left
            # to right
            container = self.container.get('x', [])
            for game_object in container[::-1]:
                if game_object is self:
                    break
                true_width = (
                        game_object.margin_right +
                        game_object.width +
                        game_object.margin_left
                )
                x2 -= true_width

            x2 -= self.margin_right

            x1 = x2 - self.width

        # determine outermost y coord based on alignment
        if 'top' in self.alignment:
            y1 = py1

            y1 += self.parent.padding_top

            container = self.container.get('y', [])
            for game_object in container:
                if game_object is self:
                    break
                true_height = (
                    game_object.margin_top +
                    game_object.height +
                    game_object.margin_bottom
                )
                y1 += true_height

            y1 += self.margin_top

            y2 = y1 + self.height
        else:
            y2 = py2

            y2 -= self.parent.padding_bottom

            # cycle objects in container backwards, because containers are
            # always defined top to bottom
            container = self.container.get('y', [])
            for game_object in container[::-1]:
                if game_object is self:
                    break
                true_height = (
                        game_object.margin_top +
                        game_object.height +
                        game_object.margin_bottom
                )
                y2 -= true_height

            y2 -= self.margin_bottom

            y1 = y2 - self.height

        # set internal bbox value and return
        self._bbox = x1, y1, x2, y2
        return self._bbox

    def click_box(self):
        """The click-able area of an object's bbox"""

        x1, y1, x2, y2 = self.get_bbox()
        cx1, cy1, cx2, cy2 = self.client.get_bbox()
        _, _, _, by2 = self.client.banner.get_bbox()

        # apply offsets
        # TODO: dynamic offsets based on camera position
        #       this will be tricky, because it's dependent on the 3D object,
        #       so for now static offsets are better than nothing
        x1 += self.x1_offset
        y1 += self.y1_offset
        x2 += self.x2_offset
        y2 += self.y2_offset

        if x1 < cx1:
            x1 = cx1 - 1
        if y1 < by2:  # bottom of banner = top of screen
            y1 = by2 - 1
        if x2 > cx2:
            x2 = cx2 + 1
        if y2 > cy2:
            y2 = cy2 + 1

        # TODO: add interfaces

        return x1, y1, x2, y2

    def clear_bbox(self):
        self._bbox = None

    def localise(self, x1, y1, x2, y2):
        """Convert incoming vectors to be relative to the current object."""

        cx1, cy1, _, _ = self.get_bbox()

        # convert relative to own bbox
        x1 = x1 - cx1 + 1
        y1 = y1 - cy1 + 1
        x2 = x2 - cx1 + 1
        y2 = y2 - cy1 + 1

        return x1, y1, x2, y2

    def globalise(self, x1, y1, x2, y2):
        cx1, cy1, _, _ = self.get_bbox()

        x1 = cx1 + x1
        y1 = cy1 + y1
        x2 = cx1 + x2
        y2 = cy1 + y2

        return x1, y1, x2, y2

    def update_click_timeouts(self):
        """
        Check and remove timeouts that have expired.
        Note, it uses client time, so ensure the client has been updated.
        """
        i = 0

        # determine attempt to find the earliest timeout that is in the future
        while i < len(self._clicked):
            t = self._clicked[i]
            if self.client.time < t.offset:
                # we've hit the threshold, so stop here. Timeouts are stored in
                # ascending time order, so all further timeouts will also be
                # in the future
                break
            i += 1

        # remove everything before the last index we found
        self._clicked = self._clicked[i:]

    def update_context_menu(self):
        """
        Checks the context menu is still there (if there is one in the first
        place). Usually context menus remain on screen until the move moves
        outside of their bounding box.
        """
        if self.context_menu:

            # if the mouse has moved outside the context menu bbox, it will
            # have definitely been dismissed
            x, y = pyautogui.position()
            if not self.context_menu.is_inside(x, y):
                self.context_menu = None
                return

            # TODO: check if it has timed out

    def update(self):
        """
        Run update methods for click timeouts and context menu.
        This method should be called once per loop only.
        """

        self.update_click_timeouts()
        self.update_context_menu()

        # add deferred draw call to client
        self.client.add_draw_call(self.draw)

    def draw(self):
        """TODO"""

        # game objects are not guaranteed a bound box
        if not self.get_bbox():
            return

        if 'go_bbox' in self.client.args.show:
            cx1, cy1, _, _ = self.client.get_bbox()

            x1, y1, x2, y2 = self.get_bbox()
            # TODO: method to determine if entity is on screen (and not
            #  obstructed)
            if self.client.is_inside(x1, y1) and self.client.is_inside(x2, y2):

                # convert local to client image
                x1, y1, x2, y2 = self.client.localise(x1, y1, x2, y2)

                # draw a rect around entity on main screen
                cv2.rectangle(
                    self.client.original_img, (x1, y1), (x2, y2),
                    self.colour, 1)

    def _draw_bounding_box(self, bbox: callable):
        cx1, cy1, _, _ = self.client.get_bbox()
        x1, y1, x2, y2 = bbox()
        if self.client.is_inside(x1, y1) and self.client.is_inside(x2, y2):
            # convert local to client image
            x1, y1, x2, y2 = self.client.localise(x1, y1, x2, y2)

            # draw a rect around entity on main screen
            cv2.rectangle(
                self.client.original_img, (x1, y1), (x2, y2),
                self.colour, 1)

    def draw_bbox(self):
        self._draw_bounding_box(self.get_bbox)

    def draw_click_box(self):
        self._draw_bounding_box(self.click_box)

    def resolve_path(self, **kwargs):
        return self.PATH_TEMPLATE.format(**kwargs)

    @property
    def templates(self):
        return self._templates

    @property
    def masks(self):
        return self._masks

    def load_templates(self, names=None, cache=True, force_reload=False):
        """
        Load template data from disk
        :param Iterable[str] names: Names to attempt to load from disk
        :param cache: Optionally cache the loaded templates to internal var
        :param bool force_reload: If true templates will be loaded from disk
            even if they are already cached.

        :return: Dictionary of templates of format {<name>: <numpy array>}
        """
        templates = dict()

        names = names or list()
        if not names:
            # if we don't specify any names, don't load anything
            return templates

        for name in names:

            if name in self._templates and not force_reload:
                continue

            path = self.resolve_path(
                root=get_root(),
                name=name
            )
            if exists(path):
                template = numpy.load(path)
                templates[name] = template
            else:
                self.logger.debug(f'Bad path: {path}')

        self.logger.debug(f'Loaded templates: {templates.keys()}')

        if cache:
            self._templates.update(**templates)
            return self._templates
        return templates

    def remove_templates(self, *names, masks=True):
        """Remove a template from internal data.

        :param str names: List of template names to remove. If '*' is provided,
            all templates (and optionally masks) will be removed.
        :param bool masks: Optionally remove masks that correspond to named
            templates.
        """

        if '*' in names:
            self._templates = {}
            if masks:
                self._masks = {}
            return

        for name in names:
            try:
                self._templates.pop(name)
                if masks:
                    try:
                        self._masks.pop(name)
                    except KeyError:
                        self.logger.debug(
                            f'Cannot remove mask with name: {name}')
            except KeyError:
                self.logger.debug(f'Cannot remove template with name: {name}')

            if name in self._templates:
                self._templates.pop(name)

    def load_masks(self, names=None, cache=True, force_reload=False):
        """
        Load template masks into a dictionary of the same structure as
        :meth:`GameObject.load_templates`. Masks are assumed to have the same
        name as the templates with '_mask' appended.
        """

        masks = dict()
        names = names or list()
        if not names:
            return masks

        for name in names:

            if name in self._masks and not force_reload:
                continue

            path = self.resolve_path(
                root=get_root(),
                name=name+'_mask'
            )
            if exists(path):
                mask = numpy.load(path)
                masks[name] = mask

        if cache:
            self._masks.update(**masks)
            return self._masks
        return masks

    def alias_mask(self, name, alias):
        """Create a copy of a mask under a different name.

        Sometimes multiple objects have the exact same profile, so rather than
        loading multiple masks it's more efficient to load the mask once and
        make a copy as needed.

        :param str name: Name of the mask we're going to duplicate.
        :param str alias: New name for the mask.

        :raises TypeError: If the mask name doesn't exist.

        """

        mask = self._masks.get(name)
        if mask is None:
            raise ValueError(f'No mask named: {name}')

        aliased_mask = mask.copy()
        self._masks[alias] = aliased_mask

    def process_img(self, img):
        """
        Process raw image from screen grab into a format ready for template
        matching.
        :param img: BGRA image section for current slot
        :return: GRAY scaled image
        """
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        return img_gray

    def contains_colour(self, colour):
        img = self.client.get_img_at(
            self.get_bbox(), mode=self.client.HSV)
        lower = COLOUR_DICT_HSV[colour][1]
        upper = COLOUR_DICT_HSV[colour][0]
        img = cv2.inRange(img, lower, upper)
        return img.any()

    def identify(self, threshold=0.8):
        """
        Compare incoming image with templates and try to find a match.

        Can be configured to do an exact match - `single_match` (i.e. the
        object's image and the templates are the same size) or a rough match
        where the template is smaller than the image and the first match found
        is returned.

        Matching can also be configured with `invert` to invert both template
        and image before matching (which is useful if the template has a lot
        of black in it), `match_method` can be used to choose between
        CCOEFF_NORMED and SQDIFF_NORMED.

        :param float threshold: Percentage match against which templates can
            be accepted. 1 means a strong match, 0 is no match.

        :returns str: Name of the matched template, if any.

        """

        if not self.templates:
            print(f'{self}: No templates loaded, cannot identify')
            return False

        max_match = -float('inf')
        if self.match_method == cv2.TM_SQDIFF_NORMED:
            threshold = 1 - threshold
            max_match = float('inf')

        matched_item = None
        for name, template in self.templates.items():
            mask = self.masks.get(name)

            img = self.img
            if self.match_invert:
                template = cv2.bitwise_not(template)
                img = cv2.bitwise_not(self.img)

            if self.single_match:

                match = cv2.matchTemplate(
                    img, template, self.match_method, mask=mask)[0][0]

                condition = (
                    (self.match_method == cv2.TM_CCOEFF_NORMED and
                     match > max_match)
                    or (self.match_method == cv2.TM_SQDIFF_NORMED and
                        match < max_match)
                )
                if condition:
                    max_match = match
                    matched_item = name

            else:
                matches = cv2.matchTemplate(
                    img, template, self.match_method, mask=mask)

                if self.match_method == cv2.TM_CCOEFF_NORMED:
                    (my, mx) = numpy.where(matches >= threshold)
                elif self.match_method == cv2.TM_SQDIFF_NORMED:
                    (my, mx) = numpy.where(matches <= threshold)
                else:
                    my, mx = [], []

                for y, x in zip(my, mx):
                    self.client.logger.info(f'found match {name} at: {x, y}')
                    return name

        condition = (
            (self.match_method == cv2.TM_CCOEFF_NORMED and
             max_match >= threshold)
            or (self.match_method == cv2.TM_SQDIFF_NORMED and
                max_match <= threshold)
        )
        if condition:
            contents = matched_item
        else:
            contents = None

        return contents

    def set_context_menu(self, x, y, width, items, config):
        menu = ContextMenu(
            self.client, self, x, y, width, items, config)
        self.context_menu = menu
        return menu

    @property
    def clicked(self):
        return self._clicked

    @property
    def clickable(self):
        """
        Determines if the game object is currently a valid target for clicking.
        Default method always returns True, should be overloaded by subclasses.
        """
        return True

    def _click(self, tmin=None, tmax=None, bbox=None, **kwargs):
        """
        :param tmin: Timeout minimum
        :param tmax: Timeout max
        :param bbox: Bounding box to click. If not set, the current game
            object's get_bbox method will be used. You can pass in
            a bounding box method to use instead. You can also pass in the
            boolean False and the current mouse position will be used (i.e.
            don't uniformly distribute inside a bbox - don't move at all).
        """

        if not self.clickable:
            return

        if bbox is False:
            x, y = self.client.screen.mouse_xy
            x, y = self.client.screen.click_aoi(x, y, x, y, **kwargs)
        elif bbox:
            x, y = self.client.screen.click_aoi(*bbox, **kwargs)
        else:
            x, y = self.client.screen.click_aoi(
                *self.get_bbox(),
                **kwargs
            )

        tmin = tmin or 1
        tmax = tmax or 3

        offset = self.client.screen.map_between(random.random(), tmin, tmax)
        self.add_timeout(offset)

        return x, y

    def click(self, tmin=None, tmax=None, speed=1, pause_before_click=False,
              shift=False, bbox=None, multi=1):
        return self._click(
            tmin=tmin, tmax=tmax,
            speed=speed, pause_before_click=pause_before_click,
            shift=shift, bbox=bbox, multi=multi,
        )

    def right_click(self, tmin=None, tmax=None, speed=1,
                    pause_before_click=False, bbox=None):
        return self._click(
            tmin=tmin, tmax=tmax,
            speed=speed, pause_before_click=pause_before_click,
            right=True, click=False, bbox=bbox,
        )

    def add_timeout(self, offset):
        self._clicked.append(Timeout(self.client, offset))

    def clear_timeout(self):
        """
        Clears all timeouts on current game object, regardless if they have
        expired or not
        :return: True if all timeouts cleared
        """
        self._clicked = list()
        return not self._clicked

    @property
    def time_left(self):

        try:
            time_left = max([c.time_left for c in self._clicked])
            return round(time_left, 2)
        except ValueError:
            # if there are no timeouts at all, max will raise ValueError,
            # which means we have no time left - return zero
            return 0

    def is_inside(self, x, y, method=None,
                  offset: Union[int, Dict[str, int], Tuple, None] = None):
        """
        Check if the provided coordinates are inside the current object's
        bounding box. X and Y coordinates must be global to the screen.

        :param int x: x coordinate to check
        :param int y: y coordinate to check
        :param func method: Optionally provide an alternative method to
            determine the bounding box. By default it will use the current
            object's get_bbox method, but some objects (e.g. game_screen
            objects) have bounding boxes both on screen and on the minimap,
            so they can be supplied here.
        :param Union[int, dict, tuple, None] offset: Optionally provide an
            offset for the bounding box. Offsets are applied left to right,
            top to bottom. So negative offset provides padding
            (meaning the bounding box is larger) on left and top; or x1, y1,
            but a margin (so bounding box is smaller) on the bottom and left;
            or x2. y2. Integer offset will be applied to all four sides, or you
            can provide finer control with tuple or dict format. Tuple must be
            in order <x1, y1, x2, y2>, Dict must have keys matching "x1" etc.

        :raises TypeError: If offset is not valid type

        :returns: True if coordinate inside.
        :rtype: bool

        """

        if method is None:
            method = self.get_bbox

        offset_x1 = 0
        offset_x2 = 0
        offset_y1 = 0
        offset_y2 = 0
        if offset:
            if isinstance(offset, int):
                offset_x1 = offset_x2 = offset_y1 = offset_y2 = offset
            elif isinstance(offset, tuple):
                offset_x1, offset_y1, offset_x2, offset_y2 = offset
            elif isinstance(offset, dict):
                offset_x1 = offset.get('x1', 0)
                offset_x2 = offset.get('x2', 0)
                offset_y1 = offset.get('y1', 0)
                offset_y2 = offset.get('y2', 0)
            else:
                raise TypeError(f'Unsupported offset type: {type(offset)}')

        x1, y1, x2, y2 = method()
        x1 += offset_x1
        x2 += offset_x2
        y1 += offset_y1
        y2 += offset_y2

        return x1 <= x <= x2 and y1 <= y <= y2


# TODO: Refactor this class to 'RightClickMenu' as it's more understandable
class ContextMenu(GameObject):

    ITEM_HEIGHT = 15
    OCR_READ_ITEMS = False

    def __init__(
            self,
            client: "wizard_eyes.client.Client",
            parent: GameObject,
            x, y, width, items, config
    ):
        super(ContextMenu, self).__init__(client, parent)

        self.x = x
        self.y = y
        self._width = width
        self.items = [ContextMenuItem(client, self, i) for i in range(items)]
        self.config = config

    @property
    def width(self):
        return self._width

    @property
    def height(self):

        header = self.config['margins']['top']
        footer = self.config['margins']['bottom']
        items = len(self.items) * self.ITEM_HEIGHT

        return header + items + footer

    def get_bbox(self):

        x1 = int(self.x - self.width / 2) - 1
        y1 = self.y

        x2 = int(self.x + self.width / 2)
        y2 = self.y + self.height

        return x1, y1, x2, y2

    def update(self):

        # moving the mouse outside context box destroys it
        if not self.is_inside(*self.client.screen.mouse_xy):
            self.parent.context_menu = None
            return

        super().update()

        for item in self.items:
            item.update()


class ContextMenuItem(GameObject):

    def __init__(self,
                 client: "wizard_eyes.client.Client",
                 parent: ContextMenu,
                 idx: int,
    ):
        super(ContextMenuItem, self).__init__(client, parent)
        self.idx = idx
        self.value = None
        self.value_changed_at = -float('inf')

    def get_bbox(self):
        header = self.parent.config['margins']['top']

        m_left = self.parent.config['margins']['left']
        m_right = self.parent.config['margins']['right']

        x1 = int(self.parent.x - self.parent.width / 2) + m_left
        y1 = self.parent.y + header + self.parent.ITEM_HEIGHT * self.idx

        x2 = int(self.parent.x + self.parent.width / 2) - m_right
        y2 = y1 + self.parent.ITEM_HEIGHT - 1

        return x1, y1, x2, y2

    def click(self, *args, **kwargs):
        super().click(*args, **kwargs)

        # clicking a context menu item destroys the context menu
        self.parent.parent.context_menu = None

    def update(self):
        super().update()

        if self.parent.OCR_READ_ITEMS:
            img = Image.fromarray(self.img)
            self.client.ocr.SetImage(img)
            value = self.client.ocr.GetUTF8Text()
            value = value.strip().replace('\n', '').replace('\r', '').lower()

            if self.value != value:
                self.value_changed_at = self.client.time

            self.value = value
