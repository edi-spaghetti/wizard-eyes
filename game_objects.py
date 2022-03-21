import time
import random
import ctypes
import logging
from os.path import dirname, exists, basename
from glob import glob
from collections import defaultdict

import numpy
import cv2
import pyautogui

# TODO: use scale factor and determine current screen to apply to any config
#       values. For the time being I'm setting system scaling factor to 100%
scale_factor = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100

# TODO: create constants file
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FILL = -1


class Timeout(object):

    def __init__(self, offset):
        self.created_at = time.time()
        self.offset = self.created_at + offset


class GameObject(object):

    PATH_TEMPLATE = '{root}/data/{name}.npy'

    # default colour for showing client image (note: BGRA)
    DEFAULT_COLOUR = (0, 0, 0, 255)

    def __init__(self, client, parent, config_path=None, container_name=None,
                 template_names=None, logging_level=None):
        self._logging_level = logging_level
        self.logger = self.setup_logger()

        self.client = client
        self.parent = parent
        self.context_menu = None
        self._bbox = None
        self.config = self._get_config(config_path)
        self.container_name = container_name
        self.containers = self.setup_containers()

        self.colour = self.DEFAULT_COLOUR

        self._templates = dict()
        self._masks = dict()
        self.load_templates(names=template_names)
        self.load_masks(names=template_names)

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
        x1, y1, x2, y2 = self.get_bbox()
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

    def update_click_timeouts(self):
        """
        Check and remove timeouts that have expired.
        Note, it uses client time, so ensure the client has been updated.
        """
        i = 0
        while i < len(self._clicked):
            t = self._clicked[i]
            if self.client.time > t.offset:
                self._clicked = self._clicked[:i]
                break

            i += 1

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

    def resolve_path(self, **kwargs):
        return self.PATH_TEMPLATE.format(**kwargs)

    @property
    def templates(self):
        return self._templates

    @property
    def masks(self):
        return self._masks

    def load_templates(self, names=None, cache=True):
        """
        Load template data from disk
        :param list names: Names to attempt to load from disk
        :param cache: Optionally cache the loaded templates to internal var
        :return: Dictionary of templates of format {<name>: <numpy array>}
        """
        templates = dict()

        names = names or list()
        if not names:
            # if we don't specify any names, don't load anything
            return templates

        for name in names:
            path = self.resolve_path(
                root=dirname(__file__),
                name=name
            )
            if exists(path):
                template = numpy.load(path)
                templates[name] = template
            else:
                self.logger.warning(f'Bad path: {path}')

        self.logger.debug(f'Loaded templates: {templates.keys()}')

        if cache:
            self._templates = templates
        return templates

    def load_masks(self, names=None, cache=True):
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
            path = self.resolve_path(
                root=dirname(__file__),
                name=name+'_mask'
            )
            if exists(path):
                mask = numpy.load(path)
                masks[name] = mask

        if cache:
            self._masks = masks
        return masks

    def process_img(self, img):
        """
        Process raw image from screen grab into a format ready for template
        matching.
        :param img: BGRA image section for current slot
        :return: GRAY scaled image
        """
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        return img_gray

    def identify(self, threshold=None):
        """
        Compare incoming image with templates and try to find a match
        :param float threshold: Percentage match against which templates can
            be accepted.
        """

        if not self.templates:
            print(f'{self}: No templates loaded, cannot identify')
            return False

        max_match = None
        matched_item = None
        for name, template in self.templates.items():
            match = cv2.matchTemplate(
                self.img, template, cv2.TM_CCOEFF_NORMED)[0][0]

            if max_match is None:
                max_match = match
                matched_item = name
            elif match > max_match:
                max_match = match
                matched_item = name

        threshold = threshold or 0.8
        if max_match and max_match > threshold:
            contents = matched_item
        else:
            contents = None

        return contents

    def set_context_menu(self, x, y, width, items, config):
        menu = ContextMenu(
            self.client, self.parent, x, y, width, items, config)
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

        if not self.clickable:
            return

        if bbox:
            x, y = self.client.screen.click_aoi(*bbox, **kwargs)
        else:
            x, y = self.client.screen.click_aoi(
                *self.get_bbox(),
                **kwargs
            )

        tmin = tmin or 1
        tmax = tmax or 3

        offset = self.client.screen.map_between(random.random(), tmin, tmax)
        self._clicked.append(Timeout(offset))

        return x, y

    def click(self, tmin=None, tmax=None, speed=1, pause_before_click=False,
              shift=False, bbox=None):
        return self._click(
            tmin=tmin, tmax=tmax,
            speed=speed, pause_before_click=pause_before_click,
            shift=shift, bbox=bbox,
        )

    def right_click(self, tmin=None, tmax=None, speed=1,
                    pause_before_click=False, bbox=None):
        return self._click(
            tmin=tmin, tmax=tmax,
            speed=speed, pause_before_click=pause_before_click,
            right=True, click=False, bbox=None,
        )

    def add_timeout(self, offset):
        self._clicked.append(Timeout(offset))

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
            time_left = min([c.offset for c in self._clicked])
            time_left = time_left - self.client.time
            return round(time_left, 2)
        except ValueError:
            return 0

    def is_inside(self, x, y):
        x1, y1, x2, y2 = self.get_bbox()
        return x1 <= x <= x2 and y1 <= y <= y2


# TODO: Refactor this class to 'RightClickMenu' as it's more understandable
class ContextMenu(GameObject):

    ITEM_HEIGHT = 15

    def __init__(self, client, parent, x, y, width, items, config):
        super(ContextMenu, self).__init__(client, parent)

        self.x = x
        self.y = y
        # TODO: fix this bug
        self.width = width
        self.items = [ContextMenuItem(client, self, i) for i in range(items)]
        self.config = config

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


class ContextMenuItem(GameObject):

    def __init__(self, client, parent, idx):
        super(ContextMenuItem, self).__init__(client, parent)
        self.idx = idx

    def get_bbox(self):
        header = self.parent.config['margins']['top']

        m_left = self.parent.config['margins']['left']
        m_right = self.parent.config['margins']['right']

        x1 = int(self.parent.x - self.parent.width / 2) + m_left
        y1 = self.parent.y + header + self.parent.ITEM_HEIGHT * self.idx

        x2 = int(self.parent.x + self.parent.width / 2) - m_right
        y2 = y1 + self.parent.ITEM_HEIGHT - 1

        return x1, y1, x2, y2


class Tabs(GameObject):
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
        'influence': ['quests']
    }

    def __init__(self, client):

        # set up templates with defaults & selected modifiers
        template_names = (
                self.STATIC_TABS +
                [f'{t}_selected' for t in self.STATIC_TABS])
        for name, types in self.MUTABLE_TABS.items():
            for type_ in types:
                template = f'{name}_{type_}'
                selected = f'{name}_{type_}_selected'
                template_names.append(template)
                template_names.append(selected)

        super(Tabs, self).__init__(
            client, client, config_path='tabs',
            container_name='personal_menu',
            template_names=template_names,
        )

        # load in the default tab mask
        self.load_masks(['tab'], cache=True)

        # dynamically build tab items based on what can be found
        self.active_tab = None
        self._tabs = None

        # add in placeholders for the tabs we expect to find (this will
        # helper the linter)
        # TODO: handle mutable tabs e.g. quests/achievement diary or spellbooks
        self.combat = None
        self.stats = None
        self.equipment = None
        self.quests = None
        self.inventory = None
        self.prayer = None
        self.spellbook_arceuus = None

    @property
    def width(self):
        # TODO: double tab stack if client width below threshold
        return self.config['width'] * 13

    @property
    def height(self):
        # TODO: double tab stack if client width below threshold
        return self.config['height'] * 1

    def build_tab_items(self):
        """
        Dynamically generate tab items based on what can be detected by
        template matching. Must be run after init (see client init) because it
        uses the container system from config, which is not available at init.
        """

        items = dict()
        cx1, cy1, cx2, cy2 = self.get_bbox()

        # TODO: tabs may be unavailable e.g. were're on the login screen, or
        #  we're on tutorial island and some tabs are disabled.

        # TODO: add key bindings, so tabs can be opened/closed with F-keys
        #  (or RuneLite key bindings)

        tabs = list()
        for tab in self.STATIC_TABS:
            templates = list()

            template = self.templates.get(tab)
            if template is None:
                continue
            templates.append((tab, template))

            name = f'{tab}_selected'
            selected = self.templates.get(name)
            if selected is None:
                continue
            templates.append((name, selected))

            tabs.append((tab, templates))
        for group, names in self.MUTABLE_TABS.items():

            templates = list()
            for tab in names:
                tab = f'{group}_{tab}'
                template = self.templates.get(tab)

                if template is None:
                    continue
                templates.append((tab, template))

                name = f'{tab}_selected'
                selected = self.templates.get(name)
                if selected is None:
                    continue
                templates.append((name, selected))

            tabs.append((group, templates))

        for tab, templates in tabs:

            cur_confidence = -float('inf')
            cur_x = cur_y = cur_h = cur_w = None
            cur_template_name = ''
            confidences = list()
            for template_name, template in templates:
                match = cv2.matchTemplate(
                    self.img, template, cv2.TM_CCOEFF_NORMED,
                    mask=self.masks.get('tab'),
                )
                _, confidence, _, (x, y) = cv2.minMaxLoc(match)

                # log confidence for later
                confidences.append(f'{template_name}: {confidence:.3f}')

                if confidence > cur_confidence:
                    cur_confidence = confidence
                    cur_x = x
                    cur_y = y
                    cur_h, cur_w = template.shape
                    cur_template_name = template_name

            selected = cur_template_name.endswith('selected')

            if None in {cur_x, cur_y, cur_h, cur_w}:
                continue

            self.logger.info(
                f'{tab}: '
                f'chosen: {cur_template_name}, '
                f'selected: {selected}, '
                f'confidence: {confidences}'
            )

            x1, y1, x2, y2 = cur_x, cur_y, cur_x + cur_w - 1, cur_y + cur_h - 1
            # convert back to screen space so we can set global bbox
            sx1 = x1 + cx1 - 1
            sy1 = y1 + cy1 - 1
            sx2 = x2 + cx1 - 1
            sy2 = y2 + cy1 - 1

            # create dynamic tab item
            item = TabItem(tab, self.client, self, selected=selected)
            item.set_aoi(sx1, sy1, sx2, sy2)
            item.load_templates([t for t, _ in templates])
            item.load_masks(['tab'])

            # cache it to dict and add as class attribute for named access
            items[tab] = item
            setattr(self, tab, item)

            if selected:
                self.active_tab = item

        self._tabs = items
        return items

    def _click(self, *args, **kwargs):
        self.logger.warning('Do not click container, click the tabs.')

    def update(self):
        """
        Run update on each of the tab items.
        Note, it does not update click timeouts, as this class should not be
        clicked directly (attempting to do so throws a warning).
        """

        for tab in self._tabs.values():
            tab.update()

            if tab.selected:
                self.active_tab = tab


class TabItem(GameObject):

    PATH_TEMPLATE = '{root}/data/tabs/{name}.npy'

    def __str__(self):
        return f'TabItem<{self.name}>'

    def __repr__(self):
        return f'TabItem<{self.name}>'

    def __init__(self, name, *args, selected=False, **kwargs):
        super(TabItem, self).__init__(*args, **kwargs)
        self.name = name
        self.selected = selected
        self.interface = TabInterface(self.client, self)

    @property
    def img(self):
        img = super(TabItem, self).img

        # draw an extra 1 pixel sized backboard so masking doesn't fail
        # (seems to be a bug if template is same size as image)
        y, x = img.shape
        img2 = numpy.zeros((y+1, x), dtype=numpy.uint8)
        img2[:y, :x] = img

        return img2

    def update(self):
        """
        Run standard click timeout updates, then check the templates to see
        if the tab is currently selected or not.
        """

        super(TabItem, self).update()

        # TODO: there is actually a third state where tabs are disabled (e.g.
        #  during cutscenes, on tutorial island etc.)

        cur_confidence = -float('inf')
        cur_x = cur_y = cur_h = cur_w = None
        cur_template_name = ''
        confidences = list()
        for template_name, template in self.templates.items():
            match = cv2.matchTemplate(
                self.img, template, cv2.TM_CCOEFF_NORMED,
                mask=self.masks.get('tab'),
            )
            _, confidence, _, (x, y) = cv2.minMaxLoc(match)

            # log confidence for later
            confidences.append(f'{template_name}: {confidence:.3f}')

            if confidence > cur_confidence:
                cur_confidence = confidence
                cur_x = x
                cur_y = y
                cur_h, cur_w = template.shape
                cur_template_name = template_name

        selected = cur_template_name.endswith('selected')

        self.logger.debug(
            f'{self.name}: '
            f'chosen: {cur_template_name}, '
            f'selected: {selected}, '
            f'confidence: {confidences}'
        )

        self.selected = selected

        # TODO: convert to base class method
        if f'{self.name}_bbox' in self.client.args.show and selected:
            cx1, cy1, _, _ = self.client.get_bbox()
            x1, y1, x2, y2 = self.get_bbox()
            if self.client.is_inside(x1, y1) and self.client.is_inside(x2, y2):
                # convert local to client image
                x1, y1, x2, y2 = self.client.localise(x1, y1, x2, y2)

                # draw a rect around entity on main screen
                cv2.rectangle(
                    self.client.original_img, (x1, y1), (x2, y2),
                    self.colour, 1)

        # recursively call the icons on the interface
        self.interface.update()


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
                    # TODO: find out why mask of same size causes error
                    # mask=mask,
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

                    icon = InterfaceIcon(name, self.client, self)
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

    def update(self):
        """
        Run update on each of the icons (if the tab is selected - and the
        interface therefore open)
        Note, it does not update click timeouts, as this class should not be
        clicked directly (attempting to do so throws a warning).
        """

        if self.parent_tab.selected:
            for icon in self.icons.values():
                icon.update()


class InterfaceIcon(GameObject):
    """
    Class to represent icons/buttons/items etc. dynamically generated in
    an instance of :class:`TabInterface`.
    """

    PATH_TEMPLATE = '{root}/data/tabs/{name}.npy'

    def __init__(self, name, *args, **kwargs):
        super(InterfaceIcon, self).__init__(*args, **kwargs)
        self.name = name
        self.confidence = None
        self.state = None

    def update(self):
        """
        Run the standard click timer updates, then run template matching to
        determine the current state of the icon. Usually it will have a
        different appearance if activated/clicked.
        """
        super(InterfaceIcon, self).update()

        cur_confidence = -float('inf')
        cur_state = None
        for state, template in self.templates.items():
            mask = self.masks.get(state)
            match = cv2.matchTemplate(
                self.img, template, cv2.TM_CCOEFF_NORMED,
                # TODO: find out why mask of same size causes error
                # mask=mask,
            )
            _, confidence, _, _ = cv2.minMaxLoc(match)

            if confidence > cur_confidence:
                cur_state = state
                cur_confidence = confidence

        self.confidence = cur_confidence
        self.state = cur_state

        # TODO: convert to base class method
        if f'{self.name}_bbox' in self.client.args.show:
            cx1, cy1, _, _ = self.client.get_bbox()
            x1, y1, x2, y2 = self.get_bbox()
            if self.client.is_inside(x1, y1) and self.client.is_inside(x2, y2):
                # convert local to client image
                x1, y1, x2, y2 = self.client.localise(x1, y1, x2, y2)

                # draw a rect around entity on main screen
                cv2.rectangle(
                    self.client.original_img, (x1, y1), (x2, y2),
                    self.colour, 1)


class Dialog(object):

    def __init__(self, client):
        self._client = client
        self.config = client.config['dialog']
        self.makes = list()

    @property
    def width(self):
        return self.config['width']

    @property
    def height(self):
        return self.config['height']

    def add_make(self, name):
        make = DialogMake(self._client, self, name, len(self.makes))
        self.makes.append(make)

        return make

    def get_bbox(self):
        if self._client.name == 'RuneLite':
            cx1, cy1, cx2, cy2 = self._client.get_bbox()

            cl_margin = self._client.config['margins']['left']
            cb_margin = self._client.config['margins']['bottom']

            x1 = cx1 + cl_margin
            y1 = cy2 - cb_margin - self.height

            x2 = x1 + self.width - 1
            y2 = cy2 - cb_margin

        else:
            raise NotImplementedError

        return x1, y1, x2, y2


class DialogMake(object):

    PATH_TEMPLATE = '{root}/data/dialog/make/{name}.npy'

    def __init__(self, client, dialog, name, index):
        self._client = client
        self.dialog = dialog
        self.config = dialog.config['makes']
        self.name = name
        self.index = index
        self.template = self.load_template(name)
        self._bbox = None

    def load_template(self, name):
        path = self.PATH_TEMPLATE.format(
            root=dirname(__file__),
            name=name
        )
        if exists(path):
            return numpy.load(path)

    @property
    def width(self):
        # TODO: scaling make buttons
        if len(self.dialog.makes) <= 4:
            return self.config['max_width']

    @property
    def height(self):
        return self.config['height']

    def get_bbox(self):

        if self._bbox:
            return self._bbox

        if self._client.name == 'RuneLite':

            cx1, cy1, cx2, cy2 = self._client.get_bbox()

            cl_margin = self._client.config['margins']['left']
            cb_margin = self._client.config['margins']['bottom']

            # TODO: multiple make buttons
            padding_left = int(self.dialog.width / 2 - self.width / 2)
            padding_bottom = self.config['padding']['bottom']

            dialog_tabs_height = 23  # TODO: self.dialog.tabs.height

            x1 = cx1 + cl_margin + padding_left
            y1 = cy2 - cb_margin - dialog_tabs_height - padding_bottom - self.height

            x2 = x1 + self.width
            y2 = y1 + self.height

        else:
            raise NotImplementedError

        # cache bbox for performance
        self._bbox = x1, y1, x2, y2

        return x1, y1, x2, y2

    def process_img(self, img):
        """
        Process raw image from screen grab into a format ready for template
        matching.
        TODO: build base class so we don't have to duplicate this
        :param img: BGRA image section for current slot
        :return: GRAY scaled image
        """
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        return img_gray

    def identify(self, img):
        """
        Determine if the deposit inventory button is visible on screen
        :param img: Screen grab subsection where button is expected to be
        :return: True if matched, else False
        """

        if self.template is None:
            return False

        x, y, _, _ = self._client.get_bbox()
        x1, y1, x2, y2 = self.get_bbox()
        # numpy arrays are stored rows x columns, so flip x and y
        img = img[y1 - y:y2 - y, x1 - x:x2 - x]

        img = self.process_img(img)
        result = cv2.matchTemplate(img, self.template, cv2.TM_CCOEFF_NORMED)
        match = result[0][0]

        threshold = 0.8
        return match > threshold


class Bank(GameObject):

    def __init__(self, client):
        super(Bank, self).__init__(
            client, client,
            config_path='bank',
        )
        self._bbox = None
        self.utilities = BankUtilities(self.client, self)
        self.tabs = BankTabContainer(self.client, self)
        self.close = CloseBank(self.client, self)

    @property
    def width(self):
        return self.config['width']

    @property
    def height(self):

        ct_margin = self.client.config['margins']['top']
        cb_margin = self.client.config['margins']['bottom']

        dialog_height = self.client.config['dialog']['height']
        banner_height = self.client.config['banner']['height']

        padding_top = self.config['padding']['top']
        padding_bottom = self.config['padding']['bottom']

        # remove each item from client height in order
        height = self.client.height

        # in order from top to bottom remove each item's height
        height -= ct_margin
        height -= banner_height
        height -= padding_top
        # don't remove bank window - that's what we're calculating!
        height -= padding_bottom
        height -= dialog_height
        height -= cb_margin

        # remainder is the height of the bank window
        return height

    def get_bbox(self):

        if self._bbox:
            return self._bbox

        if self.client.name == 'RuneLite':

            cx1, cy1, cx2, cy2 = self.client.get_bbox()
            cli_min_width = self.client.config['min_width']

            banner_height = self.client.config['banner']['height']

            cl_margin = self.client.config['margins']['left']
            ct_margin = self.client.config['margins']['top']
            cb_margin = self.client.config['margins']['bottom']

            dialog_height = self.client.config['dialog']['height']

            padding_left = self.config['padding']['min_left']
            padding_left += int((self.client.width - cli_min_width) / 2)
            padding_top = self.config['padding']['top']
            padding_bottom = self.config['padding']['bottom']

            x1 = cx1 + cl_margin + padding_left
            y1 = cy1 + ct_margin + banner_height + padding_top
            x2 = x1 + self.width
            y2 = cy2 - cb_margin - dialog_height - padding_bottom - 1

        else:
            raise NotImplementedError

        # cache bbox for performance
        self._bbox = x1, y1, x2, y2

        return x1, y1, x2, y2


class CloseBank(GameObject):

    HOTKEY = 'esc'

    def __init__(self, client, parent):
        super(CloseBank, self).__init__(client, parent)

    def click(self, tmin=None, tmax=None):

        self.client.screen.press_key(self.HOTKEY)
        tmin = tmin or 1
        tmax = tmax or 3
        offset = self.client.screen.map_between(random.random(), tmin, tmax)
        self._clicked.append(Timeout(offset))

        return self.HOTKEY

    def right_click(self, tmin=None, tmax=None):
        raise NotImplementedError

    def is_inside(self, x, y):
        raise NotImplementedError


class BankTabContainer(object):

    MAX_TABS = 9

    def __init__(self, client, bank):
        self._client = client
        self.bank = bank
        self.tabs = self._create_tabs()
        self.active_index = 0

    @property
    def active(self):
        """
        Convenience method to pull the currently active tab
        :return: Currently active tab
        :rtype: BankTab
        """
        return self.tabs[self.active_index]

    def _create_tabs(self):
        tabs = list()

        for _ in range(self.MAX_TABS):
            tabs.append(None)

        return tabs

    def set_tab(self, idx, is_open=False):
        """
        Set up a new bank tab object at the provided index
        :param idx: Index of the tab within the container
        :param is_open: Optionally set open state of new tab
        :return: new BankTab object
        """
        tab = BankTab(idx, self._client, self)
        self.tabs[idx] = tab

        # optionally set newly created tab as open, making sure all others
        # are closed
        self.set_open(idx, is_open)

        return tab

    def set_open(self, idx, value):
        """
        Sets the target tab as open, and updates some bookkeeping variables
        :param idx: Index of the tab to update
        :param value: Any value that evaluates to True or False with bool()
        :return: True if setting was successful, else False
        """

        if self.tabs[idx] is None:
            return False

        self.active_index = idx
        self.tabs[idx].is_open = bool(value)

        # if we set the target tab as open, make sure all others are then
        # marked as closed. Note, it's possible (though not currently
        # supported) to close all tabs e.g. using runelite tags
        if self.tabs[idx].is_open:
            map(lambda t: t.set_open(False), filter(
                    lambda u: u.idx != idx and u is not None,
                    self.tabs
            ))

        return True


class BankTab(object):

    SLOTS_HORIZONTAL = 8
    MAX_SLOTS = 150  # TODO: dynamic max based on client height

    def __init__(self, idx, client, container):
        self.idx = idx
        self._client = client
        self.container = container
        self.config = container.bank.config['tabs']
        self.slots = self._create_slots()
        self.is_open = False

    def _create_slots(self):
        slots = list()

        for _ in range(self.MAX_SLOTS):
            slots.append(None)

        return slots

    def set_open(self, value):
        # use parent method to ensure container data is consistent
        self.container.set_open(self.idx, value)

    def set_slot(self, idx, template_names):
        """
        Setup a bank slot object at provided index with provided template names
        :param idx: Index for the new slot
        :type idx: int
        :param template_names: List of template names the slot should load
        :return: new BankSlot object
        """
        slot = BankSlot(idx, self._client, self, template_names)
        self.slots[idx] = slot

        return slot

    def identify(self, img, threshold=None):
        """
        Runs identification on each slot in the bank tab
        :param img: Screen grab of the whole client
        :return: List of items identified
        """

        # TODO: check bank window is also open

        # check that current tab is actually open
        if self.idx != self.container.active_index:
            return list()

        # we need client bbox to zero the slot coordinates
        x, y, _, _ = self._client.get_bbox()

        items = list()
        for slot in self.slots:

            # skip any slots that haven't been set
            if slot is None:
                items.append(None)
                continue

            x1, y1, x2, y2 = slot.get_bbox()
            # numpy arrays are stored rows x columns, so flip x and y
            slot_img = img[y1 - y:y2 - y, x1 - x:x2 - x]

            name = slot.identify(slot_img, threshold)
            items.append(name)

        return items


class BankSlot(GameObject):

    PATH_TEMPLATE = '{root}/data/bank/slots/{tab}/{index}/{name}.npy'

    def __init__(self, idx, client, parent, template_names):
        self.idx = idx
        super(BankSlot, self).__init__(
            client, parent,
            config_path='bank.tabs.slots',
            template_names=template_names,
        )

        self._bbox = None
        self.contents = None

    def load_templates(self, names=None, cache=True):
        """
        Load template data from disk
        :param names: List of names to attempt to load from disk
        :type names: list
        :return: Dictionary of templates of format {<name>: <numpy array>}
        """
        templates = dict()

        names = names or list()
        if not names:
            glob_path = self.PATH_TEMPLATE.format(
                root=dirname(__file__),
                tab=self.parent.idx,
                index=self.idx,
                name='*'
            )

            # print(f'{self.idx} GLOB PATH = {glob_path}')

            paths = glob(glob_path)
            names = [basename(p).replace('.npy', '') for p in paths]

        for name in names:
            path = self.PATH_TEMPLATE.format(
                root=dirname(__file__),
                tab=self.parent.idx,
                index=self.idx,
                name=name
            )
            if exists(path):
                template = numpy.load(path)
                templates[name] = template

        # print(f'{self.idx} Loaded templates: {templates.keys()}')
        if cache:
            self._templates = templates
        return templates

    def get_bbox(self):

        if self.client.name == 'RuneLite':
            col = self.idx % self.parent.SLOTS_HORIZONTAL
            row = self.idx // self.parent.SLOTS_HORIZONTAL

            bx1, by1, bx2, by2 = self.parent.container.bank.get_bbox()

            bx_offset = self.config['offsets']['left']
            by_offset = self.config['offsets']['top']

            itm_width = self.config['width']
            itm_height = self.config['height']
            itm_x_margin = self.config['margins']['right']
            itm_y_margin = self.config['margins']['bottom']

            x1 = bx1 + bx_offset + ((itm_width + itm_x_margin - 1) * col)
            y1 = by1 + by_offset + ((itm_height + itm_y_margin - 1) * row)

            x2 = x1 + itm_width - 1
            y2 = y1 + itm_height - 1
        else:
            raise NotImplementedError

        return x1, y1, x2, y2

    def process_img(self, img):
        """
        Process raw image from screen grab into a format ready for template
        matching.
        :param img: BGRA image section for current slot
        :return: GRAY scaled image
        """
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        return img_gray

    def identify(self, threshold=None):
        """
        Compare incoming image with templates and try to find a match
        :return:
        """

        if not self.templates:
            print(f'Slot {self.idx}: no templates loaded, cannot identify')
            return False

        max_match = None
        matched_item = None
        for name, template in self.templates.items():
            match = cv2.matchTemplate(
                self.img, template, cv2.TM_CCOEFF_NORMED)[0][0]

            if max_match is None:
                max_match = match
                matched_item = name
            elif match > max_match:
                max_match = match
                matched_item = name

        threshold = threshold or 0.8
        if max_match and max_match > threshold:
            self.contents = matched_item
        # TODO: test for unknown items (i.e. slot is not empty)
        else:
            self.contents = None

        return self.contents


class BankUtilities(object):

    def __init__(self, client, bank):
        self._client = client
        self.bank = bank
        self.config = bank.config['utilities']
        self.deposit_inventory = DepositInventory(self._client, self)
        self._bbox = None

    @property
    def width(self):
        return self.config['width']

    @property
    def height(self):
        return self.config['height']

    def get_bbox(self):

        if self._bbox:
            return self._bbox

        if self._client.name == 'RuneLite':

            bx1, by1, bx2, by2 = self.bank.get_bbox()

            bb_margin = self.bank.config['margins']['bottom']
            br_margin = self.bank.config['margins']['bottom']

            x1 = bx2 - br_margin - self.width
            y1 = by2 - bb_margin - self.height

            x2 = x1 + self.width
            y2 = y1 + self.height

        else:
            raise NotImplementedError

        self._bbox = x1, y1, x2, y2
        return self._bbox


class DepositInventory(GameObject):

    PATH_TEMPLATE = '{root}/data/bank/utilities/deposit_inventory.npy'

    def __init__(self, client, parent):
        super(DepositInventory, self).__init__(
            client, parent,
            config_path='bank.utilities.deposit_inventory',
        )

        self.template = self.load_template()
        self._bbox = None

    @property
    def width(self):
        return self.config['width']

    @property
    def height(self):
        return self.config['height']

    def load_template(self):
        path = self.PATH_TEMPLATE.format(
            root=dirname(__file__),
        )
        if exists(path):
            return numpy.load(path)

    def get_bbox(self):
        if self.client.name == 'RuneLite':

            px1, py1, px2, py2 = self.parent.get_bbox()

            x_offset = self.config['offsets']['left']
            y_offset = self.config['offsets']['top']

            x1 = px1 + x_offset
            y1 = py1 + y_offset

            x2 = x1 + self.width
            y2 = y1 + self.height

        else:
            raise NotImplementedError

        self._bbox = x1, y1, x2, y2
        return self._bbox

    def identify(self, img):
        """
        Determine if the deposit inventory button is visible on screen
        :param img: Screen grab subsection where button is expected to be
        :return: True if matched, else False
        """

        if self.template is None:
            return False

        x, y, _, _ = self.client.get_bbox()
        x1, y1, x2, y2 = self.get_bbox()
        # numpy arrays are stored rows x columns, so flip x and y
        img = img[y1 - y:y2 - y, x1 - x:x2 - x]

        img = self.process_img(img)
        match = cv2.matchTemplate(img, self.template, cv2.TM_CCOEFF_NORMED)[0][0]

        threshold = 0.8
        return match > threshold


class PersonalMenu(GameObject):
    """
    The right-side menu with options that affect your personal character.
    For example, inventory, magic or logout menus
    """

    (
        COMBAT,
        SKILLS,
        QUESTS,
        INVENTORY,
        EQUIPMENT,
        PRAYER,
        MAGIC,
        FRIENDS,
        ACCOUNT,
        GROUPS,
        SETTINGS,
        EMOTES,
        MUSIC,
        LOGOUT,
        WORLD_SWITCHER,
    ) = range(15)

    def __init__(self, client):
        super(PersonalMenu, self).__init__(
            client, client, config_path='personal_menu',
            container_name='personal_menu'
        )
        self._context = None
        self._menus = self.create_menus()

    def create_menus(self):
        menus = dict()

        menus[self.COMBAT] = None
        menus[self.SKILLS] = None
        menus[self.QUESTS] = None
        # TODO: refactor to inherit from GameObject
        menus[self.INVENTORY] = self.client.inventory
        menus[self.EQUIPMENT] = None
        menus[self.PRAYER] = None
        menus[self.MAGIC] = None
        menus[self.FRIENDS] = None
        menus[self.ACCOUNT] = None
        menus[self.GROUPS] = None
        menus[self.SETTINGS] = None
        menus[self.EMOTES] = None
        menus[self.MUSIC] = None
        menus[self.LOGOUT] = LogoutMenu(self.client, self)
        menus[self.WORLD_SWITCHER] = WorldSwitcherMenu(self.client, self)

        return menus

    def get_menu(self, enum):
        return self._menus.get(enum)

    def toggle_context(self, new_context):
        # clicking a tab / context button while on a different menu just
        # switches to the new menu
        if new_context != self._context:
            self._context = new_context
        # if the menu is already open, then it closes the menu
        else:
            self._context = None


class Inventory(object):

    SLOTS_HORIZONTAL = 4
    SLOTS_VERTICAL = 7

    def __init__(self, client):
        self.client = client
        self.config = client.config['inventory']
        self.slots = self._create_slots()
        # remove this once refactored to GameObject class
        self.containers = dict()

    @property
    def width(self):
        return self.config['width']

    @property
    def height(self):
        return self.config['height']

    # TODO: Convert to GameObject subclass!!

    def _eval_config_value(self, value):
        return eval(str(value))

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

    def get_bbox(self):
        if self.client.name == 'RuneLite':

            cli_bbox = self.client.get_bbox()
            client_x2 = cli_bbox[2]
            client_y2 = cli_bbox[3]
            right_margin = self.client.config['margins']['right']
            bottom_margin = self.client.config['margins']['bottom']
            tab_height = self.client.tabs.height

            x1 = client_x2 - right_margin - self.width
            y1 = client_y2 - bottom_margin - tab_height - self.height

            x2 = x1 + self.width
            y2 = y1 + self.height
        else:
            raise NotImplementedError

        return x1, y1, x2, y2

    def _create_slots(self):
        """
        Create a set of empty slots
        :return: List of slots
        :rtype: list
        """
        slots = list()
        for i in range(self.SLOTS_HORIZONTAL * self.SLOTS_VERTICAL):
            slots.append(None)

        return slots

    def set_slot(self, idx, template_names):
        """
        Setup a slot object at provided index with provided template names
        :param idx: Index for the new slot
        :param template_names: List of template names the slot should load
        :return: new Slot object
        """
        slot = Slot(idx, self.client, self, template_names)
        self.slots[idx] = slot

        return slot

    def identify(self, img, threshold=None):
        """
        Runs identification on each slot in the inventory
        :param img: Screen grab of the whole client
        :return: List of items identified
        """

        # we need client bbox to zero the slot coordinates
        x, y, _, _ = self.client.get_bbox()

        items = list()
        for slot in self.slots:

            x1, y1, x2, y2 = slot.get_bbox()
            # numpy arrays are stored rows x columns, so flip x and y
            slot_img = img[y1 - y:y2 - y, x1 - x:x2 - x]

            name = slot.identify(slot_img, threshold=threshold)
            items.append(name)

        return items

    def contains(self, item_names):
        """
        Convenience function to test if any slot contains any of the items
        provided by name
        :param item_names: Any iterable, containing item names to test for.
            Will support None as well.
        :return: True if inventory contains any of the items, false False
        """
        for slot in self.slots:
            if slot.contents in item_names:
                return True

        return False

    def first(self, item_names, order=1, clicked=None):
        """
        Get the first inventory item that matches the provided filters
        :param set item_names: Options for items
        :param int order: Must be 1 or -1 for forward or reverse order
        :param bool clicked: If True, return the first clicked inventory slot,
            else if False return the first unclicked inventory slot
        :return: Slot matching filters or None if no matches
        """
        for slot in self.slots[::order]:

            if clicked is not None:
                if clicked and not slot.clicked:
                    continue
                elif not clicked and slot.clicked:
                    continue

            if slot.contents in item_names:
                return slot

    def filter_slots(self, item_names):
        slots = list()
        for slot in self.slots:
            if slot.contents in item_names:
                slots.append(slot)
        return slots


# TODO: fix this god awful mess
class SlotMixin:

    def get_bbox(self):

        if self._bbox:
            return self._bbox

        if self.client.name == 'RuneLite':
            col = self.idx % self.parent.SLOTS_HORIZONTAL
            row = self.idx // self.parent.SLOTS_HORIZONTAL

            inv_bbox = self.parent.get_bbox()
            inv_x1 = inv_bbox[0]
            inv_y1 = inv_bbox[1]

            inv_x_margin = self.parent.config['margins']['left']
            inv_y_margin = self.parent.config['margins']['top']

            itm_width = self.config['width']
            itm_height = self.config['height']
            itm_x_margin = self.config['margins']['right']
            itm_y_margin = self.config['margins']['bottom']

            x1 = inv_x1 + inv_x_margin + ((itm_width + itm_x_margin - 1) * col)
            y1 = inv_y1 + inv_y_margin + ((itm_height + itm_y_margin - 1) * row)

            x2 = x1 + itm_width - 1
            y2 = y1 + itm_height - 1
        else:
            raise NotImplementedError

        # cache bbox for performance
        self._bbox = x1, y1, x2, y2

        return x1, y1, x2, y2

    def identify(self, img, threshold=None):
        """
        Compare incoming image with templates and try to find a match
        :param img:
        :return:
        """

        if not self.templates:
            print(f'Slot {self.idx}: no templates loaded, cannot identify')
            return False

        img = self.process_img(img)

        max_match = None
        matched_item = None
        for name, template in self.templates.items():
            match = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)[0][0]

            if max_match is None:
                max_match = match
                matched_item = name
            elif match > max_match:
                max_match = match
                matched_item = name

        threshold = threshold or 0.8
        if max_match and max_match > threshold:
            self.contents = matched_item
        # TODO: test for unknown items (i.e. slot is not empty)
        else:
            self.contents = None

        return self.contents


class Slot(SlotMixin, GameObject):

    PATH_TEMPLATE = '{root}/data/inventory/{index}/{name}.npy'

    def __init__(self, idx, client, parent, template_names):
        self.idx = idx
        super(Slot, self).__init__(
            client, parent,
            template_names=template_names,
            config_path='inventory.slots',
        )

        self._bbox = None
        self.contents = None

    def resolve_path(self, **kwargs):
        return self.PATH_TEMPLATE.format(**kwargs)

    def load_templates(self, names=None, cache=True):
        """
        Load template data from disk
        :param names: List of names to attempt to load from disk
        :type names: list
        :return: Dictionary of templates of format {<name>: <numpy array>}
        """
        templates = dict()

        names = names or list()
        if not names:
            glob_path = self.resolve_path(
                root=dirname(__file__),
                index='*',
                name='*'
            )

            # print(f'{self.idx} GLOB PATH = {glob_path}')

            paths = glob(glob_path)
            names = [basename(p).replace('.npy', '') for p in paths]

        for name in names:
            path = self.resolve_path(
                root=dirname(__file__),
                index=self.idx,
                name=name
            )
            if exists(path):
                template = numpy.load(path)
                templates[name] = template

        # print(f'{self.idx} Loaded templates: {templates.keys()}')
        if cache:
            self._templates = templates
        return templates


class Magic(Inventory):

    SLOTS_HORIZONTAL = 5
    SLOTS_VERTICAL = 9

    def __init__(self, client, parent, spellbook=None):
        super(Magic, self).__init__(client)
        self.parent = parent
        self.spellbook = spellbook
        self.config = client.config[spellbook]
        self.slots = self._create_slots()

    def set_slot(self, idx, template_names):
        """
        Setup a slot object at provided index with provided template names
        :param idx: Index for the new slot
        :param template_names: List of template names the slot should load
        :return: new Slot object
        """
        slot = SpellSlot(idx, self.client, self, template_names)
        self.slots[idx] = slot

        return slot


class SpellSlot(SlotMixin, GameObject):

    PATH_TEMPLATE = '{root}/data/magic/{spellbook}/{name}.npy'

    SPELL_NAMES = {
        'lunar': [
            'lunar_home_teleport', 'bake_pie', 'geomancy', 'cure_plant', 'monster_examine',
            'npc_contact', 'cure_other', 'humidify', 'moonclan_teleport', 'tele_group_moonclan',
            'cure_me', 'ourania_telport', 'hunter_kit', 'waterbirth_telport', 'tele_group_waterbirth',
            'cure_group', 'stat_spy', 'barbarian_teleport', 'tele_group_barbarian', 'spin_flax',
            'superglass_make', 'tan_leather', # TODO: rest of the spellbook
        ]
    }

    def __init__(self, idx, client, parent, template_names):
        self.idx = idx
        super(SpellSlot, self).__init__(
            client, parent,
            template_names=template_names,
            config_path=f'{parent.spellbook}.slots',
        )

        self._bbox = None
        self.contents = None

    @property
    def name(self):
        return self.SPELL_NAMES[self.parent.spellbook][self.idx]

    def load_templates(self, names=None, cache=True):
        templates = dict()
        path = self.resolve_path(
            root=dirname(__file__)
        )

        if exists(path):
            template = numpy.load(path)
            templates[self.name] = template

        if cache:
            self._templates = templates
        return templates

    def resolve_path(self, **kwargs):
        kwargs['spellbook'] = self.parent.spellbook
        kwargs['name'] = self.name
        return self.PATH_TEMPLATE.format(**kwargs)


class Banner(GameObject):

    def __init__(self, client):
        super(Banner, self).__init__(
            client, client, config_path='banner'
        )


class MiniMapWidget(GameObject):

    def __init__(self, client):
        self.minimap = MiniMap(client, self)
        self.logout = LogoutButton(client, self)
        super(MiniMapWidget, self).__init__(
            client, client, config_path='minimap',
            container_name='minimap',
        )


class MiniMap(GameObject):

    MAP_PATH_TEMPLATE = '{root}/data/maps/{z}/{x}_{y}.png'
    PATH_TEMPLATE = '{root}/data/minimap/{name}.npy'
    RUNESCAPE_SURFACE = 0
    TAVERLY_DUNGEON = 20

    def __init__(self, client, parent, logging_level=None, **kwargs):
        self.logout_button = LogoutButton(client, parent)
        super(MiniMap, self).__init__(
            client, parent, config_path='minimap.minimap',
            logging_level=logging_level, **kwargs,
        )
        self._map_cache = dict()
        self._chunks = dict()
        self._coordinates = None

        # TODO: configurable feature matching methods
        self._detector = self._create_detector()
        self._matcher = self._create_matcher()
        self._mask = self._create_mask()

        # container for identified items/npcs/symbols etc.
        self._icons = dict()

        # image for display
        self.display_img = None

    def update(self):

        self.run_gps()
        self.identify()

    # minimap icon detection methods

    def identify(self, threshold=0.99):
        """
        Identify items/npcs/icons etc. on the minimap
        :param threshold:
        :return: A list of matches items of the format (item name, x, y)
            where x and y are tile coordinates relative to the player position
        """

        marked = set()
        checked = set()
        results = set()

        # get the player's current position on the map
        # assume gps has been run already
        v, w, X, Y, Z = self._coordinates

        # reset mark on all icons, so know which ones we've checked
        for i in self._icons.values():
            i.refresh()

        for name, template in self.templates.items():

            # for some reason masks cause way too many false matches,
            # so don't use a mask.
            matches = cv2.matchTemplate(
                self.img, template, cv2.TM_CCOEFF_NORMED)

            (my, mx) = numpy.where(matches >= threshold)
            for y, x in zip(my, mx):

                # guard statement prevents two templates matching the same
                # icon, which would cause duplicates
                if (x, y) in marked:
                    continue
                marked.add((x, y))

                px, py, _, _ = self.client.game_screen.player.mm_bbox()
                mm_x, mm_y, _, _ = self.get_bbox()
                px = px - mm_x + 1
                py = py - mm_y + 1

                # calculate item relative pixel coordinate to player
                rx = x - px
                ry = y - py
                # rx = int((x - self.config['width'] / 2) * self.scale)
                # ry = int((y - self.config['height'] / 2) * self.scale)

                # convert pixel coordinate into tile coordinate
                tx = rx // self.tile_size
                ty = ry // self.tile_size

                # TODO: method to add coordinates
                # calculate icon's global map coordinate
                # v += tx
                # w += ty

                # key by pixel
                key = rx, ry

                added_on_adjacent = False
                try:
                    icon = self._icons[key]

                    # This usually happens when a tagged npc dies and is
                    # untagged, so the coordinates match, but it should be a
                    # different entity
                    if icon.name != name:
                        continue

                    icon.update()
                    checked.add(key)
                    continue
                except KeyError:

                    # FIXME: calculate pixel position on map and use that to
                    #        determine nearest candidate
                    icon_copy = [i.key for i in self._icons.values()]
                    max_dist = 1
                    for icon_key in icon_copy:
                        # TODO: method to calc distance between coords
                        if (abs(tx - icon_key[0]) <= max_dist and
                                abs(ty - icon_key[1]) <= max_dist):
                            # move npc to updated key
                            icon = self._icons.pop(icon_key)
                            self._icons[key] = icon
                            icon.update(key=key)
                            added_on_adjacent = True
                            continue

                # finally if we still can't find it, we must have a new one
                if key not in checked and not added_on_adjacent:

                    icon = self.client.game_screen.create_game_entity(
                        name, name, key, self.client, self.client)

                    icon.update(key)
                    self._icons[key] = icon

        # do one final check to remove any that are no longer on screen
        keys = list(self._icons.keys())
        for k in keys:
            icon = self._icons[k]
            if not icon.checked:
                self._icons.pop(k)

        return results

    # GPS map matching methods

    def _create_detector(self):
        return cv2.ORB_create()

    def _create_matcher(self):
        return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def _create_mask(self):

        shape = (self.config['width'] + 1, self.config['height'] + 1)
        mask = numpy.zeros(
            shape=shape, dtype=numpy.dtype('uint8'))

        x, y = mask.shape
        x //= 2
        y //= 2

        size = self.config['width'] // 2 - self.config['padding']

        mask = cv2.circle(mask, (x, y), size, WHITE, FILL)

        # TODO: create additional cutouts for orbs that slightly overlay the
        #       minimap. Not hugely important, but may interfere with feature
        #       matching.

        return mask

    def x0(self):
        x0, _, x1, _ = self.get_bbox()
        return (x0 + (x1 - x0) // 2) - 1

    def set_coordinates(self, v, w, x, y, z):
        self._coordinates = v, w, x, y, z

    def get_coordinates(self):
        return self._coordinates

    def _map_key_points(self, match, kp1, kp2):
        # get pixel coordinates of feature within minimap image
        x1, y1 = kp1[match.queryIdx].pt
        # get pixel coords of feature in main map
        x2, y2 = kp2[match.trainIdx].pt

        # calculate player coordinate in main map
        px = int((self.config['width'] / 2 - x1) * self.scale + x2)
        py = int((self.config['height'] / 2 - y1) * self.scale + y2)

        # convert player pixel coordinate into tile coordinate
        px //= self.tile_size
        py //= self.tile_size

        return px, py

    def run_gps(self, train_chunk=None):

        query_img = self.img
        kp1, des1 = self._detector.detectAndCompute(query_img, self._mask)

        if train_chunk:
            train_img = self.get_chunk(*train_chunk)
            radius = (self.chunk_shape_x // 2) // self.tile_size
        else:
            radius = 25
            train_img = self.get_local_zone(*self._coordinates, radius=radius)
        kp2, des2 = self._detector.detectAndCompute(train_img, None)

        matches = self._matcher.match(des1, des2)
        self.logger.debug(f'got {len(matches)} matches')

        filtered_matches = self._filter_matches_by_grouping(matches, kp1, kp2)
        self.logger.debug(f'filtered to {len(filtered_matches)} matches')

        # determine pixel coordinate relationship of minimap to map for each
        # of the filtered matches, and pick the modal tile coordinate
        mapped_coords = defaultdict(int)
        for match in filtered_matches:
            tx, ty = self._map_key_points(match, kp1, kp2)
            mapped_coords[(tx, ty)] += 1
        sorted_mapped_coords = sorted(mapped_coords.items(), key=lambda item: item[1])
        (tx, ty), freq = sorted_mapped_coords[-1]
        self.logger.debug(f'got tile coord {tx, ty} (frequency: {freq})')

        # determine relative coordinate change to create new coordinates
        v, w, x, y, z = self._coordinates
        rel_v = tx - radius
        rel_w = ty - radius
        self.logger.debug(f'relative change: {rel_v, rel_w}')

        # it is the responsibility of the script to determine if a proposed
        # coordinate change is possible since the last time the gps was pinged.
        # TODO: record each time gps is pinged and calculate potential
        #       destinations since last gps pinged
        if abs(rel_v) > 4 or abs(rel_w) > 4:
            self.logger.debug(f'excessive position change: {rel_v, rel_w}')

        new_v = int((v + rel_v) % (self.max_tile + 1))
        new_w = int((w + rel_w) % (self.max_tile + 1))

        new_x = x + self.compare_coordinate(v + rel_v)
        # NOTE: map coordinates have (0, 0) as bottom left
        #       so we must flip the y value
        new_y = y - self.compare_coordinate(v + rel_v)

        self.logger.debug(f'new coords: '
                         f'{new_v, new_w, new_x, new_y, z}')
        new_coordinates = new_v, new_w, new_x, new_y, z

        # GPS needs to be shown in a separate windows because it isn't
        # strictly part of the client image.
        if 'gps' in self.client.args.show:

            train_img_copy = train_img.copy()
            ptx0, pty0 = int(tx * self.tile_size), int(ty * self.tile_size)
            ptx1 = ptx0 + self.tile_size - 1
            pty1 = pty0 + self.tile_size - 1

            self.logger.debug(f'position: {ptx0}, {pty0}')

            train_img_copy = cv2.rectangle(
                train_img_copy, (ptx0, pty0), (ptx1, pty1), WHITE, FILL)

            show_img = cv2.drawMatches(
                query_img, kp1, train_img_copy, kp2, filtered_matches,
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            # set display image, it will be shown later in the application
            # event cycle
            self.display_img = show_img

        self._coordinates = new_coordinates
        return new_coordinates

    def _filter_matches_by_grouping(self, matches, kp1, kp2):

        # pre-filter matches in case we get lots of poor matches
        filtered_matches = [m for m in matches if m.distance < 70]

        groups = defaultdict(list)
        for m in filtered_matches:
            tx, ty = self._map_key_points(m, kp1, kp2)
            groups[(tx, ty)].append(m)

        # normalise the number of matches per group
        max_num_matches = max([len(v) for k, v in groups.items()], default=0)
        normalised_average = dict()
        for (k, v) in groups.items():
            average_distance = sum([m_.distance for m_ in v]) / len(v)
            normalised_len = self.client.screen.normalise(
                len(v), stop=max_num_matches)
            normalised_average[k] = (
                average_distance / normalised_len
            )

        sorted_normalised_average = sorted(
            [(k, v) for k, v in normalised_average.items()],
            # sort by normalised value, lower means more matches and lower
            key=lambda item: item[1])
        self.logger.debug(
            f'top 5 normalised matches: {sorted_normalised_average[:5]}')

        key, score = sorted_normalised_average[0]
        filtered_matches = groups[key]

        return filtered_matches

    @property
    def chunk_shape_x(self):
        return self.config['chunk_shape'][1]

    @property
    def chunk_shape_y(self):
        return self.config['chunk_shape'][0]

    @property
    def tile_size(self):
        return self.config['tile_size']

    @property
    def min_tile(self):
        return 0

    @property
    def max_tile(self):
        return (self.chunk_shape_x / self.tile_size) - 1

    @property
    def match_tolerance(self):
        return self.tile_size // 2

    def compare_coordinate(self, u):
        """
        Compare given coordinate against chunk min and max values
        :param int u: Target coordinate
        :return: -1 if coordinate is less than chunk minimum,
                  0 if within chunk bounds,
                  1 if greater than chunk maximum
        """
        return (u > self.max_tile) - (u < self.min_tile)

    def load_chunks(self, *chunks, fill_missing=None):

        for (x, y, z) in chunks:

            # attempt to load the map chunk from disk
            chunk_path = self.MAP_PATH_TEMPLATE.format(
                root=dirname(__file__),
                x=x, y=y, z=z,
            )
            chunk = cv2.imread(chunk_path)

            # resolve if disk file does not exist
            if chunk is None:
                if fill_missing is None:
                    shape = self.config.get('chunk_shape', (256, 256))
                    chunk_grey = numpy.zeros(
                        shape=shape, dtype=numpy.dtype('uint8'))
                # TODO: implement requests method
                else:
                    raise NotImplementedError
            else:
                # TODO: implement process chunk method
                chunk_grey = cv2.cvtColor(chunk, cv2.COLOR_BGR2GRAY)

            # add to internal cache
            self._chunks[(x, y, z)] = chunk_grey

    def get_chunk(self, x, y, z):

        chunk = self._chunks.get((x, y, z))
        if chunk is None:
            self.load_chunks((x, y, z))
            chunk = self._chunks.get((x, y, z))

        return chunk

    def _calculate_chunk_set(self, v, w, x, y, z, radius):

        chunks = set()

        v0 = v - radius
        v1 = v + radius
        w0 = w - radius
        w1 = w + radius
        for _v in [v0, v1]:
            cmp_v = self.compare_coordinate(_v)
            for _w in [w0, w1]:
                cmp_w = self.compare_coordinate(_w)
                new_x = x + cmp_v
                # NOTE: map coordinates have (0, 0) as bottom left
                #       so we must flip the y value
                new_y = y - cmp_w

                chunk_ref = (new_x, new_y, z)
                chunks.add(chunk_ref)

        return chunks

    def _get_chunk_set_boundary(self, chunk_set):

        min_x = min_y = float('inf')
        max_x = max_y = -float('inf')
        z = None

        for (x, y, _z) in chunk_set:
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y

            # z is assumed to be constant
            z = _z

        return min_x, min_y, max_x, max_y, z

    def _arrange_chunk_matrix(self, chunks):

        min_x, min_y, max_x, max_y, z = self._get_chunk_set_boundary(chunks)

        chunk_list = list()
        # NOTE: chunks are numbered from bottom left, so we must iterate in
        #       the opposite direction
        for y in range(max_y, min_y - 1, -1):
            chunk_row = list()
            for x in range(min_x, max_x + 1):
                chunk_row.append((x, y, z))
            chunk_list.append(chunk_row)

        return chunk_list

    def get_map(self, chunk_set):

        for name, map_info in self._map_cache.items():
            map_chunk_set = map_info.get('chunks', set())
            if chunk_set.issubset(map_chunk_set):
                return map_info

        # if we get this far it means no map is cached that covers all our
        # required chunks yet, so we must create it
        return self.create_map(chunk_set)

    def create_map(self, chunk_set):
        chunk_matrix = self._arrange_chunk_matrix(chunk_set)
        map_data = self.concatenate_chunks(chunk_matrix)

        map_info = dict(
            data=map_data,
            chunks=chunk_set,
        )

        name = len(self._map_cache)
        self._map_cache[name] = map_info
        return map_info

    def concatenate_chunks(self, chunk_matrix):

        col_data = list()
        for row in chunk_matrix:
            row_data = list()
            for (x, y, z) in row:
                chunk = self.get_chunk(x, y, z)
                row_data.append(chunk)
            row_data = numpy.concatenate(row_data, axis=1)
            col_data.append(row_data)
        concatenated_chunks = numpy.concatenate(col_data, axis=0)

        return concatenated_chunks

    def get_local_zone(self, v, w, x, y, z, radius=25):
        """
        TODO: figure out why local zone it sliding, even if coords don't change
        :param v: Horizontal tile index within current chunk
        :param w: Vertical tile index within current chunk
        :param x: Horizontal chunk index within current map
        :param y: Vertical chunk within current map
        :param z: Map index within world
        :param radius: Number of tiles around starting location to expand
            local zone image
        :return: Sub-matrix of map image around the starting location
        """

        # first check we're not trying to access tiles outside of the chunk
        assert self.compare_coordinate(v) == 0
        assert self.compare_coordinate(w) == 0

        chunk_set = self._calculate_chunk_set(v, w, x, y, z, radius)
        self.logger.debug(f'calculated chunk set {chunk_set} '
                         f'from coords: {v, w, x, y, z}')
        map_ = self.get_map(chunk_set)

        map_data = map_.get('data')
        map_chunks = map_.get('chunks')
        min_x, min_y, max_x, max_y, z = self._get_chunk_set_boundary(map_chunks)
        pixel_radius = (radius * self.tile_size)

        local_x = (
                # determine how many chunks from the left we are
                (x - min_x)
                # multiply by size (in pixels) to arrive at chunk left border
                * self.chunk_shape_x
                # add horizontal local coord inside chunk (left to right)
                + v * self.tile_size
        )
        min_local_x = local_x - pixel_radius
        max_local_x = local_x + pixel_radius

        self.logger.debug(f'x: {min_local_x}, {max_local_x}')

        local_y = (
            # determine how many chunks from the top we are
            (max_y - y)
            # multiply by size (in pixels) to arrive at chunk top border
            * self.chunk_shape_y
            # add vertical local coord inside chunk (top to bottom)
            + w * self.tile_size
        )
        min_local_y = local_y - pixel_radius
        max_local_y = local_y + pixel_radius

        self.logger.debug(f'y: {min_local_y}, {max_local_y}')

        local_zone = map_data[min_local_y:max_local_y, min_local_x:max_local_x]

        return local_zone

    def load_map_sections(self, sections):
        return numpy.concatenate(
            [numpy.concatenate(
                [cv2.imread(self.MAP_PATH_TEMPLATE.format(root=dirname(__file__), name=name))
                 for name in row], axis=1)
                for row in sections], axis=0
        )

    @property
    def scale(self):
        return self.config['scale']


class LogoutButton(GameObject):

    def __init__(self, client, parent):
        super(LogoutButton, self).__init__(
            client, parent, config_path='minimap.logout',
            container_name='logout',
        )

    @property
    def clickable(self):
        # TODO: if bank is open, return False
        return True


class WorldSwitcherMenu(GameObject):

    def __init__(self, client, parent):
        self.logout_button = WorldSwitcherMenuLogoutButton(client, self)
        super(WorldSwitcherMenu, self).__init__(
            client, parent, config_path='personal_menu.world_switcher',
            container_name=PersonalMenu.WORLD_SWITCHER
        )

    def setup_containers(self):
        containers = dict()

        containers['exit_buttons'] = {
            'y': [self.logout_button]
        }

        return containers


class WorldSwitcherMenuLogoutButton(GameObject):

    PATH_TEMPLATE = '{root}/data/pmenu/world_switcher/{name}.npy'

    def __init__(self, client, parent):
        super(WorldSwitcherMenuLogoutButton, self).__init__(
            client, parent, config_path='personal_menu.world_switcher.logout',
            container_name='exit_buttons',
            template_names=['logout', 'logout_hover'],
        )


class LogoutMenu(GameObject):

    def __init__(self, client, parent):
        self.logout_button = LogoutMenuLogoutButton(client, self)
        super(LogoutMenu, self).__init__(
            client, parent, config_path='personal_menu.logout',
            container_name=PersonalMenu.LOGOUT
        )

    def setup_containers(self):
        containers = dict()

        containers['exit_buttons'] = {
            'y': [self.logout_button]
        }

        return containers


class LogoutMenuLogoutButton(GameObject):

    PATH_TEMPLATE = '{root}/data/pmenu/logout/{name}.npy'

    def __init__(self, client, parent):
        super(LogoutMenuLogoutButton, self).__init__(
            client, parent, config_path='personal_menu.logout.logout',
            container_name='exit_buttons',
            template_names=['logout', 'logout_hover'],
        )
