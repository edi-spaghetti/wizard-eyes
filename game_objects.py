import time
import random
import ctypes
from os.path import dirname, exists, basename
from glob import glob

import numpy
import cv2
import pyautogui
import keyboard

# TODO: use scale factor and determine current screen to apply to any config
#       values. For the time being I'm setting system scaling factor to 100%
scale_factor = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100


class Timeout(object):

    def __init__(self, offset):
        self.created_at = time.time()
        self.offset = self.created_at + offset


class GameObject(object):

    def __init__(self, client, parent, config_path=None):
        self.client = client
        self.parent = parent
        self.context_menu = None
        self._bbox = None
        self.config = self._get_config(config_path)

        # audit fields
        self._clicked = list()

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

    def set_aoi(self, x1, y1, x2, y2):
        self._bbox = x1, y1, x2, y2

    def get_bbox(self):
        return self._bbox

    def clear_bbox(self):
        self._bbox = None

    def update(self):
        """
        check and remove timeouts that have expired
        """

        i = 0
        while i < len(self._clicked):
            t = self._clicked[i]
            if time.time() > t.offset:
                self._clicked = self._clicked[:i]
                break

            i += 1

        if self.context_menu:

            # if the mouse has moved outside the context menu bbox, it will
            # have definitely been dismissed
            x, y = pyautogui.position()
            if not self.context_menu.is_inside(x, y):
                self.context_menu = None
                return

            # TODO: check if it has timed out

    def set_context_menu(self, x, y, width, items, config):
        menu = ContextMenu(
            self.client, self.parent, x, y, width, items, config)
        self.context_menu = menu
        return menu

    @property
    def clicked(self):
        self.update()
        return self._clicked

    def click(self, tmin=None, tmax=None, speed=1, pause_before_click=False,
              shift=False):
        x, y = self.client.screen.click_aoi(
            *self.get_bbox(),
            speed=speed,
            pause_before_click=pause_before_click,
            shift=shift,
        )
        # TODO: configurable timeout
        tmin = tmin or 1
        tmax = tmax or 3
        offset = self.client.screen.map_between(random.random(), tmin, tmax)
        self._clicked.append(Timeout(offset))

        return x, y

    def right_click(self, tmin=None, tmax=None, speed=1, pause_before_click=False):
        x, y = self.client.screen.click_aoi(
            *self.get_bbox(), right=True, click=False, speed=speed, pause_before_click=pause_before_click)
        tmin = tmin or 1
        tmax = tmax or 3
        offset = self.client.screen.map_between(random.random(), tmin, tmax)
        self._clicked.append(Timeout(offset))

        return x, y

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
            time_left = time_left - time.time()
            return round(time_left, 2)
        except ValueError:
            return 0

    def is_inside(self, x, y):
        x1, y1, x2, y2 = self.get_bbox()
        return x1 <= x <= x2 and y1 <= y <= y2


class ContextMenu(GameObject):

    ITEM_HEIGHT = 15

    def __init__(self, client, parent, x, y, width, items, config):
        super(ContextMenu, self).__init__(client, parent)

        self.x = x
        self.y = y
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


class Tabs(object):

    def __init__(self, client):
        self._client = client
        self.config = client.config['tabs']

    @property
    def width(self):
        # TODO: double tab stack if client width below threshold
        return self.config['width'] * 13

    @property
    def height(self):
        # TODO: double tab stack if client width below threshold
        return self.config['height'] * 1

    def get_bbox(self):
        if self._client.name == 'RuneLite':
            cli_bbox = self._client.get_bbox()
            client_x2 = cli_bbox[2]
            client_y2 = cli_bbox[3]
            right_margin = self._client.config['margins']['right']
            bottom_margin = self._client.config['margins']['bottom']

            x1 = client_x2 - right_margin - self.width
            y1 = client_y2 - bottom_margin - self.height

            x2 = x1 + self.width
            y2 = y1 + self.height
        else:
            raise NotImplementedError

        return x1, y1, x2, y2


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
        super(Bank, self).__init__(client, client)
        self._bbox = None
        self.config = client.config['bank']
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
        super(BankSlot, self).__init__(client, parent)

        self.idx = idx
        self._bbox = None
        self.config = parent.config['slots']
        self.contents = None
        self.templates = self.load_templates(names=template_names)

    def load_templates(self, names=None):
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

    def identify(self, img, threshold=None):
        """
        Compare incoming image with templates and try to find a match
        :param img: Subsection of client window with same shape as templates
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
        super(DepositInventory, self).__init__(client, parent)

        self.config = parent.config['deposit_inventory']
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

    def process_img(self, img):
        """
        Process raw image from screen grab into a format ready for template
        matching.
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

        x, y, _, _ = self.client.get_bbox()
        x1, y1, x2, y2 = self.get_bbox()
        # numpy arrays are stored rows x columns, so flip x and y
        img = img[y1 - y:y2 - y, x1 - x:x2 - x]

        img = self.process_img(img)
        match = cv2.matchTemplate(img, self.template, cv2.TM_CCOEFF_NORMED)[0][0]

        threshold = 0.8
        return match > threshold


class Inventory(object):

    SLOTS_HORIZONTAL = 4
    SLOTS_VERTICAL = 7

    def __init__(self, client):
        self.client = client
        self.config = client.config['inventory']
        self.slots = self._create_slots()

    @property
    def width(self):
        return self.config['width']

    @property
    def height(self):
        return self.config['height']

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

    def first(self, item_names, order=1):
        for slot in self.slots[::order]:
            if slot.contents in item_names:
                return slot

    def filter_slots(self, item_names):
        slots = list()
        for slot in self.slots:
            if slot.contents in item_names:
                slots.append(slot)
        return slots


class Slot(GameObject):

    PATH_TEMPLATE = '{root}/data/inventory/{index}/{name}.npy'

    def __init__(self, idx, client, parent, template_names):
        super(Slot, self).__init__(client, parent)

        self.idx = idx
        self.templates = self.load_templates(names=template_names)
        self._bbox = None
        self.contents = None
        self.config = parent.config['slots']

    def resolve_path(self, **kwargs):
        return self.PATH_TEMPLATE.format(**kwargs)

    def load_templates(self, names=None):
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

        return templates

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

    def process_img(self, img):
        """
        Process raw image from screen grab into a format ready for template
        matching.
        :param img: BGRA image section for current slot
        :return: GRAY scaled image
        """
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        return img_gray

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


class SpellSlot(Slot):

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
        super(SpellSlot, self).__init__(idx, client, parent, template_names)

    @property
    def name(self):
        return self.SPELL_NAMES[self.parent.spellbook][self.idx]

    def load_templates(self, names=None):
        templates = dict()
        path = self.resolve_path(
            root=dirname(__file__)
        )

        if exists(path):
            template = numpy.load(path)
            templates[self.name] = template

        return templates

    def resolve_path(self, **kwargs):
        kwargs['spellbook'] = self.parent.spellbook
        kwargs['name'] = self.name
        return self.PATH_TEMPLATE.format(**kwargs)
