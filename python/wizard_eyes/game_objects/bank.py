import random
from glob import glob
from os.path import basename, exists

import cv2
import numpy

from ..file_path_utils import get_root
from .game_objects import GameObject
from .timeout import Timeout


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
        self.add_timeout(offset)

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
                root=get_root(),
                tab=self.parent.idx,
                index=self.idx,
                name='*'
            )

            # print(f'{self.idx} GLOB PATH = {glob_path}')

            paths = glob(glob_path)
            names = [basename(p).replace('.npy', '') for p in paths]

        for name in names:
            path = self.PATH_TEMPLATE.format(
                root=get_root(),
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
            root=get_root(),
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
