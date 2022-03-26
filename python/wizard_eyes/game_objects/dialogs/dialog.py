from os.path import exists

import cv2
import numpy

from ...file_path_utils import get_root


class Dialog(object):
    """
    Class to represent the bottom left dialog box, used for talking to NPCs,
    making some items etc.
    """

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
            root=get_root(),
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
