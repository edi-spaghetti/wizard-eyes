import ctypes

# TODO: use scale factor and determine current screen to apply to any config
#       values. For the time being I'm setting system scaling factor to 100%
scale_factor = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100


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


class Inventory(object):

    SLOTS_HORIZONTAL = 4
    SLOTS_VERTICAL = 7

    def __init__(self, client):
        self._client = client
        self.config = client.config['inventory']
        self.slots = self._create_slots()

    @property
    def width(self):
        return self.config['width']

    @property
    def height(self):
        return self.config['height']

    def get_bbox(self):
        if self._client.name == 'RuneLite':

            cli_bbox = self._client.get_bbox()
            client_x2 = cli_bbox[2]
            client_y2 = cli_bbox[3]
            right_margin = self._client.config['margins']['right']
            bottom_margin = self._client.config['margins']['bottom']
            tab_height = self._client.tabs.height

            x1 = client_x2 - right_margin - self.width
            y1 = client_y2 - bottom_margin - tab_height - self.height

            x2 = x1 + self.width
            y2 = y1 + self.height
        else:
            raise NotImplementedError

        return x1, y1, x2, y2

    def _create_slots(self):

        slots = list()
        for i in range(self.SLOTS_HORIZONTAL * self.SLOTS_VERTICAL):

            slot = Slot(i, self._client, self)
            slots.append(slot)

        return slots


class Slot(object):

    def __init__(self, idx, client, inventory):
        self.idx = idx
        self.templates = self.load_templates()
        self._bbox = None
        self._client = client
        self.inventory = inventory
        self.config = inventory.config['slots']

    def load_templates(self):
        templates = dict()
        # TODO
        return templates

    def get_bbox(self):

        if self._bbox:
            return self._bbox

        if self._client.name == 'RuneLite':
            col = self.idx % self.inventory.SLOTS_HORIZONTAL
            row = self.idx // self.inventory.SLOTS_HORIZONTAL

            inv_bbox = self.inventory.get_bbox()
            inv_x1 = inv_bbox[0]
            inv_y1 = inv_bbox[1]

            inv_x_margin = self.inventory.config['margins']['left']
            inv_y_margin = self.inventory.config['margins']['top']

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

    def identify(self, img):
        # TODO
        pass
