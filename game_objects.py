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

    def get_slot_bbox(self, idx):
        if self._client.name == 'RuneLite':
            col = idx % self.SLOTS_HORIZONTAL
            row = idx // self.SLOTS_HORIZONTAL

            inv_bbox = self.get_bbox()
            inv_x1 = inv_bbox[0]
            inv_y1 = inv_bbox[1]

            inv_x_margin = self.config['margin']['left']
            inv_y_margin = self.config['margin']['top']

            itm_width = self.config['slots']['width']
            itm_height = self.config['slots']['height']
            itm_x_margin = self.config['slots']['margins']['right']
            itm_y_margin = self.config['slots']['margins']['bottom']

            x1 = inv_x1 + inv_x_margin + ((itm_width + itm_x_margin - 1) * col)
            y1 = inv_y1 + inv_y_margin + ((itm_height + itm_y_margin - 1) * row)

            x2 = x1 + itm_width - 1
            y2 = y1 + itm_height - 1

            return x1, y1, x2, y2
