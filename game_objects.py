

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
