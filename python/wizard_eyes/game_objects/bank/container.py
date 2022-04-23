
from .interface import BankInterface
from ...dynamic_menus.container import AbstractContainer


class Bank(AbstractContainer):

    PATH_TEMPLATE = '{root}/data/bank/{name}.npy'
    CLOSE_HOTKEY = 'esc'

    def __init__(self, client):

        super().__init__(
            client, client, config_path='bank',
        )

        # TODO: remodel this container into a container of containers
        #       the bank menu has quite a lot of features, but I don't need
        #       all / any of them right now, so going to hold off implementing.

    @property
    def interface_class(self):
        return BankInterface

    @property
    def interface_init_params(self):
        return (self.client, self), dict()

    @property
    def widget_class(self):
        # TODO: bank widgets for tabs, tags and other widgets
        from unittest import mock
        return mock.MagicMock()

    def get_bbox(self):
        # TODO: fix this god awful mess

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

    def close(self):

        self.client.screen.press_key(self.CLOSE_HOTKEY)
        offset = self.client.screen.map_between(start=0.6, stop=1.2)
        self.add_timeout(offset)

        return self.CLOSE_HOTKEY
