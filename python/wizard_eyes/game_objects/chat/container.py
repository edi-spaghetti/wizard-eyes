from .interface import ChatInterface
from .widget import ChatWidget
from ...dynamic_menus.container import AbstractContainer


class Chat(AbstractContainer):
    """Container for the chat interface tabs."""

    PATH_TEMPLATE = '{root}/data/chat/{name}.npy'
    STATIC_TABS = [
        'all',
        'report',
    ]
    MUTABLE_TABS = {
        'game': ['filtered', 'on'],
        'public': ['autochat', 'on', 'friends', 'off', 'hide'],
        'private': ['on', 'friends', 'off'],
        'channel': ['on', 'friends', 'off'],
        'clan': ['on', 'friends', 'off'],
        'trade': ['on', 'friends', 'off'],
    }
    PERMUTATIONS = ['hover', 'pending', 'selected', 'hover_selected']

    def __init__(self, client):

        super(Chat, self).__init__(
            client, client, config_path='chat',
            container_name='chat',
        )

        # add placeholders for the tabs we expect to find
        self.all = None
        self.game = None
        self.public = None
        self.private = None
        self.channel = None
        self.clan = None
        self.trade = None
        self.report = None

    @property
    def widget_class(self):
        """"""
        return ChatWidget

    def load_widget_masks(self, names=None):
        """"""
        self.load_masks(['report_mask', 'chat_widget_mask'])
        for name in names:
            if name == 'report':
                self.masks['report'] = self.masks.get('report_mask')
            else:
                self.masks[name] = self.masks.get('chat_widget_mask')

    def widget_masks(self, name):
        """"""
        if name == 'report':
            return ['report_mask']
        else:
            return ['chat_widget_mask']

    @property
    def interface_class(self):
        """Interface object used when crafting dialog opens."""
        return ChatInterface

    @property
    def interface_init_params(self):
        return (self.client, self.client), dict()
