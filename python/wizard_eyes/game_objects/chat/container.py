from .interface import ChatInterface
from .widget import ChatWidget
from ...dynamic_menus.container import AbstractContainer


class Chat(AbstractContainer):
    """Container for the chat interface tabs."""

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
        super(Chat, self).__init__(client, client)

        # add placeholders for the tabs we expect to find
        # sometimes a chat interface pops up even if none of the widgets are
        # selected e.g. when you talk to an NPC. In this case use 'all'.
        self.all = self.create_widget('all')
        self.game = self.create_widget('game')
        self.public = self.create_widget('public')
        self.private = self.create_widget('private')
        self.channel = self.create_widget('channel')
        self.clan = self.create_widget('clan')
        self.trade = self.create_widget('trade')
        self.report = self.create_widget('report')

    @property
    def widget_class(self):
        """Widget class that represents the buttons on bottom left."""
        return ChatWidget

    @property
    def interface_class(self):
        """Interface object used when chat dialog opens."""
        return ChatInterface
