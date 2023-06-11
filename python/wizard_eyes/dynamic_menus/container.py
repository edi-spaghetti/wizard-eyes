from abc import ABC, abstractmethod
from typing import Union, List

from ..game_objects.game_objects import GameObject
from .interface import AbstractInterface
from .widget import AbstractWidget


class AbstractContainer(GameObject, ABC):
    """Not a real object, but a collection of widgets that may or may not
    enable an interface."""

    STATIC_TABS = []
    MUTABLE_TABS = {}
    PERMUTATIONS = []

    def __init__(self, client: 'wizard_eyes.client.Client', *args, **kwargs):
        super().__init__(client, *args, **kwargs)

        # init attributes for dynamic objects
        self.active_tab = None
        self._widgets: List[AbstractWidget] = []
        self.interface: Union[AbstractInterface, None] = None

    @property
    def name(self):
        """The name for this container defines where templates are stored,
        as well as signifying what type of widget it contains. By convention,
        this is taken from the class name."""
        return self.__class__.__name__.lower()

    def get_template_names(self, tabs):
        """
        Get all template names for given tab names.
        Resolve all types and permutations where necessary.
        Return a list of template names as strings.
        """
        template_names = list()
        for tab in tabs:
            if tab in self.STATIC_TABS:
                template_names.append(tab)
                for permutation in self.PERMUTATIONS:
                    template_names.append(f'{tab}_{permutation}')
            if tab in self.MUTABLE_TABS:
                types = self.MUTABLE_TABS[tab]
                for type_ in types:
                    template_names.append(f'{tab}_{type_}')
                    for permutation in self.PERMUTATIONS:
                        template_names.append(f'{tab}_{type_}_{permutation}')
        return template_names

    def create_widget(self, name):
        """Create, register and return a widget for this container."""

        widget = self.widget_class(
            name,
            self.client,
            self,
            template_names=self.get_template_names([name])
        )

        self._widgets.append(widget)

        return widget

    @property
    @abstractmethod
    def widget_class(self):
        """
        Define the class for tab items on this container,
        used when running :meth:`DynamicContainer.build_tab_items`.
        """

    def update(self):
        """
        Run update on each of the tab items.
        """

        super().update()

        # set to none on each update,
        # if we find an active one it should be replaced
        self.active_tab = None

        for widget in self._widgets:
            widget.update()
            if widget.selected:
                self.active_tab = widget
