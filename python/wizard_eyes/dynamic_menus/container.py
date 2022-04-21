from abc import ABC, abstractmethod

import cv2

from ..game_objects.game_objects import GameObject


class AbstractContainer(GameObject, ABC):

    STATIC_TABS = []
    MUTABLE_TABS = {}
    PERMUTATIONS = []

    def __init__(self, *args, **kwargs):

        # set up templates with defaults & modifiers
        template_names = (
                self.get_template_names(self.STATIC_TABS) +
                self.get_template_names(self.MUTABLE_TABS)
        )
        super().__init__(*args, template_names=template_names, **kwargs)
        self.load_widget_masks(template_names)

        # init attributes for dynamic objects
        self.active_tab = None
        self._tabs = None
        self.interface = None

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

    @property
    @abstractmethod
    def widget_class(self):
        """
        Define the class for tab items on this container,
        used when running :meth:`DynamicContainer.build_tab_items`.
        """

    @property
    @abstractmethod
    def interface_class(self):
        """
        Define the class for the interface that opens on activate any of the
        tab widgets. Note, each widget will generate their own class,
        so they are able to separately track icons within them. Usually this
        interface is only used for layout.
        """

    @property
    @abstractmethod
    def interface_init_params(self):
        """
        Define the args and kwargs required to initialise the interface class
        as a tuple pair.
        """

    def widget_masks(self, tab):
        """
        Get mask names for a given tab widget (static or dynamic).
        By default this will resolve the same name as the template name given.
        Return a list of resolved names as strings.
        """
        masks = self.get_template_names([tab])
        return masks

    def load_widget_masks(self, names=None):
        """
        Wrapper around game object method,
        because in some cases we want to use different masks per tab,
        other times they all use the same mask.
        """
        self.load_masks(names, cache=True)

    def post_init(self):

        # add a hidden interface (which in the case of the chat menu can
        # actually be active even if none of the widgets are)
        args, kwargs = self.interface_init_params
        self.interface = self.interface_class(*args, **kwargs)

        self.build_tab_items()

    def build_tab_items(self, threshold=0.99):
        """
        Dynamically generate tab items based on what can be detected by
        template matching. Must be run after init (see client init) because it
        uses the container system from config, which is not available at init.
        """

        items = dict()
        cx1, cy1, cx2, cy2 = self.get_bbox()

        # TODO: tabs may be unavailable e.g. were're on the login screen, or
        #       we're on tutorial island and some tabs are disabled,
        #       or we're in the bank

        # TODO: add key bindings, so tabs can be opened/closed with F-keys
        #       (or RuneLite key bindings)

        tabs = list()
        for tab in self.STATIC_TABS:
            templates = list()

            template = self.templates.get(tab)
            if template is None:
                continue
            templates.append((tab, template))

            for permutation in self.PERMUTATIONS:
                name = f'{tab}_{permutation}'
                template = self.templates.get(name)
                if template is None:
                    continue
                templates.append((name, template))

            tabs.append((tab, templates))
        for tab, types in self.MUTABLE_TABS.items():

            templates = list()
            for type_ in types:
                tab = f'{tab}_{type_}'
                template = self.templates.get(tab)

                if template is None:
                    continue
                templates.append((tab, template))

                for permutation in self.PERMUTATIONS:
                    name = f'{tab}_{type_}_{permutation}'
                    template = self.templates.get(name)
                    if template is None:
                        continue
                    templates.append((name, template))

            tabs.append((tab, templates))

        for tab, templates in tabs:

            cur_confidence = -float('inf')
            cur_x = cur_y = cur_h = cur_w = None
            cur_template_name = ''
            confidences = list()
            for template_name, template in templates:
                match = cv2.matchTemplate(
                    self.img, template, cv2.TM_CCOEFF_NORMED,
                    mask=self.masks.get(template_name),
                )
                _, confidence, _, (x, y) = cv2.minMaxLoc(match)

                # log confidence for later
                confidences.append(f'{template_name}: {confidence:.3f}')

                if confidence > cur_confidence and confidence >= threshold:
                    cur_confidence = confidence
                    cur_x = x
                    cur_y = y
                    cur_h, cur_w = template.shape
                    cur_template_name = template_name

            selected = 'selected' in cur_template_name

            if None in {cur_x, cur_y, cur_h, cur_w}:
                continue

            self.logger.debug(
                f'{tab}: '
                f'chosen: {cur_template_name}, '
                f'selected: {selected}, '
                f'confidence: {confidences}'
            )

            x1, y1, x2, y2 = cur_x, cur_y, cur_x + cur_w - 1, cur_y + cur_h - 1
            # convert back to screen space so we can set global bbox
            sx1 = x1 + cx1 - 1
            sy1 = y1 + cy1 - 1
            sx2 = x2 + cx1 - 1
            sy2 = y2 + cy1 - 1

            # create dynamic tab item
            item = self.widget_class(
                tab, self.client, self, selected=selected)
            item.set_aoi(sx1, sy1, sx2, sy2)
            item.load_templates([t for t, _ in templates])
            item.load_masks(self.widget_masks(tab))

            # cache it to dict and add as class attribute for named access
            items[tab] = item
            setattr(self, tab, item)

            if selected:
                self.active_tab = item

        self._tabs = items
        return items

    def _click(self, *args, **kwargs):
        self.logger.warning('Do not click container, click the tabs.')

    def update(self):
        """
        Run update on each of the tab items.
        Note, it does not update click timeouts, as this class should not be
        clicked directly (attempting to do so throws a warning).
        """

        # set to none on each update,
        # if we find an active one it should be replaced
        self.active_tab = None

        for tab in self._tabs.values():
            tab.update()

            if tab.selected:
                self.active_tab = tab
