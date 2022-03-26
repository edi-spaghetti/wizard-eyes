import cv2

from .widget import TabItem
from ..game_objects import GameObject


class Tabs(GameObject):
    """Container for the main screen tabs."""

    PATH_TEMPLATE = '{root}/data/tabs/{name}.npy'
    STATIC_TABS = [
        'combat',
        'stats',
        'inventory',
        'equipment',
        'prayer',
    ]

    MUTABLE_TABS = {
        'spellbook': ['standard', 'ancient', 'lunar', 'arceuus'],
        'influence': ['quests']
    }

    def __init__(self, client):

        # set up templates with defaults & selected modifiers
        template_names = (
                self.STATIC_TABS +
                [f'{t}_selected' for t in self.STATIC_TABS])
        for name, types in self.MUTABLE_TABS.items():
            for type_ in types:
                template = f'{name}_{type_}'
                selected = f'{name}_{type_}_selected'
                template_names.append(template)
                template_names.append(selected)

        super(Tabs, self).__init__(
            client, client, config_path='tabs',
            container_name='personal_menu',
            template_names=template_names,
        )

        # load in the default tab mask
        self.load_masks(['tab'], cache=True)

        # dynamically build tab items based on what can be found
        self.active_tab = None
        self._tabs = None

        # add in placeholders for the tabs we expect to find (this will
        # helper the linter)
        # TODO: handle mutable tabs e.g. quests/achievement diary or spellbooks
        self.combat = None
        self.stats = None
        self.equipment = None
        self.quests = None
        self.inventory = None
        self.prayer = None
        self.spellbook_arceuus = None

    @property
    def width(self):
        # TODO: double tab stack if client width below threshold
        return self.config['width'] * 13

    @property
    def height(self):
        # TODO: double tab stack if client width below threshold
        return self.config['height'] * 1

    def build_tab_items(self):
        """
        Dynamically generate tab items based on what can be detected by
        template matching. Must be run after init (see client init) because it
        uses the container system from config, which is not available at init.
        """

        items = dict()
        cx1, cy1, cx2, cy2 = self.get_bbox()

        # TODO: tabs may be unavailable e.g. were're on the login screen, or
        #  we're on tutorial island and some tabs are disabled.

        # TODO: add key bindings, so tabs can be opened/closed with F-keys
        #  (or RuneLite key bindings)

        tabs = list()
        for tab in self.STATIC_TABS:
            templates = list()

            template = self.templates.get(tab)
            if template is None:
                continue
            templates.append((tab, template))

            name = f'{tab}_selected'
            selected = self.templates.get(name)
            if selected is None:
                continue
            templates.append((name, selected))

            tabs.append((tab, templates))
        for group, names in self.MUTABLE_TABS.items():

            templates = list()
            for tab in names:
                tab = f'{group}_{tab}'
                template = self.templates.get(tab)

                if template is None:
                    continue
                templates.append((tab, template))

                name = f'{tab}_selected'
                selected = self.templates.get(name)
                if selected is None:
                    continue
                templates.append((name, selected))

            tabs.append((group, templates))

        for tab, templates in tabs:

            cur_confidence = -float('inf')
            cur_x = cur_y = cur_h = cur_w = None
            cur_template_name = ''
            confidences = list()
            for template_name, template in templates:
                match = cv2.matchTemplate(
                    self.img, template, cv2.TM_CCOEFF_NORMED,
                    mask=self.masks.get('tab'),
                )
                _, confidence, _, (x, y) = cv2.minMaxLoc(match)

                # log confidence for later
                confidences.append(f'{template_name}: {confidence:.3f}')

                if confidence > cur_confidence:
                    cur_confidence = confidence
                    cur_x = x
                    cur_y = y
                    cur_h, cur_w = template.shape
                    cur_template_name = template_name

            selected = cur_template_name.endswith('selected')

            if None in {cur_x, cur_y, cur_h, cur_w}:
                continue

            self.logger.info(
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
            item = TabItem(tab, self.client, self, selected=selected)
            item.set_aoi(sx1, sy1, sx2, sy2)
            item.load_templates([t for t, _ in templates])
            item.load_masks(['tab'])

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

        for tab in self._tabs.values():
            tab.update()

            if tab.selected:
                self.active_tab = tab
