from glob import glob
from os.path import exists, basename
from typing import Union, List

import cv2
import numpy

from .game_objects import GameObject
from ..file_path_utils import get_root


class PersonalMenu(GameObject):
    """
    The right-side menu with options that affect your personal character.
    For example, inventory, magic or logout menus
    """

    (
        COMBAT,
        SKILLS,
        QUESTS,
        INVENTORY,
        EQUIPMENT,
        PRAYER,
        MAGIC,
        FRIENDS,
        ACCOUNT,
        GROUPS,
        SETTINGS,
        EMOTES,
        MUSIC,
        LOGOUT,
        WORLD_SWITCHER,
    ) = range(15)

    def __init__(self, client):
        super(PersonalMenu, self).__init__(
            client, client, config_path='personal_menu',
            container_name='personal_menu'
        )
        self._context = None
        self._menus = self.create_menus()

    def create_menus(self):
        menus = dict()

        menus[self.COMBAT] = None
        menus[self.SKILLS] = None
        menus[self.QUESTS] = None
        # TODO: refactor to inherit from GameObject
        menus[self.INVENTORY] = self.client.inventory
        menus[self.EQUIPMENT] = None
        menus[self.PRAYER] = None
        menus[self.MAGIC] = None
        menus[self.FRIENDS] = None
        menus[self.ACCOUNT] = None
        menus[self.GROUPS] = None
        menus[self.SETTINGS] = None
        menus[self.EMOTES] = None
        menus[self.MUSIC] = None
        menus[self.LOGOUT] = LogoutMenu(self.client, self)
        menus[self.WORLD_SWITCHER] = WorldSwitcherMenu(self.client, self)

        return menus

    def get_menu(self, enum):
        return self._menus.get(enum)

    def toggle_context(self, new_context):
        # clicking a tab / context button while on a different menu just
        # switches to the new menu
        if new_context != self._context:
            self._context = new_context
        # if the menu is already open, then it closes the menu
        else:
            self._context = None


class Inventory(object):

    SLOTS_HORIZONTAL = 4
    SLOTS_VERTICAL = 7

    def __init__(self, client):
        self.client = client
        self.config = client.config['inventory']
        self.slots = self._create_slots()
        # remove this once refactored to GameObject class
        self.containers = dict()

    @property
    def width(self):
        return self.config['width']

    @property
    def height(self):
        return self.config['height']

    # TODO: Convert to GameObject subclass!!

    def _eval_config_value(self, value):
        return eval(str(value))

    @property
    def margin_top(self):
        val = self.config.get('margins', {}).get('top', 0)
        return self._eval_config_value(val)

    @property
    def margin_bottom(self):
        val = self.config.get('margins', {}).get('bottom', 0)
        return self._eval_config_value(val)

    @property
    def margin_left(self):
        val = self.config.get('margins', {}).get('left', 0)
        return self._eval_config_value(val)

    @property
    def margin_right(self):
        val = self.config.get('margins', {}).get('right', 0)
        return self._eval_config_value(val)

    @property
    def padding_top(self):
        val = self.config.get('padding', {}).get('top', 0)
        return self._eval_config_value(val)

    @property
    def padding_bottom(self):
        val = self.config.get('padding', {}).get('bottom', 0)
        return self._eval_config_value(val)

    @property
    def padding_left(self):
        val = self.config.get('padding', {}).get('left', 0)
        return self._eval_config_value(val)

    @property
    def padding_right(self):
        val = self.config.get('padding', {}).get('right', 0)
        return self._eval_config_value(val)

    def get_bbox(self):
        if self.client.name == 'RuneLite':

            cli_bbox = self.client.get_bbox()
            client_x2 = cli_bbox[2]
            client_y2 = cli_bbox[3]
            right_margin = self.client.config['margins']['right']
            bottom_margin = self.client.config['margins']['bottom']
            tab_height = self.client.tabs.height

            x1 = client_x2 - right_margin - self.width
            y1 = client_y2 - bottom_margin - tab_height - self.height

            x2 = x1 + self.width
            y2 = y1 + self.height
        else:
            raise NotImplementedError

        return x1, y1, x2, y2

    def _create_slots(self):
        """
        Create a set of empty slots
        :return: List of slots
        :rtype: list
        """
        slots = list()
        for i in range(self.SLOTS_HORIZONTAL * self.SLOTS_VERTICAL):
            slots.append(None)

        return slots

    def set_slot(self, idx, template_names=None):
        """
        Setup a slot object at provided index with provided template names
        :param idx: Index for the new slot
        :param template_names: List of template names the slot should load
        :return: new Slot object
        """
        slot = Slot(idx, self.client, self, template_names)
        self.slots[idx] = slot

        return slot

    def identify(self, img, threshold=None):
        """
        Runs identification on each slot in the inventory
        :param img: Screen grab of the whole client
        :return: List of items identified
        """

        # we need client bbox to zero the slot coordinates
        x, y, _, _ = self.client.get_bbox()

        items = list()
        for slot in self.slots:

            x1, y1, x2, y2 = slot.get_bbox()
            # numpy arrays are stored rows x columns, so flip x and y
            slot_img = img[y1 - y:y2 - y, x1 - x:x2 - x]

            name = slot.identify(slot_img, threshold=threshold)
            items.append(name)

        return items

    def contains(self, item_names):
        """
        Convenience function to test if any slot contains any of the items
        provided by name
        :param item_names: Any iterable, containing item names to test for.
            Will support None as well.
        :return: True if inventory contains any of the items, false False
        """
        for slot in self.slots:
            if slot.contents in item_names:
                return True

        return False

    def first(self, item_names, order=1, clicked=None):
        """
        Get the first inventory item that matches the provided filters
        :param set item_names: Options for items
        :param int order: Must be 1 or -1 for forward or reverse order
        :param bool clicked: If True, return the first clicked inventory slot,
            else if False return the first unclicked inventory slot
        :return: Slot matching filters or None if no matches
        """
        for slot in self.slots[::order]:

            if clicked is not None:
                if clicked and not slot.clicked:
                    continue
                elif not clicked and slot.clicked:
                    continue

            if slot.contents in item_names:
                return slot

    def filter_slots(self, item_names):
        slots = list()
        for slot in self.slots:
            if slot.contents in item_names:
                slots.append(slot)
        return slots


# TODO: fix this god awful mess
class SlotMixin:

    def get_bbox(self):

        if self._bbox:
            return self._bbox

        if self.client.name == 'RuneLite':
            col = self.idx % self.parent.SLOTS_HORIZONTAL
            row = self.idx // self.parent.SLOTS_HORIZONTAL

            inv_bbox = self.parent.get_bbox()
            inv_x1 = inv_bbox[0]
            inv_y1 = inv_bbox[1]

            inv_x_margin = self.parent.config['margins']['left']
            inv_y_margin = self.parent.config['margins']['top']

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

    def identify(self, img, threshold=None):
        """
        Compare incoming image with templates and try to find a match
        :param img:
        :return:
        """

        if not self.templates:
            print(f'Slot {self.idx}: no templates loaded, cannot identify')
            return False

        img = self.process_img(img)

        max_match = None
        matched_item = None
        for name, template in self.templates.items():
            match = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)[0][0]

            if max_match is None:
                max_match = match
                matched_item = name
            elif match > max_match:
                max_match = match
                matched_item = name

        threshold = threshold or 0.8
        if max_match and max_match > threshold:
            self.contents = matched_item
        # TODO: test for unknown items (i.e. slot is not empty)
        else:
            self.contents = None

        return self.contents


class Slot(SlotMixin, GameObject):

    PATH_TEMPLATE = '{root}/data/inventory/{index}/{name}.npy'

    def __init__(self, idx, client, parent, template_names):
        self.idx = idx
        super(Slot, self).__init__(
            client, parent,
            template_names=template_names,
            config_path='inventory.slots',
        )

        self._bbox = None
        self.contents = None

    def resolve_path(self, **kwargs):
        return self.PATH_TEMPLATE.format(**kwargs)

    def _load_images(self, target, names: Union[List[str], None] = None,
                     cache: bool = True):
        """
        Internal helper method to load images onto a container. Currently only
        supports templates and masks. This method differs from base class
        implementation because images are stored in indexed folders.

        :param target: Name of the container to add images to
            (templates or mask only)
        :param names: File names of the images without extension. If names is
            None then no images will be loaded, but the empty container may
            still be cached.
        :param cache: If set to True, the container will be set to an
            internal pluralised attribute e.g. self._masks or self._templates

        :return: Dictionary of images loaded

        """

        container = dict()
        if target not in {'mask', 'template'}:
            raise ValueError(f'Unsupported image container: {target}')

        mask = ''
        if target == 'mask':
            mask = '_mask'

        if names is None:
            if cache:
                setattr(self, f'_{target}s', container)
            return container

        names = names or list()
        if not names:
            glob_path = self.resolve_path(
                root=get_root(),
                index='*',
                name=f'*{mask}',
            )

            paths = glob(glob_path)
            names = [basename(p).replace('.npy', '') for p in paths]

        for name in names:
            path = self.resolve_path(
                root=get_root(),
                index=self.idx,
                name=name,
            )
            if exists(path):
                image = numpy.load(path)
                container[name] = image

        if cache:
            setattr(self, f'_{target}s', container)
        return container

    def load_templates(self, names: Union[List[str], None] = None,
                       cache: bool = True):
        """
        Load template data from disk
        :param names: List of names to attempt to load from disk
        :param cache: If True, templates will be cached to instance.
        :return: Dictionary of templates of format {<name>: <numpy array>}
        """
        return self._load_images('template', names=names, cache=cache)

    def load_masks(self, names: Union[List[str], None] = None,
                   cache: bool = True):
        """
        Load masks data from disk
        :param names: List of names to attempt to load from disk
        :param cache: If True, masks will be cached to instance.
        :return: Dictionary of masks of format {<name>: <numpy array>}
        """
        return self._load_images('mask', names=names, cache=cache)


class Magic(Inventory):

    SLOTS_HORIZONTAL = 5
    SLOTS_VERTICAL = 9

    def __init__(self, client, parent, spellbook=None):
        super(Magic, self).__init__(client)
        self.parent = parent
        self.spellbook = spellbook
        self.config = client.config[spellbook]
        self.slots = self._create_slots()

    def set_slot(self, idx, template_names):
        """
        Setup a slot object at provided index with provided template names
        :param idx: Index for the new slot
        :param template_names: List of template names the slot should load
        :return: new Slot object
        """
        slot = SpellSlot(idx, self.client, self, template_names)
        self.slots[idx] = slot

        return slot


class SpellSlot(SlotMixin, GameObject):

    PATH_TEMPLATE = '{root}/data/magic/{spellbook}/{name}.npy'

    SPELL_NAMES = {
        'lunar': [
            'lunar_home_teleport', 'bake_pie', 'geomancy', 'cure_plant', 'monster_examine',
            'npc_contact', 'cure_other', 'humidify', 'moonclan_teleport', 'tele_group_moonclan',
            'cure_me', 'ourania_telport', 'hunter_kit', 'waterbirth_telport', 'tele_group_waterbirth',
            'cure_group', 'stat_spy', 'barbarian_teleport', 'tele_group_barbarian', 'spin_flax',
            'superglass_make', 'tan_leather', # TODO: rest of the spellbook
        ]
    }

    def __init__(self, idx, client, parent, template_names):
        self.idx = idx
        super(SpellSlot, self).__init__(
            client, parent,
            template_names=template_names,
            config_path=f'{parent.spellbook}.slots',
        )

        self._bbox = None
        self.contents = None

    @property
    def name(self):
        return self.SPELL_NAMES[self.parent.spellbook][self.idx]

    def load_templates(self, names=None, cache=True):
        templates = dict()
        path = self.resolve_path(
            root=get_root()
        )

        if exists(path):
            template = numpy.load(path)
            templates[self.name] = template

        if cache:
            self._templates = templates
        return templates

    def resolve_path(self, **kwargs):
        kwargs['spellbook'] = self.parent.spellbook
        kwargs['name'] = self.name
        return self.PATH_TEMPLATE.format(**kwargs)


class LogoutButton(GameObject):

    def __init__(self, client, parent):
        super(LogoutButton, self).__init__(
            client, parent, config_path='minimap.logout',
            container_name='logout',
        )

    @property
    def clickable(self):
        # TODO: if bank is open, return False
        return True


class WorldSwitcherMenu(GameObject):

    def __init__(self, client, parent):
        self.logout_button = WorldSwitcherMenuLogoutButton(client, self)
        super(WorldSwitcherMenu, self).__init__(
            client, parent, config_path='personal_menu.world_switcher',
            container_name=PersonalMenu.WORLD_SWITCHER
        )

    def setup_containers(self):
        containers = dict()

        containers['exit_buttons'] = {
            'y': [self.logout_button]
        }

        return containers


class WorldSwitcherMenuLogoutButton(GameObject):

    PATH_TEMPLATE = '{root}/data/pmenu/world_switcher/{name}.npy'

    def __init__(self, client, parent):
        super(WorldSwitcherMenuLogoutButton, self).__init__(
            client, parent, config_path='personal_menu.world_switcher.logout',
            container_name='exit_buttons',
            template_names=['logout', 'logout_hover'],
        )


class LogoutMenu(GameObject):

    def __init__(self, client, parent):
        self.logout_button = LogoutMenuLogoutButton(client, self)
        super(LogoutMenu, self).__init__(
            client, parent, config_path='personal_menu.logout',
            container_name=PersonalMenu.LOGOUT
        )

    def setup_containers(self):
        containers = dict()

        containers['exit_buttons'] = {
            'y': [self.logout_button]
        }

        return containers


class LogoutMenuLogoutButton(GameObject):

    PATH_TEMPLATE = '{root}/data/pmenu/logout/{name}.npy'

    def __init__(self, client, parent):
        super(LogoutMenuLogoutButton, self).__init__(
            client, parent, config_path='personal_menu.logout.logout',
            container_name='exit_buttons',
            template_names=['logout', 'logout_hover'],
        )
