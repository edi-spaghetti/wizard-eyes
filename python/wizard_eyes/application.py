import sys
import time
import argparse
from random import random
from os import _exit, makedirs
from os.path import dirname, join
from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Callable
import re
from uuid import uuid4

import cv2
import numpy
import keyboard

from .client import Client
from .file_path_utils import get_root, load_pickle
from .game_objects.game_objects import GameObject
from .game_objects.minimap.gps import Map
from .game_entities.entity import GameEntity
from .dynamic_menus.widget import AbstractWidget
from .script_utils import int_or_str


class Application(ABC):
    """Base class application with methods for implementation."""

    PATH = (
        f'{get_root()}/data/recordings/{{folder}}/{{index}}_{{timestamp}}.png'
    )
    INVENTORY_TEMPLATES = None
    EQUIPMENT_TEMPLATES = None
    BANK_TEMPLATES = None
    SPELLBOOK_TEMPLATES = None
    PRAYER_TEMPLATES = None

    SKIP_PARSE_ARGS = False
    DEFAULT_MAP_SWAP_RANGE = 1
    """int: Default xax tile distance from target node on swapping the map
    for a teleport. Distance must be LESS THAN this value."""

    @property
    def client_init_args(self):
        return 'RuneLite',

    @property
    def client_init_kwargs(self):
        return {}

    def __init__(self, msg_length=100):
        self.continue_ = True
        self.saving_and_exiting = False
        self.client: Client = Client(
            *self.client_init_args, **self.client_init_kwargs
        )
        self.sleeping = not self.client.screen.on_off_state()
        # WARNING: do not call any game object's bounding box method before
        #          post init (including debugger breakpoints) as this will
        #          prematurely cache their bbox value and likely be incorrect.
        self.client.post_init()
        self.msg: List[str] = list()
        self.msg_length: int = msg_length
        self.msg_buffer = list()
        self.frame_number: int = 0
        self.afk_timer: GameObject = GameObject(self.client, self.client)
        self.target: Union[Callable, None] = None
        self.target_xy: Union[Tuple[int, int], None] = None
        self.swap_confidence: Union[float, None] = None

        self.parser: Union[argparse.ArgumentParser, None] = None
        self.args: Union[argparse.Namespace, None] = None
        if not self.SKIP_PARSE_ARGS:
            self.create_parser()
            self.parse_args()

        # set up callback for immediate exit of application
        keyboard.add_hotkey(self.exit_key, self.exit)
        self.client.logger.warning(f'Exit hotkey: {self.exit_key}')

        self.images = list()
        keyboard.add_hotkey(self.save_key, self.save_and_exit)
        self.client.logger.warning(f'Save & exit: {self.save_key}')

        # p for pause
        keyboard.add_hotkey(self.pause_key, self.toggle_sleep)
        self.client.logger.warning(
            f'Pause (with caps/num lock active): {self.pause_key}')

    def toggle_sleep(self):
        """Toggling this value will make the application continue or pause."""
        if not self.client.screen.on_off_state():
            self.client.logger.warning('Caps lock must be off to toggle sleep')
            return

        self.sleeping = not self.sleeping
        self.client.logger.warning(f'Sleeping state: {self.sleeping}')

    @property
    def exit_key(self):
        """
        Hotkey combination used by the keyboard module to set up a callback.
        On triggering, the application will immediately call
        :meth:`Application.exit`
        """
        return 'shift+esc'

    @property
    def buffer(self):
        """
        Set a limit on the maximum number of frames the application can
        hold in memory.
        """
        return 100

    @property
    def pause_key(self):
        """Hotkey for pausing the application. Must be used in combination
        with caps or num lock. This allows the user to type without
        accidentally triggering the pause function."""
        return 'p'

    @property
    def save_key(self):
        """
        Hotkey combination used by the keyboard module to set up a callback.
        On triggering, the application will immediately call
        :meth:`Application.save_and_exit`.
        Note, images will only be saved to disk if they are being buffered,
        which requires the command line params be set.
        """
        return 'ctrl+u'

    def save_and_exit(self):
        """
        Save client images to disk if configured to do so.
        Note, if showing the images, they may be annotated.

        todo: handle the main thread exiting before images have been saved
        """
        print('Saving ...')
        # TODO: manage folder creation

        # stop the event loop so we're not still adding to the buffer
        self.saving_and_exiting = True
        unique_folder = uuid4().hex
        containing_folder = dirname(dirname(self.PATH))
        new_folder_path = join(containing_folder, unique_folder)
        makedirs(new_folder_path)

        for i, (timestamp, image) in enumerate(self.images):
            self.client.logger.info(
                f'Saving image {i + 1} of {len(self.images)}')
            path = self.PATH.format(
                folder=unique_folder, index=i, timestamp=timestamp)
            self.client.screen.save_img(image, path)
        print(f'Saved to: {new_folder_path}')
        self.continue_ = False
        self.exit()

    def exit(self):
        """
        Shut down the application while still running without getting
        threadlock from open cv2.imshow calls.
        """
        print('Exiting ...')
        cv2.destroyAllWindows()
        _exit(1)

    def add_default_start_xy_arg(self, parser):
        parser.add_argument(
            '--start-xy', type=int_or_str,
            nargs=2,
            help='Specify starting coordinates by <x,y> or <map,label>'
        )

    def add_default_map_name_arg(self, parser):
        parser.add_argument(
            '--map-name',
            help='Optionally specify starting map '
                 '(required if start-xy from non-named coordinates)'
        )

    def add_default_runtime_arg(self, parser):
        parser.add_argument(
            '--run-time', type=int, default=float('inf'),
            help='set a maximum run time for the script in seconds'
        )

    def create_parser(self):
        """"""
        parser = argparse.ArgumentParser()

        self.add_default_start_xy_arg(parser)
        self.add_default_map_name_arg(parser)
        self.add_default_runtime_arg(parser)

        self.parser = parser
        return parser

    def parse_args(self):
        args, _ = self.parser.parse_known_args()
        self.args = args

        # post parse start xy - resolve map and label if provided.
        gps = self.client.minimap.minimap.gps

        try:
            a, b = self.args.start_xy
        except TypeError:
            self.client.logger.warning('No start xy on init')
            return args

        if isinstance(a, int) and isinstance(b, int):
            self.args.start_xy = (a, b)
            if self.args.map_name is None:
                self.parser.error(
                    'Map name required if start-xy are not named'
                )
            else:
                try:
                    gps.load_map(self.args.map_name)
                except FileNotFoundError:
                    self.client.logger.warning(
                        f'Cannot load map on post-parse args: '
                        f'{self.args.map_name}')
        elif isinstance(a, str) and isinstance(b, str):
            try:
                gps.load_map(a)
                node = gps.current_map.label_to_node(b).pop()
                self.args.start_xy = node
            except FileNotFoundError:
                self.client.logger.warning(
                    f'Cannot load map on post-parse args: '
                    f'{self.args.map_name}')
        else:
            raise NotImplementedError(
                f'Expected (x,y) or (map,label) - got {a, b}'
            )

        gps.set_coordinates(*self.args.start_xy)

        return args

    def create_custom_map(self, map_name, klass):
        """Create a custom map class instance in the same way a class would
        normally be loaded with
        :meth:`wizard_eyes.game_objects.minimap.GielenorPositionSystem.load-map`.

        :param str map_name: Internal name of the map, used to store on GPS
            maps dict.
        :param callable klass: Class definition that should inherit from the
            Maps base class and be able to initialise in the same way.
        """

        gps = self.client.minimap.minimap.gps

        path = gps.PATH_TEMPLATE.format(root=get_root(), name=map_name)
        data = load_pickle(path)

        chunks = data.get('chunks', {})
        graph = data.get('graph', {})
        labels = data.get('labels', {})
        offsets = data.get(
            'offsets', (Map.DEFAULT_OFFSET_X, Map.DEFAULT_OFFSET_Y))
        gps.maps[map_name] = klass(
            self.client,
            chunks,
            name=map_name,
            graph=graph,
            labels=labels,
            offsets=offsets,
        )
        # FIXME: parseargs changes start_xy from named coordinates to x,y
        #  coordinates, so this doesn't actually work if custom map is
        #  required on load
        if self.args.start_xy[0] == map_name:
            gps.current_map = gps.maps[map_name]
            gps.current_map.copy_original()

    def _setup_game_entity(
            self, label, map_=None, count=1
    ) -> Union[List[GameEntity], GameEntity]:
        """
        Helper method to set up any arbitrary game entity based on map nodes.
        """

        mm = self.client.minimap.minimap
        map_ = map_ or mm.gps.current_map

        nodes = map_.find(label=label)
        entities = list()
        for x, y in nodes:
            if len(entities) >= count:
                continue

            meta = map_.get_meta(label)
            data = meta.get('nodes', {}).get((x, y), {})

            key = (int((x - self.args.start_xy[0]) * mm.tile_size),
                   int((y - self.args.start_xy[1]) * mm.tile_size))

            width = data.get('width', 1)
            height = data.get('height', 1)

            entity = self.client.game_screen.create_game_entity(
                label, label, key, self.client, self.client,
                tile_width=width, tile_height=height,
            )
            entity.set_global_coordinates(x, y)
            if count == 1:
                return entity
            else:
                entities.append(entity)

        return entities

    @abstractmethod
    def setup(self):
        """
        Run any of the application setup required *before* entering the main
        event loop.
        """

    def _update_game_entities(self, *entities, mapping=None):
        """
        Update a list of game entities relative to player.
        Note, GPS must have been run this cycle or the key will be outdated.

        :param entities: List of game entity objects to update
        :param dict mapping: A mapping of entity name to map name. Used to
            determine if the entities provided should be checked, because
            they're on the current map and therefore positioned relative to
            player, or should be skipped because they're on a different map.
        """

        mm = self.client.minimap.minimap
        gps = self.client.minimap.minimap.gps

        for entity in entities:

            # skip entities not on the current map
            if mapping:
                map_name = mapping.get(entity.name)
                if map_name != gps.current_map.name:
                    continue

            x, y = entity.get_global_coordinates()
            px, py = gps.get_coordinates(real=True)
            key = (x - px) * mm.tile_size, (y - py) * mm.tile_size
            entity.update(key=key)

    def _add_afk_timeout(self, min_, max_):
        self.afk_timer.add_timeout(
            self.client.TICK * min_
            + random() * self.client.TICK * max_
        )

    def inventory_icons_loaded(self):
        """
        Common method to check inventory icons have been loaded before we do
        other actions.
        :return: True if all inventory icons slots have been loaded.
        """

        inv = self.client.tabs.inventory
        at = self.client.tabs.active_tab

        if len(inv.interface.icons) < self.client.INVENTORY_SIZE:
            # if inventory is open we can locate as normal
            # if inventory is disabled it means we're in the bank,
            # so the inventory is open anyway
            if at is inv or inv.state == 'disabled':
                inv.interface.locate_icons({
                    'item': {
                        'templates': self.INVENTORY_TEMPLATES,
                        'quantity': self.client.INVENTORY_SIZE},
                })

        return len(inv.interface.icons) == self.client.INVENTORY_SIZE

    def equipment_icons_loaded(self, cache=False):
        """
        Common method to check icons in the equipment menu have been loaded.

        :param bool cache: If true, found equipment icons will be cached to
            the app class under the same name they were defined. For exmaple.
            if EQUIPMENT_TEMPLATES contains 'rune_scimitar' then the icon
            will be added to the class (if found), accessible by
            self.rune_scimitar.

        :return: True if all equipment icons slots have been loaded.
        """

        eq = self.client.tabs.equipment
        at = self.client.tabs.active_tab

        if len(eq.interface.icons) < len(self.EQUIPMENT_TEMPLATES):
            # if equipment tab is open, attempt to locate each piece of
            # equipment one at a time. They should be unique, so quantity 1.
            if at is eq:
                for equipment in self.EQUIPMENT_TEMPLATES:
                    eq.interface.locate_icons({
                        'item': {
                            'templates': [equipment],
                            'quantity': 1
                        },
                    }, update=True)

                    # optionally cache the found game object to app class
                    if cache:
                        icon = eq.interface.icons_by_state(equipment)
                        if icon:
                            setattr(self, equipment, icon[0])

        return len(eq.interface.icons) == len(self.EQUIPMENT_TEMPLATES)

    def bank_icons_loaded(self, cache=False):
        """
        Common method to check icons in the bank menu. Currently only supports
        the top level bank.interface, rather than specific bank tabs, because
        we currently have no way of determining the currently open tab.

        :param bool cache: If true, found icons will be cache to the app class
            under the same name they were defined.

        :return: True if all bank icons have been loaded.

        """

        bt = self.client.bank
        inv = self.client.tabs.inventory

        if len(bt.interface.icons) < len(self.BANK_TEMPLATES):
            if inv.state == 'disabled':

                for item in self.BANK_TEMPLATES:

                    bt.interface.locate_icons({
                        'bank_item': {
                            'templates': [item],
                            'quantity': 1,
                        }
                    }, update=True)

                    # optionally cache the found game object to app class
                    if cache:
                        icon = bt.interface.icons_by_state(item)
                        if icon:
                            setattr(self, item, icon[0])

                    # slight offset to clickbox to avoid runelite inventory
                    # tabs being clicked when on very edge pixel of bank item
                    icon = bt.interface.icons_by_state(item)
                    if icon:
                        icon[0].y1_offset = 3
                        icon[0].y2_offset = -3
                        icon[0].x1_offset = 3
                        icon[0].x2_offset = -3

        return len(bt.interface.icons) == len(self.BANK_TEMPLATES)

    def spellbook_icons_loaded(self, cache=False):
        """
        Common method to check icons in the spellbook menu have been loaded.

        :param bool cache: If true, found spell icons will be cached to
            the app class under the same name they were defined.

        :return: True if all spell icons slots have been loaded.
        """

        sb = self.client.tabs.spellbook
        at = self.client.tabs.active_tab

        if len(sb.interface.icons) < len(self.SPELLBOOK_TEMPLATES):
            # if spellbook tab is open, attempt to locate each spell slot,
            # one at a time. They should be unique, so quantity 1.
            if at is sb:
                for spell in self.SPELLBOOK_TEMPLATES:
                    sb.interface.locate_icons({
                        'spell': {
                            'templates': [spell],
                            'quantity': 1
                        },
                    }, update=True)

                    # optionally cache the found game object to app class
                    if cache:
                        icon = sb.interface.icons_by_state(spell)
                        if icon:
                            setattr(self, spell, icon[0])

        return len(sb.interface.icons) == len(self.SPELLBOOK_TEMPLATES)

    def prayer_icons_loaded(self, cache=False):
        """
        Common method to check icons in the prayer menu have been loaded.

        :param bool cache: If true, found prayer icons will be cached to
            the app class under the same name they were defined.

        :return: True if all prayer icons slots have been loaded.
        """

        p = self.client.tabs.prayer
        at = self.client.tabs.active_tab

        if len(p.interface.icons) < len(self.PRAYER_TEMPLATES):
            # if prayer tab is open, attempt to locate each prayer slot,
            # one at a time. They should be unique, so quantity 1.
            if at is p:
                for prayer in self.PRAYER_TEMPLATES:
                    p.interface.locate_icons({
                        'prayer': {
                            'templates': [prayer],
                            'quantity': 1
                        },
                    }, update=True)

                    # optionally cache the found game object to app class
                    if cache:
                        icon = p.interface.icons_by_state(prayer)
                        if icon:
                            setattr(self, prayer, icon[0])

        return len(p.interface.icons) == len(self.PRAYER_TEMPLATES)

    @abstractmethod
    def update(self):
        """
        Update things like internal state, as well as run the update methods
        of any game objects that are required.
        """

    def set_target(self, entity, x, y, method=None):

        if not isinstance(entity, GameEntity):
            return

        x1, y1, _, _ = entity.get_bbox()
        if method is None:
            method = entity.get_bbox
        x1, y1, _, _ = method()

        rx = x - x1
        ry = y - y1
        self.target = method
        self.target_xy = rx, ry

    def update_target(self):
        if self.target is None or self.target_xy is None:
            return

        # calculate where the target coordinates should be
        x1, y1, _, _ = self.target()
        rx, ry = self.target_xy
        tx = x1 + rx
        ty = y1 + ry

        # compare with actual current mouse position
        mx, my = self.client.screen.mouse_xy
        if (tx, ty) == (mx, my):
            return

        # ensure that the updated coordinates are valid
        if not self.client.game_screen.is_clickable(tx, ty, tx, ty):
            self.clear_target()
            return

        # otherwise update mouse position to target
        self.client.screen.mouse_to(tx, ty)

    def clear_target(self):
        self.target = None
        self.target_xy = None

    def _calculate_timeout(self, entity, action_timeout=0):
        """Calculate the timeout required when clicking a game entity.

        Usually when we click a game entity there are two components that
        determine how long the entire action should take. First we have to move
        the player to be in base contact with the object, then the action
        itself takes a certain amount of time.

        This function calculates the overall timeout, assuming running.

        :param wizard_eyes.game_entities.entity.GameEntity entity: The entity
            we want to click.
        :param int action_timeout: The time in seconds the action takes.

        :returns int: The total time to complete an action from clicking a
            game entity.

        """
        mm = self.client.minimap.minimap
        gps = self.client.minimap.minimap.gps

        dist = mm.distance_between(
            gps.get_coordinates(),
            entity.get_global_coordinates()
        )
        # divide 2 because we assume running
        # TODO: support walking/running detection
        dist_timeout = max([dist * (self.client.TICK / 2), self.client.TICK])

        return dist_timeout + action_timeout

    def _click_entity(self, entity, tmin, tmax, mouse_text, method=None,
                      delay=True, speed=1, multi=1):
        """
        Click a game entity safely, by asserting the mouse-over text matches.

        :param entity: Any game entity, assumed to be on screen.
        :param tmin: Timeout minimum to assign to click
        :param tmax: Timeout maximum to assign to click
        :param mouse_text: Text we're expecting to see when the mouse is
            hovering over the entity

        :returns: True if the entiy was clicked, else False.

        """
        mo = self.client.mouse_options

        if not entity.is_inside(*self.client.screen.mouse_xy, method=method):
            x, y = self.client.screen.mouse_to_object(entity, method=method)
            if x is None or y is None:
                return False
            self.set_target(entity, x, y, method=method)
            # give the game some time to update the new mouse options
            if delay:
                time.sleep(0.1)
                self.msg.append(f'Mouse to: {x, y}')
                return False
            else:
                x, y = entity.click(
                    tmin=tmin, tmax=tmax, bbox=False,
                    pause_before_click=True, speed=speed,
                    multi=multi,
                )
                self.clear_target()
                result = x is not None and y is not None
                self.msg.append(f'Clicked {entity}: {result}')
                return result

        elif re.match(mouse_text, mo.state):
            x, y = entity.click(tmin=tmin, tmax=tmax, bbox=False, multi=multi)
            self.clear_target()
            result = x is not None and y is not None
            self.msg.append(f'Clicked: {entity}: {result}')
            return result
        else:
            # move the mouse to another random position and
            # hope we cant find it. Usually this happens when the
            # mouse moves to a new position and the game client
            # doesn't update, the model has some gaps, or the tile estimation
            # is inaccurate.
            x, y = self.client.screen.mouse_to_object(entity, method=method)
            if x is None or y is None:
                return False
            self.set_target(entity, x, y, method=method)
            # give the game some time to update the new mouse options
            if delay:
                time.sleep(0.1)
            # TODO: right click to see if we can find it
            self.msg.append(f'{entity} occluded: {mo.state}')
            return False

    def _click_tab(self, tab: AbstractWidget):
        if tab.clicked:
            self.msg.append(f'Waiting {tab} menu')
        else:
            tab.click(tmin=0.1, tmax=0.2)
            self.msg.append(f'Clicked {tab} menu')

    def _right_click(self, item: GameObject, width: int = 200, items: int = 8,
                     cm_config: dict = None):
        """Right click a game object and create a context menu on it.

        :param GameObject item:
        :param dict cm_config:

        """

        x, y = item.right_click(
            tmin=0.6, tmax=0.9, pause_before_click=True)
        # TODO: tweak values for context menu config
        if not cm_config:
            cm_config = dict(margins=dict(
                top=20, bottom=5, left=5, right=5))
        item.set_context_menu(x, y, width, items, cm_config)
        self.msg.append(f'right clicked {item}')

        # add an afk timer so we don't *immediately* click
        # the menu option
        self.afk_timer.add_timeout(
            self.client.TICK + random())

    def _swap_map_from_item(
            self, item, map_, node, post_script=None,
            range_=DEFAULT_MAP_SWAP_RANGE, confidence=None):
        """Swap maps due to clicking an entity. It may be a right click menu
        or a left click on an object or game entity."""
        gps = self.client.minimap.minimap.gps
        mo = self.client.mouse_options
        mm = self.client.minimap.minimap
        confidence = confidence or 0.0

        if mo.state in {'loading', 'waiting'}:
            self.msg.append('waiting game load')
        else:

            cur_confidence = gps.confidence
            cur_node = gps.get_coordinates(real=True)
            cur_map = gps.current_map
            gps.load_map(map_, set_current=True)
            node = gps.current_map.label_to_node(node).pop()
            gps.set_coordinates(*node, add_history=False)
            pos = gps.update(auto=False, draw=False)
            new_confidence = gps.confidence
            self.swap_confidence = new_confidence
            is_none = pos[0] is None or pos[1] is None
            if is_none:
                dist = float('inf')
            else:
                dist = mm.distance_between(pos, node)
            if not is_none and dist < range_ and new_confidence > confidence:
                gps.clear_coordinate_history()
                self.msg.append(f'teleported to: {map_}: {node}')

                # optionally run extra code after the map swapped
                if callable(post_script):
                    post_script()

            else:
                # set gps back to where it was
                gps.load_map(cur_map.name, set_current=True)
                gps.set_coordinates(*cur_node, add_history=False)
                gps.confidence = cur_confidence
                self.msg.append(f'waiting {item} teleport')

    def _teleport_with_item(
            self, item, map_: str, node: str, idx: Union[int, None] = None,
            post_script=None, width: int = 200, items: int = 8, config=None,
            range_=DEFAULT_MAP_SWAP_RANGE, tmin=None, tmax=None,
            mouse_text=None, multi=1, confidence=None, method=None):
        """
        Teleport to a new map location with an object in inventory
        or equipment slot.

        :param GameObject item: game object that will be clicked
        :param str map_: Name of the map we are expecting to travel to
        :param str node: Name of the node label we expect to arrive at
        :param idx: If set, item is assumed to be a right click teleport.
            This parameter is the index of the context menu item we need to
            click on the right click menu
        :param post_script: Optionally provide a function that can be called
            with no parameters to be run after the teleport menu option has
            been clicked.
        :param int width: Width in pixels of the new context menu
        :param int items: Number of menu items to create in new menu
        :param config: Context menu config
        :param int range_: Teleport desitination may not be exact, specify the
            potential distance from arrival node in tiles.

        """

        if idx is None:
            if item.clicked:
                self._swap_map_from_item(
                    item, map_, node, post_script=post_script, range_=range_,
                    confidence=confidence)
            elif mouse_text:
                self._click_entity(
                    item, tmin, tmax, mouse_text, delay=True, multi=multi,
                    method=method
                )
            else:

                bbox = None
                if method:
                    bbox = method()

                # TODO: tweak timeouts on left click teleport
                item.click(tmin=tmin, tmax=tmax,
                           pause_before_click=True, multi=multi,
                           bbox=bbox)
                self.msg.append(f'clicked teleport to {map_}')

        elif item.context_menu:
            # TODO: find context menu items dynamically
            inf = item.context_menu.items[idx]

            if inf.clicked:
                self._swap_map_from_item(
                    inf, map_, node, post_script=post_script, range_=range_,
                    confidence=confidence)
            else:
                inf.click(tmin=float('inf'), tmax=float('inf'),
                          pause_before_click=True)
                self.msg.append(f'clicked teleport to {map_}')

        elif item.clicked:
            self.msg.append(
                f'Waiting {item} context menu')
        else:
            self._right_click(item, width=width, items=items, cm_config=config)

    def hop_worlds(self):
        """"""

        tabi = self.client.tabs.interface
        mo = self.client.mouse_options

        if tabi.clicked:
            if mo.state in mo.SYSTEM_TEMPLATES:
                self.msg.append('hopping worlds')
            else:
                self.msg.append('waiting hop worlds')
        else:
            self.client.screen.press_hotkey('ctrl', 'shift', 'left', delay=0.5)
            tabi.add_timeout(3)
            self.msg.append('Clicked hop worlds hotkey')

    @abstractmethod
    def action(self):
        """
        Perform an action (or not) depending on the current state of the
        application. It is advisable to limit actions to one per run cycle.
        """

    def run(self):
        # run
        print('Entering Main Loop')
        self.client.activate()
        while self.continue_:

            # put application in a holding pattern while we clean up
            if self.saving_and_exiting:
                time.sleep(0.1)
                continue

            self.frame_number += 1

            # set up logging for new cycle
            self.msg = list()
            t1 = time.time()

            # caps lock to pause the script
            # p to exit
            # TODO: convert these to utility functions
            if self.sleeping:
                msg = f'Sleeping @ {self.client.time}'
                sys.stdout.write('\b' * self.msg_length)
                sys.stdout.write(f'{msg:{self.msg_length}}')
                sys.stdout.flush()
                time.sleep(0.1)
                continue

            # ensure the client is updated every frame and run the
            # application's update method
            self.client.update()
            self.afk_timer.update()
            self.update()
            self.update_target()

            # do an action (or not, it's your life)
            if self.afk_timer.time_left:
                self.msg.append(f'AFK: {self.afk_timer.time_left:.3f}')
            else:
                self.action()

            # log run cycle
            t2 = time.time()  # not including show image time
            self.msg.insert(0, f'Cycle {self.frame_number} {t2 - t1:.3f}')
            msg = ' - '.join(self.msg)
            self.msg_buffer.append(msg)
            if len(self.msg_buffer) > 69:
                self.msg_buffer = self.msg_buffer[1:]  # remove oldest

            sys.stdout.write('\b' * self.msg_length)
            sys.stdout.write(f'{msg[:self.msg_length]:{self.msg_length}}')
            sys.stdout.flush()

            self.show()

    def show(self):
        """
        Show images per client args.
        """

        # if we're in static mode draw calls will overwrite out static image,
        # take a copy of it and swap the original back at the end.
        img = None
        if self.client.args.static_img:
            img = self.client._original_img
            self.client._original_img = img.copy()

        # all subscribed draw calls can now be executed
        for draw in self.client.draw_calls:
            draw()

        images = list()
        if self.client.args.show:
            name = 'Client'
            images.append((name, self.client.original_img))

        if self.client.args.show_map:
            name = 'Map'
            gps = self.client.minimap.minimap.gps
            if gps.current_map is not None:
                images.append((name, gps.current_map.img_colour))

        if self.client.args.save:
            self.images = self.images[:self.buffer - 1]
            self.images.append((self.client.time, self.client.original_img))

        if self.client.args.message_buffer:
            w, h = self.client.args.buffer_wh
            buffer = numpy.ones((h, w, 4), dtype=numpy.uint8)

            for i, msg in enumerate(self.msg_buffer, start=1):
                buffer = cv2.putText(
                    buffer, msg, (10, 10 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                    (50, 50, 50, 255), thickness=1)

            images.append(('Logs', buffer))

        if self.client.args.show_gps:
            name = 'Gielenor Positioning System'
            gps = self.client.minimap.minimap.gps
            if gps.current_map is not None:
                images.append((name, gps.show_img))

        if images:
            for i, (name, image) in enumerate(images):
                try:
                    cv2.imshow(name, image)
                except cv2.error as err:
                    self.client.game_screen.player.logger.error(
                        f'cannot show: {name}, {err}'
                    )
                    raise

                widths = [im.shape[1] for _, im in images[:i]]
                cv2.moveWindow(name, 5 + sum(widths), 20)
            cv2.waitKey(1)

        # set the original original image back for next loop update
        if self.client.args.static_img:
            self.client._original_img = img
