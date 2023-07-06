from abc import ABC, abstractmethod
import argparse
from os import makedirs
from os.path import dirname, join
from random import random, uniform
import re
import sys
import time
from typing import Union, List, Dict, SupportsIndex
from uuid import uuid4

import cv2
import numpy
import keyboard

from .client import Client
from .dynamic_menus.icon import AbstractIcon
from .dynamic_menus.widget import AbstractWidget
from .file_path_utils import get_root, load_pickle
from .game_entities.entity import GameEntity
from .game_entities.screen import ClickChecker
from .game_objects.game_objects import GameObject
from .game_objects.minimap.gps import Map
from .script_utils import int_or_str

import wizard_eyes.consumables


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
        self.targets: Dict[SupportsIndex, AbstractIcon] = {}
        # to keep track of objects we want to click
        self.consumables: List = []  # keep track of consumable inventory items
        # keep track of entities
        self.entities: List[GameEntity] = []
        self.entities_mapping: Dict[str, str] = {}
        """dict: Mapping of entity ids to the map they belong to."""
        self.swap_confidence: Union[float, None] = None
        self.click_checker: ClickChecker = ClickChecker(self.client)

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
        self.continue_ = False

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

    def create_parser(self) -> argparse.ArgumentParser:
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

    def register_consumable(
            self,
            consumable: 'wizard_eyes.consumables.AbstractConsumable'):
        """Associate the consumable with the application"""

        # keep a permanent record of consumable for reference later
        self.consumables.append(consumable)
        return consumable

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

        mapping = mapping or self.entities_mapping

        mm = self.client.minimap.minimap
        gps = self.client.minimap.minimap.gps

        for entity in entities:

            # skip entities not on the current map
            if mapping:
                map_name = mapping.get(entity.id)
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

    @abstractmethod
    def update(self):
        """
        Update things like internal state, as well as run the update methods
        of any game objects that are required.
        """

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
                      delay=True, speed=1, multi=1, click_check=True):
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
                self.msg.append('Cannot mouse off screen edge')
                return False
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
                if x is None or y is None:
                    entity.clear_timeout()
                    self.msg.append('Misclick on screen edge')
                    return False

                if click_check:
                    self.click_checker.start(
                        x, y, red=True, on_failure=entity.clear_timeout,
                        on_success=self.click_checker.reset,
                    )

                self.msg.append(f'Clicked {entity}')
                self.afk_timer.add_timeout(uniform(0.1, 0.2))
                return True

        elif re.search(mouse_text, mo.state):
            x, y = entity.click(tmin=tmin, tmax=tmax, bbox=False, multi=multi)
            if x is None or y is None:
                entity.clear_timeout()
                self.msg.append('Misclick on screen edge')
                return False

            if click_check:
                self.click_checker.start(
                    x, y, red=True, on_failure=entity.clear_timeout,
                    on_success=self.click_checker.reset,
                )
            self.msg.append(f'Clicked: {entity}')
            self.afk_timer.add_timeout(uniform(0.1, 0.2))
            return True
        else:
            # move the mouse to another random position and
            # hope we cant find it. Usually this happens when the
            # mouse moves to a new position and the game client
            # doesn't update, the model has some gaps, or the tile estimation
            # is inaccurate.
            x, y = self.client.screen.mouse_to_object(entity, method=method)
            if x is None or y is None:
                self.msg.append('Cannot mouse off screen edge')
                return False
            # give the game some time to update the new mouse options
            if delay:
                time.sleep(0.1)
            # TODO: right click to see if we can find it
            self.msg.append(f'{entity} occluded: {mo.state}')
            return False

    def _click_tab(self, tab: AbstractWidget):
        """Click the tab icon. Can be used to open or close a tab."""
        if tab.clicked:
            self.msg.append(f'Waiting {tab} menu')
        else:
            tab.click(tmin=.8, tmax=1.2)
            self.msg.append(f'Clicked {tab} menu')

    def _right_click(self, item: GameObject, set_ocr=False):
        """Right-click a game object and create a context menu on it.

        :param GameObject item: The game object to right click.
        :param bool set_ocr: Whether to automatically read OCR on menu items.

        """

        x, y = item.right_click(
            tmin=0.6, tmax=0.9, pause_before_click=True)
        if x is None or y is None:
            self.msg.append('Misclick on screen edge')
            return False
        item.set_context_menu(x, y)
        item.context_menu.OCR_READ_ITEMS = set_ocr
        self.msg.append(f'right clicked {item}')

        # add an afk timer, so we don't *immediately* click
        # the menu option
        self.afk_timer.add_timeout(self.client.TICK + random())
        return True

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

            self.client.logger.debug(
                f'{cur_confidence or 0:.1f} at {cur_map.name}, '
                f'{confidence or 0:.1f} target -> '
                f'{new_confidence or 0:.1f} actual ({map_})'
            )

            if is_none:
                dist = float('inf')
            else:
                dist = mm.distance_between(pos, node)

            self.client.logger.debug(
                f'{dist:.1f} ({range_})'
            )

            if dist < range_ and new_confidence > confidence:
                gps.clear_coordinate_history()
                self.msg.append(f'teleported to: {map_}: {node}')

                # optionally run extra code after the map swapped
                if callable(post_script):
                    post_script()

                return True

            else:
                # set gps back to where it was
                gps.load_map(cur_map.name, set_current=True)
                gps.set_coordinates(*cur_node, add_history=False)
                gps.confidence = cur_confidence
                self.msg.append(f'waiting {item} teleport')

                return False

    def _teleport_with_item(
            self, item, map_: str, node: str, idx: Union[int, None] = None,
            post_script=None, click_check=True,
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
            click on the right click menu. If positive, the index is assumed
            to be the exact index of the menu item. If negative, the index
            will be read dynamically from mouse text.
        :param post_script: Optionally provide a function that can be called
            with no parameters to be run after the teleport menu option has
            been clicked.
        :param int range_: Teleport destination may not be exact, specify the
            potential distance from arrival node in tiles.

        :returns: True if the teleport was successful, False otherwise.

        """

        if idx is None:
            if item.clicked:
                return self._swap_map_from_item(
                    item, map_, node, post_script=post_script, range_=range_,
                    confidence=confidence)
            elif mouse_text:
                self._click_entity(
                    item, tmin, tmax, mouse_text, delay=True, multi=multi,
                    method=method, click_check=click_check)
                self.afk_timer.add_timeout(uniform(0.1, 0.2))
            else:

                bbox = None
                if method:
                    bbox = method()

                # TODO: tweak timeouts on left click teleport
                x, y = item.click(tmin=tmin, tmax=tmax,
                           pause_before_click=True, multi=multi,
                           bbox=bbox)
                if x is None or y is None:
                    item.clear_timeout()
                    self.msg.append('Misclick on screen edge')
                    return False

                if click_check:
                    self.click_checker.start(
                        x, y, red=True, on_failure=item.clear_timeout,
                        on_success=self.click_checker.reset,
                    )
                self.afk_timer.add_timeout(uniform(0.1, 0.2))
                self.msg.append(f'clicked teleport to {map_}')

        elif item.context_menu:

            if idx >= 0:
                inf = item.context_menu.items[idx]
            else:
                for inf in item.context_menu.items:
                    if re.search(mouse_text, inf.value):
                        break
                else:
                    self.client.screen.mouse_away_object(item.context_menu)
                    msg = (
                        f'Could not find {mouse_text} in '
                        f'{", ".join([i.value for i in item.context_menu.items])}'
                    )
                    self.client.logger.warning(msg)
                    self.msg.append(msg)
                    return False

            inf.click(tmin=float('inf'), tmax=float('inf'),
                      pause_before_click=True)
            item.add_timeout(uniform(tmin or 5, tmax or 10))
            self.afk_timer.add_timeout(uniform(0.1, 0.2))
            self.msg.append(f'clicked teleport to {map_}')

        elif item.clicked:
            return self._swap_map_from_item(
                item, map_, node, post_script=post_script, range_=range_,
                confidence=confidence)
        else:
            self._right_click(item, set_ocr=self.client.ocr is not None)
            self.afk_timer.add_timeout(uniform(0.1, 0.3))

        return False

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

    def out_of_supplies(
            self,
            consumable: 'wizard_eyes.consumables.AbstractConsumable'):
        self.client.logger.warning(
            f'Out of supplies: {consumable.name}')
        self.continue_ = False
        return

    def consume(
            self,
            consumable: 'wizard_eyes.consumables.AbstractConsumable'):
        """Consume food, potions etc."""

        inv = self.client.tabs.inventory
        all_consumables = []
        for consumable_ in self.consumables:
            all_consumables.extend(consumable_.template_names)

        if self.client.tabs.active_tab != inv:
            self._click_tab(inv)
            self.afk_timer.add_timeout(uniform(.2, .6))
        else:
            target_object = self.targets.get(consumable.name)
            if target_object:
                # ensure target update separately, because it may no longer be
                # part of tab icons loaded, e.g. because the inventory
                # not full of known items
                target_object.update()

            states = {None, 'nothing', 'something'}
            if target_object and target_object.state not in states:

                # check if any other consumable has been clicked recently, as
                # some objects, like food, have a dela before new consumables
                # can be clicked. In theory this should take into account
                # combo eats, but that sounds like a paint to implement. sorry.
                any_clicked = any([
                    i.clicked for i in inv.interface.icons.values()
                    if i.state in all_consumables
                ])
                time_left = max([
                    i.time_left for i in inv.interface.icons.values()
                    if i.state in all_consumables
                ])

                if any_clicked:
                    self.msg.append(
                        f'waiting {consumable.name} at: {target_object}: '
                        f'{time_left:.3f}')
                else:
                    target_object.click(
                        # standard food has a 3 tick delay,
                        # add a litle extra just in case
                        tmin=self.client.TICK * 3.5, tmax=self.client.TICK * 5,
                        pause_before_click=True
                    )
                    consumable.recalculate(target_object.state)
                    self.msg.append(f'consumed {consumable.name}')
            else:
                new_target = inv.interface.choose_target_icon(
                    *consumable.template_names, clicked=False)

                # TODO: bank run
                if new_target is None:
                    return self.out_of_supplies(consumable)

                self.targets[consumable.name] = new_target
                self.msg.append(f'set new target: {new_target}')

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
            if self.client.skip_frame:
                self.client.skip_frame = False
                continue
            self.afk_timer.update()
            self.client.right_click_menu.update()
            self.click_checker.update()
            self.update()

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
            try:
                draw()
            except Exception as e:
                self.client.logger.debug(f'Error in draw call: {e}')

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
