from abc import ABC, abstractmethod
import argparse
from os import makedirs
from os.path import dirname, join
from random import random, uniform, shuffle
import re
import sys
import time
from datetime import datetime
from typing import Union, List, Dict, SupportsIndex, Optional, Tuple, Set
from uuid import uuid4

import cv2
import numpy
import keyboard
import win32gui

from .client import Client
from .dynamic_menus.icon import AbstractIcon
from .dynamic_menus.interface import AbstractInterface
from .dynamic_menus.widget import AbstractWidget
from .file_path_utils import get_root, load_pickle
from .navigation import map_labels
from .game_entities.entity import GameEntity
from .game_entities.screen import ClickChecker
from .game_objects.game_objects import GameObject
from .game_objects.gauges.gps import Map
from .log_overlay import LogOverlay
from .script_utils import int_or_str
from . import heartbeat, login, rest, startup_checks

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

    LONG_TIMEOUT = 60
    LONG_TIMEOUT_CHANCE = 0.001
    MED_TIMEOUT = 6
    MED_TIMEOUT_CHANCE = 0.01
    SHORT_TIMEOUT = 2
    SHORT_TIMEOUT_CHANCE = 0.1

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

    @property
    def tick(self) -> int:
        return int(self.client.time // self.client.TICK)

    def __init__(self, msg_length=100):
        self.continue_ = True
        self.saving_and_exiting = False
        self.force_save_images = False
        self.client: Client = Client(
            *self.client_init_args, **self.client_init_kwargs
        )
        self.sleeping = not self.client.screen.on_off_state()
        # before anything works out positions from the window's size
        startup_checks.ensure_maximised(self.client, self.client.logger)
        # WARNING: do not call any game object's bounding box method before
        #          post init (including debugger breakpoints) as this will
        #          prematurely cache their bbox value and likely be incorrect.
        self.client.post_init()
        # warn (never fail) if the RuneLite profile or game layout aren't
        # what the engine expects; passes are cached, see startup_checks
        startup_checks.run(self.client, self.client.logger)
        # face north + fully top-down before any bot logic runs
        self.camera_state = startup_checks.ensure_camera(
            self.client, self.client.logger)
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
        self.locatable_interfaces: List[AbstractInterface] = []
        self.swap_confidence: Union[float, None] = None
        self.click_checker: ClickChecker = ClickChecker(self.client)
        self.start_time: float = float('inf')

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
        return 'ctrl+q'

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
        return 'ctrl+p'

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

    def stop(self):
        if self.farm_heartbeat is not None:
            self.farm_heartbeat.stop()
        self.continue_ = False
        self.client.continue_ = False
        self.client.gauges.minimap.gps.stop()

    def save_and_exit(self):
        """
        Save client images to disk if configured to do so.
        Note, if showing the images, they may be annotated.

        todo: handle the main thread exiting before images have been saved
        """
        self.client.logger.warning('Saving ...')
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
        self.client.logger.warning(f'Saved to: {new_folder_path}')
        self.exit()

    def exit(self):
        """
        Shut down the application while still running without getting
        threadlock from open cv2.imshow calls.
        """
        self.client.logger.warning('Exiting ...')
        cv2.destroyAllWindows()
        self.stop()
        keyboard.unhook_all()

    def add_default_start_xy_arg(self, parser):
        argument = parser.add_argument(
            '--start-xy', type=int_or_str,
            nargs=2,
            help='Specify starting coordinates by <x,y> or <map,label>'
        )
        gps = self.client.gauges.minimap.gps
        if gps.DEFAULT_METHOD == gps.GRID_INFO_MATCH:
            argument.default = (-1, -1)
            argument.required = False

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

    def add_window_index(self, parser):
        """Deprecated: window selection args now live on the client parser
        (``--window-title`` / ``--window-index``, see
        :meth:`Client._get_win_handle`). Kept so subclasses calling it don't
        break."""

    def create_parser(self) -> argparse.ArgumentParser:
        """Create an argument parser for the application."""
        parser = argparse.ArgumentParser()

        self.add_default_start_xy_arg(parser)
        self.add_default_map_name_arg(parser)
        self.add_default_runtime_arg(parser)
        self.add_window_index(parser)

        self.parser = parser
        return parser

    def parse_args(self):
        args, _ = self.parser.parse_known_args()
        self.args = args

        # post parse start xy - resolve map and label if provided.
        gps = self.client.gauges.minimap.gps

        # grid match doesn't start-xy on input
        if gps.DEFAULT_METHOD == gps.GRID_INFO_MATCH:
            gps.load_map('gielinor')
            return args

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

        gps = self.client.gauges.minimap.gps

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
            self,
            label: Union[str, Tuple[int, int, int, int]],
            map_: Optional[Map] = None,
            count: Union[int, float] = 1,
            regions: Optional[Set[Tuple[int, int, int]]] = None,
    ) -> Union[List[GameEntity], GameEntity]:
        """
        Helper method to set up any arbitrary game entity.

        On the main map, objects come from the game cache and NPCs from
        ``config/npcs.json`` (see ``navigation.map_labels``), nearest first
        if the position is known; on other maps, from the map's own nodes.
        """

        mm = self.client.gauges.minimap
        map_ = map_ or mm.gps.current_map
        regions = regions or set()

        if map_.name == 'gielinor':
            return self._setup_map_entities(label, count, regions)

        if isinstance(label, str):
            nodes = map_.find(label=label)
        elif isinstance(label, tuple):
            nodes = [label]
            label = map_.node_to_label(label).pop()
        else:
            raise TypeError(f'Expected str or tuple, got {type(label)}')

        entities = list()
        for x, y, z in nodes:

            # filters
            rx = x // mm.gps.TILES_PER_REGION
            ry = y // mm.gps.TILES_PER_REGION
            if regions and (rx, ry, z) not in regions:
                continue

            if len(entities) >= count:
                continue

            meta = map_.get_meta(label)
            data = meta.get('nodes', {}).get((x, y, z), {})

            key = (
                int((self.args.start_xy[0] - x) * mm.tile_size),
                int((self.args.start_xy[1] - y) * mm.tile_size)
            )

            width = data.get('width', 1)
            height = data.get('height', 1)

            entity = self.client.game_screen.create_game_entity(
                label, label, key, self.client, self.client,
                tile_width=width, tile_height=height,
            )
            entity.set_global_coordinates(x, y, z)
            if count == 1:
                return entity
            else:
                entities.append(entity)

        return entities

    def _setup_map_entities(self, label, count, regions):
        """Game entities for a label, or a tile, on the main map."""
        mm = self.client.gauges.minimap
        size = mm.gps.TILES_PER_REGION
        if isinstance(label, str):
            things = [(map_labels.anchor(t), t.size)
                      for t in map_labels.find(label)]
            name = label
        elif isinstance(label, tuple):
            at = [t for t in map_labels.object_map().at(label) if t.actions]
            things = [(label, at[0].size if at else (1, 1))]
            name = at[0].name if at else 'tile'
        else:
            raise TypeError(f'Expected str or tuple, got {type(label)}')

        if regions:
            things = [t for t in things
                      if (t[0][0] // size, t[0][1] // size, t[0][2])
                      in regions]
        if not things:
            self.client.logger.warning(
                f'Nothing called {label!r} in the map data')
        pos = mm.gps.get_coordinates()
        if pos[0] != -1:
            things.sort(key=lambda t: max(abs(t[0][0] - pos[0]),
                                          abs(t[0][1] - pos[1])))
        entities = []
        for (x, y, z), (width, height) in things:
            if len(entities) >= count:
                break
            entity = self.client.game_screen.create_game_entity(
                name, name, (0, 0), self.client, self.client,
                tile_width=width, tile_height=height,
            )
            entity.set_global_coordinates(x, y, z)
            if count == 1:
                return entity
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

    def _update_game_entities(
            self,
            *entities: GameEntity,
            mapping: Dict[str, str] = None,
            map_by_z: bool = False,
            regions: Set[Tuple[int, int, int]] = None
    ):
        """
        Update a list of game entities relative to player.
        Note, GPS must have been run this cycle or the key will be outdated.

        :param entities: List of game entity objects to update
        :param dict mapping: A mapping of entity id to map name. Used to
            determine if the entities provided should be checked, because
            they're on the current map and therefore positioned relative to
            player, or should be skipped because they're on a different map.
        :param bool map_by_z: If True, entities are only updated if they are
            on the same z-level as the player.
        :param set regions: Set of regions to filter entities by. If set,
            entities not in these regions will be skipped.

        """

        mapping = mapping or self.entities_mapping

        mm = self.client.gauges.minimap
        gps = self.client.gauges.minimap.gps

        for entity in entities:

            # filters
            if regions:
                _, _, z = gps.get_coordinates()
                x, y, _ = entity.get_global_coordinates()
                rx = x // mm.gps.TILES_PER_REGION
                ry = y // mm.gps.TILES_PER_REGION
                if (rx, ry, z) not in regions:
                    continue

            # skip entities not on the current map
            if mapping:
                if map_by_z:
                    ez = entity.get_global_coordinates()[2]
                    pz = gps.get_coordinates()[2]
                    if ez != pz:
                        continue
                else:
                    map_name = mapping.get(entity.id)
                    if map_name != gps.current_map.name:
                        continue

            x, y, _ = entity.get_global_coordinates()
            px, py, _ = gps.get_coordinates(real=True)
            key = (
                (x - px) * mm.tile_size,
                (py - y) * mm.tile_size,
            )
            entity.update(key=key)

    def _add_afk_timeout(self, min_, max_):
        self.afk_timer.add_timeout(
            self.client.TICK * min_
            + random() * self.client.TICK * max_
        )

    def random_afk(
            self,
            long=None,
            long_chance=None,
            med=None,
            med_chance=None,
            short=None,
            short_chance=None,
    ):
        if long is None:
            long = self.LONG_TIMEOUT
        if long_chance is None:
            long_chance = self.LONG_TIMEOUT_CHANCE
        if med is None:
            med = self.MED_TIMEOUT
        if med_chance is None:
            med_chance = self.MED_TIMEOUT_CHANCE
        if short is None:
            short = self.SHORT_TIMEOUT
        if short_chance is None:
            short_chance = self.SHORT_TIMEOUT_CHANCE

        afk = random()
        if afk < long_chance:
            self.afk_timer.add_timeout(uniform(
                self.client.TICK,
                self.client.TICK * long
            ))
        elif afk < med_chance:
            self.afk_timer.add_timeout(uniform(
                self.client.TICK,
                self.client.TICK * med
            ))
        elif afk < short_chance:
            self.afk_timer.add_timeout(uniform(
                self.client.TICK,
                self.client.TICK * short
            ))

        return afk

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
        mm = self.client.gauges.minimap
        gps = self.client.gauges.minimap.gps

        dist = mm.distance_between(
            gps.get_coordinates(),
            entity.get_global_coordinates()
        )
        # divide 2 because we assume running
        # TODO: support walking/running detection
        dist_timeout = max([dist * (self.client.TICK / 2), self.client.TICK])

        return dist_timeout + action_timeout

    def _click_entity(
        self,
        entity,
        tmin,
        tmax,
        mouse_text,
        method=None,
        delay=True,
        speed=1,
        multi=1,
        click_check=True,
        shift=False,
        on_click_check_success=None,
        on_click_check_fail=None,
        excluded_objects: List[GameObject] = None,
    ):
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

        on_click_check_success = (
                on_click_check_success or self.click_checker.reset
        )
        on_click_check_fail = (
                on_click_check_fail or entity.clear_timeout
        )
        excluded_objects = excluded_objects or list()

        if not entity.is_inside(*self.client.screen.mouse_xy, method=method):
            x, y = self.client.screen.mouse_to_object(entity, method=method)
            if x is None or y is None:
                self.msg.append('Cannot mouse off screen edge')
                return False
            for obj in excluded_objects:
                if obj.is_inside(x, y):
                    msg = f'Cannot mouse to excluded object: {obj}'
                    self.client.logger.warning(msg)
                    self.msg.append(msg)
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
                    multi=multi, shift=shift,
                )
                if x is None or y is None:
                    entity.clear_timeout()
                    self.msg.append('Misclick on screen edge')
                    return False

                if click_check:
                    self.click_checker.start(
                        x, y, red=True, on_failure=on_click_check_fail,
                        on_success=on_click_check_success,
                    )

                self.msg.append(f'Clicked {entity}')
                self.afk_timer.add_timeout(uniform(0.1, 0.2))
                return True

        elif re.search(mouse_text, mo.state):
            x, y = entity.click(
                tmin=tmin, tmax=tmax, bbox=False, multi=multi, shift=shift
            )
            if x is None or y is None:
                entity.clear_timeout()
                self.msg.append('Misclick on screen edge')
                return False

            if click_check:
                self.click_checker.start(
                    x, y, red=True, on_failure=on_click_check_fail,
                    on_success=on_click_check_success,
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

    MAX_BANK_CLOSES = 2
    """Tries at closing a bank that's hiding the tabs at start up."""
    _bank_closes = 0

    def close_bank_to_locate_tabs(self) -> bool:
        """If the bank was already open when the bot started, the side tabs
        are greyed out and can't be told apart, so tabs the bot relies on
        are never located. Close the bank so they can be.

        Only input if needed: nothing happens once the tabs are located.

        :returns: True if the bank was closed (nothing else should be
            done this frame).
        """
        bank = self.client.bank
        if not bank.open or bank.clicked:
            return False
        # the logout tab is placed by the gauges, not by its template
        missing = [w for w in self.client.tabs.widgets
                   if w.auto_locate and not w.located and w.name != 'logout']
        if not missing or self._bank_closes > self.MAX_BANK_CLOSES:
            return False
        self._bank_closes += 1
        if self._bank_closes > self.MAX_BANK_CLOSES:
            self.client.logger.error(
                f'Tabs not located after closing the bank '
                f'{self.MAX_BANK_CLOSES} times: {missing}; stopping')
            self.continue_ = False
            return True
        self.client.logger.info(
            f'Bank open before tabs were located, closing it: {missing}')
        bank.close()
        self.afk_timer.add_timeout(2 + random())
        return True

    LOGIN_CHECK_SECONDS = 3
    """How often to check we're still logged in (with --username)."""
    MAX_RELOGINS = 3
    """Give up after this many logins within an hour (e.g. the account is
    being played somewhere else)."""
    LOGGED_OUT = ('login', 'lobby', 'dialog', 'client_updated', 'gone')
    _next_login_check = 0.0
    _logged_out_checks = 0
    _relogins: Tuple[float, ...] = ()

    def login_state(self) -> str:
        """What the client window shows (see :func:`login.classify`);
        "gone" if there's no such window any more."""
        if not win32gui.IsWindow(self.client._win_handle):
            return 'gone'
        img = self.client.img
        return 'loading' if img is None else login.classify(img)[0]

    def check_login(self, stopping: bool = False) -> bool:
        """Log back in if we've been logged out (needs --username).

        Cheap when logged in: an image match every few seconds. Two checks
        in a row have to agree before anything is done.

        :param stopping: the bot wants to stop; check now.
        :returns: True if we logged back in, and the bot should carry on.
        """
        username = getattr(self.client.args, 'username', None)
        now = time.time()
        if not username or not (stopping or now >= self._next_login_check):
            return False
        self._next_login_check = now + self.LOGIN_CHECK_SECONDS
        state = self.login_state()
        if state == 'banned':
            self.account_banned()
            return False
        if state not in self.LOGGED_OUT:
            self._logged_out_checks = 0
            return False
        self._logged_out_checks += 1
        if self._logged_out_checks < 2:
            if not stopping:
                return False
            time.sleep(1.5)
            self.client.update()
            if self.login_state() not in self.LOGGED_OUT:
                return False
        self._logged_out_checks = 0
        return self.log_back_in(username)

    def log_back_in(self, username: str, counted: bool = True) -> bool:
        """:param counted: False after a rest (being logged out was
            expected, so it doesn't count towards ``MAX_RELOGINS``)."""
        log = self.client.logger
        self._relogins = tuple(
            t for t in self._relogins if time.time() - t < 3600)
        if counted and len(self._relogins) >= self.MAX_RELOGINS:
            log.error(f'Logged out, but already logged in '
                      f'{self.MAX_RELOGINS} times in the last hour; stopping')
            self.stop()
            return False
        if counted:
            self._relogins += (time.time(), )
            wait = uniform(5, 20)
            log.warning(f'Logged out; logging back in in {wait:.0f}s')
            time.sleep(wait)
        else:
            log.info('Logging back in')
        rect = tuple(self.client.get_bbox())
        try:
            window = login.ensure_logged_in(
                username, timeout=self.client.args.login_timeout)
        except login.AccountBanned:
            self.account_banned()
            return False
        except login.LoginError as e:
            log.error(f'Could not log back in: {e}; stopping')
            self.stop()
            return False
        self.client._win_handle = window.handle
        startup_checks.ensure_maximised(self.client, log)
        self.client.set_rect()
        now_at = tuple(self.client.get_bbox())
        if now_at != rect:
            # everything located so far would be in the wrong place
            log.error(f'Client window moved from {rect} to {now_at}; '
                      f'stopping')
            self.stop()
            return False
        startup_checks.ensure_camera(self.client, log)
        self.client.activate()
        self.afk_timer.clear_timeout()
        self.on_relogin()
        log.info('Logged back in, carrying on')
        return True

    banned = False
    rest_schedule: Optional[rest.RestSchedule] = None
    resting: Optional[rest.Rest] = None
    REST_LINE_SECONDS = 60
    _rest_waiting = False

    def start_rests(self, argv=None):
        """Rests from the command line (``--sleep`` / ``--breaks``, see
        :mod:`wizard_eyes.rest`)."""
        log = self.client.logger
        try:
            self.rest_schedule = rest.from_args(
                sys.argv[1:] if argv is None else argv)
        except (ValueError, SystemExit) as e:
            log.error(f'Rest options: {e}; playing without rests')
            self.rest_schedule = None
        if self.rest_schedule is None:
            return
        log.info(f'Rests: {self.rest_schedule}')
        if not getattr(self.client.args, 'username', None):
            log.warning('Rests without --username: after the game logs '
                        'the bot out it will stop instead of logging back in')
        first = self.rest_schedule.due(time.time()) or \
            self.rest_schedule.next
        if first is not None:
            when = datetime.fromtimestamp(first.start).strftime('%H:%M')
            log.info(f'Next rest at {when}: {first}')

    def can_rest(self) -> bool:
        """Is now a good moment to stop for a rest? Bots override this to
        finish what mustn't be left half done (a trade, a meeting, a trip
        through danger). Asked every frame while a rest is due."""
        return True

    def take_rest(self) -> bool:
        """Take a rest if one is due and the bot can stop now; True if one
        was taken (the frame is over)."""
        if self.rest_schedule is None:
            return False
        due = self.rest_schedule.due(time.time())
        if due is None:
            self._rest_waiting = False
            return False
        if not self.can_rest():
            if not self._rest_waiting:
                self.client.logger.info(f'A {due.kind} is due; waiting for '
                                        f'a good moment')
            self._rest_waiting = True
            return False
        self._rest_waiting = False
        self.rest(due)
        return True

    def rest(self, period: 'rest.Rest'):
        """Do nothing until the rest is over, then make sure we're in the
        game again."""
        log = self.client.logger
        log.info(f'Resting: {period}')
        self.resting = period
        if self.farm_heartbeat is not None:
            self.farm_heartbeat.send()
        last_line = 0.0
        try:
            while self.continue_ and self.client.continue_:
                now = time.time()
                if now >= period.end:
                    break
                if now - last_line >= self.REST_LINE_SECONDS:
                    # (a line now and then: the farm sees the bot is alive)
                    last_line = now
                    left = (period.end - now) / 60
                    msg = (f'Time {self._runtime_str()} 0.000 - Resting '
                           f'({period.kind}) {left:.0f} min to go')
                    sys.stdout.write('\b' * self.msg_length)
                    line = msg[:self.msg_length]
                    sys.stdout.write(f'{line:{self.msg_length}}')
                    sys.stdout.flush()
                time.sleep(min(5.0, max(0.1, period.end - now)))
        finally:
            self.resting = None
            self.rest_schedule.taken(time.time())
        if not (self.continue_ and self.client.continue_):
            return
        log.info(f'Rest over ({period.kind})')
        self.after_rest()

    def after_rest(self):
        """Back from a rest: log in again if the game logged us out."""
        self.client.update()
        state = self.login_state()
        if state == 'banned':
            return self.account_banned()
        if state in self.LOGGED_OUT:
            username = getattr(self.client.args, 'username', None)
            if not username:
                self.client.logger.error('Logged out during the rest and no '
                                         '--username to log back in; '
                                         'stopping')
                return self.stop()
            self.log_back_in(username, counted=False)
        else:
            self.afk_timer.clear_timeout()
            self.on_rest_over()
        self._next_login_check = 0.0

    def on_rest_over(self):
        """Called after a rest the game didn't log us out of (a short
        break). Override to recheck anything that may have changed."""

    def _runtime_str(self) -> str:
        runtime = time.time() - self.start_time
        return (f'{runtime // 3600:.0f}h {(runtime % 3600) // 60:.0f}m '
                f'{runtime % 60:.0f}s')

    def account_banned(self):
        """The login screen says the account is banned: tell the farm (its
        watchdog alerts, and doesn't restart the bot) and stop."""
        self.banned = True
        self.client.logger.error(f'{login.BANNED_TEXT[0].upper()}'
                                 f'{login.BANNED_TEXT[1:]}; stopping')
        if self.farm_heartbeat is not None:
            self.farm_heartbeat.send()
        self.stop()

    def on_relogin(self):
        """Called after logging back in mid-run: forget anything in
        progress. Override in bots that keep such state."""

    def _click_tab(self, tab: AbstractWidget):
        """Click the tab icon. Can be used to open or close a tab."""
        if tab.clicked:
            self.msg.append(f'Waiting {tab} menu')
        elif not tab.clickable:
            self.msg.append(f'{tab} not located yet')
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
        # self.afk_timer.add_timeout(self.client.TICK + random())
        return True

    def _swap_map_from_item(
            self, item, map_, node, post_script=None,
            range_=DEFAULT_MAP_SWAP_RANGE, confidence=None):
        """Swap maps due to clicking an entity. It may be a right click menu
        or a left click on an object or game entity."""
        gps = self.client.gauges.minimap.gps
        mo = self.client.mouse_options
        mm = self.client.gauges.minimap
        confidence = confidence or 0.0

        if mo.state in {'loading', 'waiting'}:
            self.msg.append('waiting game load')
        else:

            cur_confidence = gps.confidence
            cur_node = gps.get_coordinates(real=True)[:2]
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
            post_script=None, click_check=True, shift=False,
            range_=DEFAULT_MAP_SWAP_RANGE, tmin=None, tmax=None,
            mouse_text=None, multi=1, confidence=None, method=None) -> Optional[bool]:
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
        :param bool shift: If left-clicking, where or not to do a shift click.
        :param int range_: Teleport destination may not be exact, specify the
            potential distance from arrival node in tiles.

        :returns: True if the teleport was successful,
            None if no action was taken, False otherwise.

        """

        if idx is None:
            if item.clicked:
                result = self._swap_map_from_item(
                    item, map_, node, post_script=post_script, range_=range_,
                    confidence=confidence)
                return result or None
            elif mouse_text:
                result = self._click_entity(
                    item, tmin, tmax, mouse_text, delay=True, multi=multi,
                    method=method, click_check=click_check, shift=shift)
                if result:
                    # we clicked something, but map hasn't been swapped
                    return False
                else:
                    # no click
                    return None
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
                    return

                if click_check:
                    self.click_checker.start(
                        x, y, red=True, on_failure=item.clear_timeout,
                        on_success=self.click_checker.reset,
                    )
                self.msg.append(f'clicked teleport to {map_}')
                return False

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
                    return None

            inf.click(tmin=float('inf'), tmax=float('inf'),
                      pause_before_click=True)
            self.msg.append(f'clicked teleport to {map_}')
            return False

        elif item.clicked:
            result = self._swap_map_from_item(
                item, map_, node, post_script=post_script, range_=range_,
                confidence=confidence)
            return result or None
        else:
            result = self._right_click(
                item, set_ocr=self.client.ocr is not None
            )
            if result:
                # we clicked something, but map hasn't been swapped
                return False
            else:
                # no click
                return None

    def hop_worlds(self):
        """"""

        tabi = self.client.tabs.logout.interface
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
        self.stop()
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
            self.afk_timer.add_timeout(uniform(.3, .6))
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
                        # add a litTle extra just in case
                        tmin=self.client.TICK * 4, tmax=self.client.TICK * 6,
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

    def locate_interfaces(self):
        """Some setup requires action, such as locating dynamic interfaces.

        This method will attempt to locate all the requested interfaces by
        clicking each of their respective widgets to bring the interface on
        screen where it can then be located for later use.

        The interfaces can be added on app setup e.g.

        .. code-block:: python

            class IronMiner2000(Application):
                def setup(self):
                    self.locatable_interfaces.append(
                        self.client.tabs.inventory.interface
                    )

                def action(self):
                    if self.locate_interfaces():
                        self.msg.append('we clicked something')
                        return

                    # mine that iron, champ!

        :returns: True if an action was taken.

        """

        interfaces = self.locatable_interfaces.copy()
        # shuffle them so we don't click in the same order
        shuffle(interfaces)

        for interface in interfaces:
            if not interface.located:
                if self.client.tabs.active_tab != interface.widget:
                    self._click_tab(interface.widget)
                    # we've clicked something, which is a common action,
                    # so let the main action method know about it.
                    return True

        # no common action taken
        return False

    @abstractmethod
    def action(self):
        """
        Perform an action (or not) depending on the current state of the
        application. It is advisable to limit actions to one per run cycle.
        """

    log_overlay: Optional[LogOverlay] = None

    progress: Optional[Union[int, float, str]] = None
    """Optional: anything that changes as the bot gets things done (e.g. a
    count of inventories). Sent with the heartbeat; a farm's watchdog
    restarts a bot whose progress stops changing."""
    farm_heartbeat: Optional[heartbeat.Heartbeat] = None

    def start_log_overlay(self):
        """With --log-overlay: a strip with the latest log lines over the
        client. The application runs the same without it."""
        if not getattr(self.client.args, 'log_overlay', False):
            return
        try:
            self.log_overlay = LogOverlay(self.client.title)
        except OSError as e:
            self.client.logger.warning(f'No log overlay: {e}')
            return
        self.client.logger.addHandler(self.log_overlay)

    def run(self):
        # run
        print('Entering Main Loop')
        self.start_log_overlay()
        self.client.activate()
        self.start_time = time.time()
        self.start_rests()
        # (only when a farm started the bot: see wizard_eyes.heartbeat)
        self.farm_heartbeat = heartbeat.start_for(self)
        while self.continue_:

            # put application in a holding pattern while we clean up
            if self.saving_and_exiting:
                time.sleep(0.1)
                continue

            # check if we've run out of time
            if time.time() - self.start_time > self.args.run_time:
                self.client.logger.warning('Out of time')
                self.stop()
                break

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
            if self.check_login():
                continue
            if self.client.skip_frame:
                self.client.skip_frame = False
                continue
            if self.take_rest():
                continue
            self.afk_timer.update()
            self.client.right_click_menu.update()
            self.click_checker.update()
            self.update()

            # do an action (or not, it's your life)
            if self.afk_timer.time_left:
                self.msg.append(f'AFK: {self.afk_timer.time_left:.3f}')
            elif self.close_bank_to_locate_tabs():
                self.msg.append('closing bank to locate tabs')
            else:
                self.action()

            # being logged out looks like a lot of other problems (bank not
            # opening etc.), so check before believing the bot's decision
            if not self.continue_ and self.client.continue_ and \
                    self.check_login(stopping=True):
                self.continue_ = True

            # log run cycle
            t2 = time.time()  # not including show image time
            runtime = t2 - self.start_time
            hours = runtime // 3600
            minutes = (runtime % 3600) // 60
            seconds = runtime % 60
            runtime_str = f'{hours:.0f}h {minutes:.0f}m {seconds:.0f}s'
            self.msg.insert(0, f'Time {runtime_str} {t2 - t1:.3f}')
            msg = ' - '.join(self.msg)
            self.msg_buffer.append(msg)
            if len(self.msg_buffer) > 69:
                self.msg_buffer = self.msg_buffer[1:]  # remove oldest

            sys.stdout.write('\b' * self.msg_length)
            sys.stdout.write(f'{msg[:self.msg_length]:{self.msg_length}}')
            sys.stdout.flush()
            if self.log_overlay is not None:
                self.log_overlay.write(msg)

            self.show()

        self.stop()

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
            gps = self.client.gauges.minimap.gps
            if gps.current_map is not None:
                images.append((name, gps.current_map.img_copy))

        if self.client.args.save or self.force_save_images:
            self.images = self.images[1:self.buffer]
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
            gps = self.client.gauges.minimap.gps
            if gps.current_map is not None:
                images.append((name, gps.show_img))

        if images:
            for i, (name, image) in enumerate(images):
                try:
                    cv2.imshow(name, image)
                except cv2.error as err:
                    pass
                    # self.client.game_screen.player.logger.error(
                    #     f'cannot show: {name}, {err}'
                    # )
                    # raise

                widths = [im.shape[1] for _, im in images[:i]]
                cv2.moveWindow(name, 5 + sum(widths), 20)
            cv2.waitKey(1)

        # set the original original image back for next loop update
        if self.client.args.static_img:
            self.client._original_img = img
