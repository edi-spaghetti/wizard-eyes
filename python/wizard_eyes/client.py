import json
import win32gui
import argparse
import time
from os.path import join
from typing import Callable

import cv2
import numpy
from ahk import AHK

from .game_objects.personal_menu import Inventory, PersonalMenu
from .game_objects.tabs.container import Tabs
from .game_objects.chat.container import Chat
from .game_objects.bank.container import Bank
from .game_objects.dialogs.dialog import Dialog
from .game_objects.banner import Banner
from .game_objects.minimap.widget import MiniMapWidget
from .screen_tools import Screen
from .game_entities.screen import GameScreen
from .mouse_options import MouseOptions
from .file_path_utils import get_root


class Client(object):

    TICK = 0.6
    STATIC_IMG_PATH_TEMPLATE = '{root}/data/client/{name}.png'
    INVENTORY_SIZE = 28
    ATTACK = 'attack'
    STRENGTH = 'strength'
    DEFENCE = 'defence'
    RANGED = 'ranged'
    PRAYER = 'prayer'
    MAGIC = 'magic'
    RUNECRAFTING = 'runecrafting'
    CONSTRUCTION = 'construction'
    HITPOINTS = 'hitpoints'
    AGILITY = 'agility'
    HERBLORE = 'herblore'
    THIEVING = 'thieving'
    CRAFTING = 'crafting'
    FLETCHING = 'fletching'
    SLAYER = 'slayer'
    HUNTER = 'hunter'
    MINING = 'mining'
    SMITHING = 'smithing'
    FISHING = 'fishing'
    COOKING = 'cooking'
    FIREMAKING = 'firemaking'
    WOODCUTTING = 'woodcutting'
    FARMING = 'farming'
    SKILLS = (
        ATTACK, STRENGTH, DEFENCE, RANGED, PRAYER, MAGIC, RUNECRAFTING,
        CONSTRUCTION, HITPOINTS, AGILITY, HERBLORE, THIEVING, CRAFTING,
        FLETCHING, SLAYER, HUNTER, MINING, SMITHING, FISHING, COOKING,
        FIREMAKING, WOODCUTTING, FARMING
    )

    def __init__(self, name):
        self.args = self.parse_args()
        self.title = None
        self._rect = None
        self._original_img = None
        self._img = None
        self._img_colour = None
        self._draw_calls = None
        self.time = time.time()
        self._start_time = self.time
        self._ahk = self._get_ahk()
        self.name = name
        self._client = self._get_client(name)
        self._win_handle = self._get_win_handle()
        self.containers = None
        self.config = get_config('clients')[name]
        self.screen = Screen(self)

        # TODO: method to load inventory templates from config
        self.inventory = Inventory(self)
        self.bank = Bank(self)
        self.tabs = Tabs(self)
        self.chat = Chat(self)
        self.dialog = Dialog(self)
        self.minimap = MiniMapWidget(self)
        self.banner = Banner(self)
        self.personal_menu = PersonalMenu(self)
        self.game_screen = GameScreen(self)
        self.mouse_options = MouseOptions(self)

    def post_init(self):
        """Run some post init functions that require instantiated attributes"""

        self.setup_containers()
        self.tabs.post_init()
        self.chat.post_init()
        self.bank.post_init()

    def parse_args(self):
        """
        Create an arg parser namespace object to control behaviour of
        various features of the client and its child components.
        """

        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--show', nargs='*', default=[],
            help='optionally specify objects that should display their '
                 'results to the client image. This helps to visualise what '
                 'a client is doing without reading logs.')

        parser.add_argument(
            '--show-gps', action='store_true', default=False,
            help='Show the feature matching used to run gps calculation.'
        )

        parser.add_argument(
            '--show-map', action='store_true', default=False,
            help='Show the local zone (if available) with player marked.'
        )

        parser.add_argument(
            '--save', action='store_true', default=False,
            help='With this enabled the application will be able to save '
                 'client images to a buffer, which can be optionally saved to '
                 'disk.'
        )

        parser.add_argument(
            '--message-buffer', action='store_true', default=False,
            help='Optionally display application logs to separate window.'
        )
        parser.add_argument(
            '--buffer-wh', type=int, nargs=2,
            default=(300, 700),
            help='Specify message buffer width and height (in that order)'
        )

        parser.add_argument(
            '--tracker', nargs='+', default=[],
            help='Use CSRT object tracker to track entity screen positions.'
        )

        parser.add_argument(
            '--static-img',
            help='Optionally specific a static image for the client to read '
                 'on update. This allows you to test without being logged in.'
        )

        args, _ = parser.parse_known_args()
        return args

    def process_img(self, img):
        """
        Process raw image from screen grab into a format ready for template
        matching.
        TODO: refactor client class so we don't have to copy this in from
              game objects.
        :param img: BGRA image section for current slot
        :return: GRAY scaled image
        """
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        return img_gray

    def save_img(self, name=None, original=False):

        if self.args.static_img:
            print(f'Already loaded from static image')
            return

        path = self.static_img_path(name=name)
        img = self.img
        if original:
            img = self.original_img

        cv2.imwrite(path, img)
        print(f'Client image saved to: {path}')

    @property
    def img(self):
        """
        Collects and caches a processed, working image of the current
        client screen. If already cached, that image will be used. If the
        image needs to be refreshed, ensure :meth:`Client.update` is called
        before any methods access the client image.
        The original image is also cached under _original_image.
        """
        if self._img is not None:
            return self._img

        if self.args.static_img:
            path = self.static_img_path(name=self.args.static_img)
            try:
                img = cv2.imread(path)
                # static image gets saved as BGR, but the screen grab is
                # always BGRA - which causes things to break later in some
                # cases - so make sure we have a consistent format.
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            except cv2.error:
                img = numpy.array(
                    (self.height, self.width, 3), dtype=numpy.uint8)
        else:
            img = self.screen.grab_screen(*self.get_bbox())

        img_processed = self.process_img(img)
        self._original_img = img
        self._img = img_processed
        return img_processed

    @property
    def original_img(self):
        return self._original_img

    @property
    def draw_calls(self):
        return self._draw_calls

    def static_img_path(self, name=None):
        name = name or self.args.static_img

        return self.STATIC_IMG_PATH_TEMPLATE.format(
            root=get_root(), name=name
        )

    def add_draw_call(self, func: Callable):
        self._draw_calls.append(func)

    def update(self):
        """Reload the client image and time."""

        # invalidate the img cache, the next time the client image is
        # accessed, it will re-grab the screen
        if not self.args.static_img:
            self._img = None
            self._original_img = None
            _ = self.img  # noqa

        # update the timer. All child components should use this time to
        # ensure consistent measurements
        self.time = time.time()

        # reset draw calls
        self._draw_calls = list()
        self.add_draw_call(self.screen.draw_mouse)

    def setup_containers(self):
        """
        Containers should be defined x: left to right, y: top to bottom
        :return: Dictionary of container configuration
        """

        containers = dict()

        containers['minimap'] = {
            'y': [self.banner, self.minimap]
        }

        containers['mouse_options'] = {
            'y': [self.banner, self.mouse_options]
        }

        containers['personal_menu'] = {
            'y': [self.personal_menu, self.tabs]
        }

        containers['dynamic_tabs'] = {
            'y': [self.tabs]
        }

        containers['chat'] = {
            'y': [self.chat]
        }

        self.containers = containers
        return containers

    def _get_win_handle(self):
        return win32gui.FindWindow(None, self.title)

    def activate(self):
        if not self._client.is_active():
            self._client.activate()

    def set_rect(self):
        """
        Sets bounding box for current client.
        Note, we're using this instead of AHK, because AHK in python has issues
        with dual monitors and I can't find an option to set CoordMode in
        python
        """
        self._rect = win32gui.GetWindowRect(self._win_handle)

    @property
    def width(self):
        if not self._rect:
            self.set_rect()

        return abs(self._rect[2] - self._rect[0]) - 1

    @property
    def height(self):
        if not self._rect:
            self.set_rect()

        return abs(self._rect[3] - self._rect[1]) - 1

    @property
    def margin_top(self):
        return self.config.get('margins', {}).get('top', 0)

    @property
    def margin_bottom(self):
        return self.config.get('margins', {}).get('bottom', 0)

    @property
    def margin_left(self):
        return self.config.get('margins', {}).get('left', 0)

    @property
    def margin_right(self):
        return self.config.get('margins', {}).get('right', 0)

    @property
    def padding_top(self):
        return self.config.get('padding', {}).get('top', 0)

    @property
    def padding_bottom(self):
        return self.config.get('padding', {}).get('bottom', 0)

    @property
    def padding_left(self):
        return self.config.get('padding', {}).get('left', 0)

    @property
    def padding_right(self):
        return self.config.get('padding', {}).get('right', 0)

    def resize(self, x, y):

        if not self._rect:
            self.set_rect()

        win32gui.MoveWindow(
            self._win_handle,
            self._rect[0], self._rect[1],
            self.width + x,
            self.height + y,
            True
        )

        # update rect
        self.set_rect()

    def get_bbox(self):
        """
        Get bounding box for client.
        Returns cached bounding box if already set, otherwise sets and returns
        :return: Client window bounding box coordinates of format
                 (x1, y1, x2, y2)
        :rtype: tuple
        """

        if not self._rect:
            self.set_rect()

        return self._rect

    def localise(self, x1, y1, x2, y2):
        """
        Convert incoming vectors to be relative to the current object.
        TODO: refactor duplicate of GameObject method
        """

        cx1, cy1, _, _ = self.get_bbox()

        # convert relative to own bbox
        x1 = x1 - cx1 + 1
        y1 = y1 - cy1 + 1
        x2 = x2 - cx1 + 1
        y2 = y2 - cy1 + 1

        return x1, y1, x2, y2

    def is_inside(self, x, y, method=None):
        """
        Returns True if the vector is inside bound box.
        TODO: refactor duplicate of GameObject method
        """

        if method is None:
            method = self.get_bbox

        x1, y1, x2, y2 = method()
        return x1 <= x <= x2 and y1 <= y <= y2

    def _get_client(self, name):
        """
        Return a handle to the client window.
        :param name: Name of the client to run, does not need to be exact
        :raises: NotImplementedError if open client not found
        :return: Window object
        """
        client = self._ahk.find_window_by_title(name.encode('ascii'))

        if not client:
            # TODO: allow no client mode on static client
            raise NotImplementedError(
                f"Client '{name}' not found"
                f" - auto-client opening not currently supported"
            )

        # cache out title as utf string
        self.title = client.title.decode('utf-8')

        return client

    def _get_ahk(self):
        path = get_config('paths')['AHK']
        return AHK(executable_path=path)

    def logout(self):
        # TODO: implement logout method
        return False


def get_config(name):
    path = join(get_root(), 'config', name+'.json')
    with open(path, 'r') as f:
        return json.load(f)
