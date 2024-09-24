import atexit
import json
import win32gui
import argparse
import time
from os.path import join, exists
from typing import Callable, Union, Tuple

import cv2
import numpy
import tesserocr

from .game_objects.game_objects import GameObject
from .game_objects.right_click_menu import RightClickMenu
from .game_objects.tabs.container import Tabs
from .game_objects.chat.container import Chat
from .game_objects.bank.container import Bank
from .game_objects.counters.container import Counters
from .game_objects.banner import Banner
from .game_objects.gauges.widget import GaugesWidget
from .screen_tools import Screen
from .game_entities.screen import GameScreen
from .mouse_options import MouseOptions
from .file_path_utils import get_root
from .constants import DEFAULT_ZOOM


class Client(GameObject):
    """The client object is the main interface to the game screen."""

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

    HSV = 0
    BGRA = 1
    GRAY = 2

    def __init__(self, name, zoom=DEFAULT_ZOOM):
        self.args = self.parse_args()
        self.title = None
        self._rect = None
        self.continue_ = True
        self.skip_frame = False  # sometimes img fails
        self._original_img = None
        self._img = None
        self._hsv_img = None
        self._img_colour = None
        self._draw_calls = None
        self.last_time: float = -float('inf')
        self.time: float = time.time()
        self._start_time: float = self.time
        self.name: str = name
        self._win_handle: int = 0
        self._get_win_handle()
        self.containers = None
        self.screen: Screen = Screen(self)

        super().__init__(self, self)
        self.ocr: Union[tesserocr.PyTessBaseAPI, None] = self.init_ocr()
        self.config = get_config('clients')[name]

        self.bank: Bank = Bank(self)
        self.tabs: Tabs = Tabs(self)
        self.chat: Chat = Chat(self)
        self.gauges: GaugesWidget = GaugesWidget(self)
        self.banner: Banner = Banner(self)
        self.game_screen: GameScreen = GameScreen(self, zoom=zoom)
        self.mouse_options: MouseOptions = MouseOptions(self)
        self.counters: Counters = Counters(self)
        self.right_click_menu: RightClickMenu = RightClickMenu(
            self, self, -1, -1)

    @property
    def minimap(self):
        self.client.logger.warning(
            'Client.minimap is deprecated, use Client.guages instead'
        )
        return self.gauges

    def init_ocr(self) -> Union[tesserocr.PyTessBaseAPI, None]:
        # Assume tessdata is cloned relative to this repo
        # download from https://github.com/tesseract-ocr/tessdata.git
        path = join(get_root(), '..', 'tessdata')
        if not exists(path):
            self.logger.warning(f'No OCR tessdata found at: {path}')
            return

        ocr = tesserocr.PyTessBaseAPI(path=path)
        atexit.register(ocr.End)
        return ocr

    def post_init(self):
        """Run some post init functions that require instantiated attributes"""

        self.setup_client_containers()

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
        args.show = set(args.show)
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
            try:
                img = self.screen.grab_screen(*self.get_bbox())
            except Exception as e:
                if e is KeyboardInterrupt:
                    raise
                else:
                    self.skip_frame = True
                    return

        img_processed = self.process_img(img)
        self._hsv_img = self.convert_to_hsv(img)
        self._original_img = img
        self._img = img_processed
        return img_processed

    @property
    def original_img(self):
        return self._original_img

    @property
    def hsv_img(self):
        return self._hsv_img

    def convert_to_hsv(self, img):
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        return hsv

    def get_img_at(self, bbox, mode=None):
        """Pass in a bounding box, get a subsection of client image at that
        location."""

        x1, y1, x2, y2 = self.localise(*bbox)
        if mode == self.BGRA:
            return self.original_img[y1:y2, x1:x2]
        elif mode == self.HSV:
            return self.hsv_img[y1:y2, x1:x2]
        else:
            return self.img[y1:y2, x1:x2]

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

    def remove_draw_call(self, func: Callable):
        idx = self._draw_calls.index(func)
        if idx > -1:
            self._draw_calls.pop(idx)

    def update(self):
        """Reload the client image and time."""

        # invalidate the img cache, the next time the client image is
        # accessed, it will re-grab the screen
        if not self.args.static_img:
            self._img = None
            self._original_img = None
            _ = self.img  # noqa

        if self.skip_frame:
            return

        # update the timer. All child components should use this time to
        # ensure consistent measurements
        self.last_time = self.time
        self.time = time.time()

        # reset draw calls
        self._draw_calls = list()
        self.add_draw_call(self.screen.draw_mouse)

    def setup_client_containers(self):
        """
        Containers should be defined x: left to right, y: top to bottom
        :return: Dictionary of container configuration
        """

        containers = dict()

        containers['minimap'] = {
            'y': [self.banner, self.gauges]
        }

        containers['mouse_options'] = {
            'y': [self.banner, self.mouse_options, self.counters]
        }

        self.containers = containers
        return containers

    def _get_win_handle(self):
        """Get the window handle for the client.

        :return: The window handle for the client.
        :rtype: int

        """

        def find_by_title(handle, *_):
            title = win32gui.GetWindowText(handle)
            if "Old School RuneScape" in title or "RuneLite" in title:
                self._win_handle = handle

        win32gui.EnumWindows(find_by_title, None)

    def _activate(self, handle: int, *_):
        """Activate the given window if it matches the client window.

        :param int handle: The handle of the window to activate.

        """
        if handle == self._win_handle:
            # no idea why, but pressing alt before setting foreground makes
            # it work - isn't windows a wonderful operating system?
            self.client.screen.press_key('alt')
            win32gui.SetForegroundWindow(handle)

    def activate(self):
        """Activate the client window.

        It will first attempt to activate the client window using the
        command+index hotkey. This works a lot better if you have pinned the
        game client to your taskbar.

        If that fails, it will enumerate all windows and attempt to activate
        the client window by title. This usually works, but under certain
        conditions it doesn't.

        """

        try:
            self.client.screen.press_key(
                # if SKIP_PARSE_ARGS is set, self.args may fail
                f'command left+{self.args.window_index}'
            )
            title = win32gui.GetWindowText(win32gui.GetForegroundWindow())
            if "Old School RuneScape" in title or "RuneLite" in title:
                self._win_handle = win32gui.GetForegroundWindow()
            else:
                raise AttributeError
        except AttributeError:
            self.client.logger.warning('hotkey failed, trying alternative')
            win32gui.EnumWindows(self._activate, None)

    def set_rect(self):
        """Sets bounding box for current client."""
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

    def get_bbox(self) -> Tuple[int, int, int, int]:
        """Get bounding box for client.

        Returns cached bounding box if already set, otherwise sets and returns

        :return: Client window bounding box coordinates of format
                 (x1, y1, x2, y2)
        :rtype: tuple
        """

        if not self._rect:
            self.set_rect()

        return self._rect

    def logout(self):
        # TODO: implement logout method
        return False


def get_config(name):
    path = join(get_root(), 'config', name+'.json')
    with open(path, 'r') as f:
        return json.load(f)
