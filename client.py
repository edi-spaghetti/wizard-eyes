import json
import win32gui
import argparse
import time
from os.path import join, dirname

import cv2
from ahk import AHK

from game_objects import (
    Inventory,
    Tabs,
    Bank,
    Dialog,
    Banner,
    MiniMapWidget,
    PersonalMenu,
)
from screen_tools import Screen
from game_screen import GameScreen


class Client(object):

    TICK = 0.6

    def __init__(self, name):
        self.args = self.parse_args()
        self.title = None
        self._rect = None
        self._original_img = None
        self._img = None
        self.time = time.time()
        self._ahk = self._get_ahk()
        self.name = name
        self._client = self._get_client(name)
        self._win_handle = self._get_win_handle()
        self.config = get_config('clients')[name]
        self.screen = Screen()

        # TODO: method to load inventory templates from config
        self.inventory = Inventory(self)
        self.bank = Bank(self)
        self.tabs = Tabs(self)
        self.dialog = Dialog(self)
        self.minimap = MiniMapWidget(self)
        self.banner = Banner(self)
        self.personal_menu = PersonalMenu(self)
        self.game_screen = GameScreen(self)

        self.containers = self.setup_containers()
        # TODO: untangle this, so we can build tab items on init
        self.tabs.build_tab_items()

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

        args = parser.parse_args()
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

        img = self.screen.grab_screen(*self.get_bbox())
        img_processed = self.process_img(img)
        self._original_img = img
        self._img = img_processed
        return img_processed

    def update(self):
        """Reload the client image and time."""

        # collect and process the current client screen
        img = self.screen.grab_screen(*self.get_bbox())
        self._original_img = img
        img_processed = self.process_img(img)
        self._img = img_processed

        # update the timer. All child components should use this time to
        # ensure consistent measurements
        self.time = time.time()

    def setup_containers(self):
        """
        Containers should be defined x: left to right, y: top to bottom
        :return: Dictionary of container configuration
        """

        containers = dict()

        containers['minimap'] = {
            'y': [self.banner, self.minimap]
        }

        containers['personal_menu'] = {
            'y': [self.personal_menu, self.tabs]
        }

        containers['dynamic_tabs'] = {
            'y': [self.tabs]
        }

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

    def _get_client(self, name):
        """
        Return a handle to the client window.
        :param name: Name of the client to run, does not need to be exact
        :raises: NotImplementedError if open client not found
        :return: Window object
        """
        client = self._ahk.find_window_by_title(name.encode('ascii'))

        if not client:
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
    path = join(dirname(__file__), 'config', name+'.json')
    with open(path, 'r') as f:
        return json.load(f)
