import sys
import json
import win32gui
import argparse
import time
from os.path import join, dirname
from os import _exit
from abc import ABC, abstractmethod

import cv2
import numpy
import keyboard
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

    @property
    def original_img(self):
        return self._original_img

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

    def is_inside(self, x, y):
        """
        Returns True if the vector is inside bound box.
        TODO: refactor duplicate of GameObject method
        """

        x1, y1, x2, y2 = self.get_bbox()
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


class Application(ABC):
    """Base class application with methods for implementation."""

    PATH = f'{dirname(__file__)}/data/recordings/{{}}.png'

    def __init__(self, client='RuneLite', msg_length=100):
        self.continue_ = True
        self.client = Client(client)
        self.msg = list()
        self.msg_length = msg_length
        self.msg_buffer = list()

        # set up callback for immediate exit of application
        keyboard.add_hotkey(self.exit_key, self.exit)

        self.images = list()
        keyboard.add_hotkey(self.save_key, self.save_and_exit)

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
        self.continue_ = False

        for i, image in enumerate(self.images):
            path = self.PATH.format(i)
            self.client.screen.save_img(image, path)
        print(f'Saved to: {self.PATH}')
        self.exit()

    def exit(self):
        """
        Shut down the application while still running without getting
        threadlock from open cv2.imshow calls.
        """
        print('Exiting ...')
        cv2.destroyAllWindows()
        _exit(1)

    @abstractmethod
    def setup(self):
        """
        Run any of the application setup required *before* entering the main
        event loop.
        """

    @abstractmethod
    def update(self):
        """
        Update things like internal state, as well as run the update methods
        of any game objects that are required.
        """

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

            # set up logging for new cycle
            sys.stdout.write('\b' * self.msg_length)
            self.msg = list()
            t1 = time.time()

            # caps lock to pause the script
            # p to exit
            # TODO: convert these to utility functions
            if not self.client.screen.on_off_state():
                msg = f'Sleeping @ {self.client.time}'
                sys.stdout.write(f'{msg:{self.msg_length}}')
                sys.stdout.flush()
                time.sleep(0.1)
                continue

            # ensure the client is updated every frame and run the
            # application's update method
            self.client.update()
            self.update()

            # do an action (or not, it's your life)
            self.action()

            t2 = time.time()
            self.msg.insert(0, f'Cycle {t2 - t1:.3f}')

            # log run cycle
            msg = ' - '.join(self.msg)
            self.msg_buffer.append(msg)
            if len(self.msg_buffer) > 69:
                self.msg_buffer = self.msg_buffer[1:]  # remove oldest

            sys.stdout.write(f'{msg[:self.msg_length]:{self.msg_length}}')
            sys.stdout.flush()

            # do image stuff
            if self.client.args.show:
                cv2.imshow('Client', self.client.original_img)

            if self.client.args.save:
                self.images = self.images[:self.buffer - 1]
                self.images.append(self.client.original_img)

            if self.client.args.message_buffer:
                buffer = numpy.ones((700, 300, 4), dtype=numpy.uint8)

                for i, msg in enumerate(self.msg_buffer, start=1):
                    buffer = cv2.putText(
                        buffer, msg, (10, 10 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                        (50, 50, 50, 255), thickness=1)
                cv2.imshow('Logs', buffer)

            if 'gps' in self.client.args.show:
                name = 'Gielenor Positioning System'
                cv2.imshow(name, self.client.minimap.minimap.display_img)

            # TODO: configurable window position
            if self.client.args.show:
                cv2.moveWindow('Client', 10, 20)
            if self.client.args.message_buffer:
                cv2.moveWindow(
                    'Logs', self.client.original_img.shape[1] + 5, 20)
            if 'gps' in self.client.args.show:
                cv2.moveWindow(
                    'Gielenor Positioning System',
                    self.client.original_img.shape[1] + 5 + 300 + 5, 20
                )

            if (self.client.args.show
                    or self.client.args.message_buffer
                    or 'gps' in self.client.args.show):
                cv2.waitKey(1)


def get_config(name):
    path = join(dirname(__file__), 'config', name+'.json')
    with open(path, 'r') as f:
        return json.load(f)
