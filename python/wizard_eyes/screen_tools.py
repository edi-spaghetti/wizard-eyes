import ctypes
import random
import time
import platform
from os.path import join

import numpy
import cv2
import pyautogui
import keyboard
from PIL import Image

try:
    import win32gui
    import win32ui
    import win32con
except ImportError:
    # TODO: linux modules
    win32gui = None
    win32ui = None
    win32con = None

from .file_path_utils import get_root
from .constants import WHITE


if platform.system().lower() == 'windows':
    # this actually only needs to be runs once per session, but fixes different
    # monitors screen grabbing the resolution. Solution from here;
    # https://stackoverflow.com/questions/44398075/can-dpi-scaling-be-enabled-disabled-programmatically-on-a-per-session-basis
    PROCESS_PER_MONITOR_DPI_AWARE = 2
    user32 = ctypes.windll.user32
    ctypes.windll.shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)
    user32.SetProcessDPIAware()

# shouldn't be using pyautogui's built in pause functionality, because all
# scripts should have a safety guard to pause/stop the script
# setting this value makes clicks occur at the expected speed
pyautogui.PAUSE = 0


class Screen(object):
    """
    Methods loosely related to interacting with the computer monitor the
    game is being played on.
    """

    CLICK_SPEED_LOWER_BOUND = 0.08
    CLICK_SPEED_UPPER_BOUND = 0.15
    MOUSE_ORB_RADIUS = 5

    def __init__(self, client):
        self.client = client
        self._mouse_position = None

    @property
    def mouse_xy(self):
        """Give the current position of the mouse as x, y coordinates."""
        pos = pyautogui.position()
        return int(pos.x), int(pos.y)

    def grab_screen(self, x1=0, y1=0, x2=0, y2=0):
        """
        Grab an area of the screen and return as numpy array
        """
        # TODO: proper solution that allows any bounding box to be grabbed
        return self.client._window.img

    def on_off_state(self):
        """
        Uses num lock as an on/off switch to determine state
        :return: 1 if num lock is active else 0
        """
        try:
            hllDll = ctypes.WinDLL("User32.dll")
            return (hllDll.GetKeyState(win32con.VK_NUMLOCK)
                    or hllDll.GetKeyState(win32con.VK_CAPITAL))
        except AttributeError:
            # TODO: better linux solution in keyboard class
            state = self.client._window._display.get_keyboard_control().led_mask
            return state in {1, 2, 3}

    def gen_bbox(self):
        xy = []
        input('press enter for x1')
        xy.extend(pyautogui.mouseinfo.position())
        input('press enter for x2')
        xy.extend(pyautogui.mouseinfo.position())
        return tuple(xy)

    def map_between(self, value=None, start=0, stop=1):
        """
        Maps a value between start and stop
        E.g. 0.5 between 0 and 100 would return 50
        :param value: Percentage between start and stop
        :type value: float
        :param start: minimum value to return
        :param stop: maximum value to return
        :return: mapped value
        :rtype: float
        """
        value = value or random.random()

        return (stop - start) * value + start

    def normalise(self, value=None, start=0, stop=1):
        value = value or random.random()
        try:
            return (value - start) / (stop - start)
        except ZeroDivisionError:
            # start and stop are the same, so either value works
            return stop

    def wait_and_click(self, start=None, stop=None, click=True, key=None,
            right=False):
        """
        Waits and optional clicks in a timely manner
        :param start:
        :param stop:
        :param click:
        :return:
        """
        start = start or self.CLICK_SPEED_LOWER_BOUND
        stop = stop or self.CLICK_SPEED_UPPER_BOUND

        wait_period = self.map_between(random.random(), start, stop)

        if sum([bool(p) for p in [click, key, right]]) > 1:
            raise NotImplementedError(
                "Don't do more than one action at the same time")

        if click:
            pyautogui.mouseDown()
        elif key:
            pyautogui.keyDown(key)
        elif right:
            pyautogui.mouseDown(button=pyautogui.RIGHT)

        # print(f'waiting {wait_period}s')
        time.sleep(wait_period)

        if click:
            pyautogui.mouseUp()
        elif key:
            pyautogui.keyUp(key)
        elif right:
            pyautogui.mouseUp(button=pyautogui.RIGHT)

        return wait_period

    def click_aoi(self, x1, y1, x2, y2, speed=1, pause_before_click=False,
                  click=True, right=False, shift=False):
        """
        clicks an area of interst
        :param aoi: dictionary of top left and bottom right
                    within which to click
        :param speed: Define of click as percentage of defaults
        :param shift: Press shift before clicking, release on completion
        :return: position clicked
        """

        if shift:
            keyboard.press('SHIFT')

        # check if we have actually provided a point instead of area
        if x1 == x2 and y1 == y2:
            x, y = x1, y1
        else:
            x, y = self.distribute_normally(x1, y1, x2, y2)
        pyautogui.moveTo(x, y)

        if pause_before_click:
            wait_period = self.map_between(random.random(), 0.05, 0.1)
            time.sleep(wait_period)

        self.wait_and_click(
            self.CLICK_SPEED_LOWER_BOUND * speed,
            self.CLICK_SPEED_UPPER_BOUND * speed,
            click=click,
            right=right,
        )

        if shift:
            keyboard.release('SHIFT')

        return x, y

    def mouse_to(self, x, y):
        pyautogui.moveTo(x, y)

    def mouse_to_object(self, game_object, method=None):
        """
        Move the mouse to a provided game object's bounding box.
        By default use game_object.get_bbox() to determine bounding box, but
        an alternative can also be supplied.
        """

        if method:
            bbox = method()
        else:
            bbox = game_object.get_bbox()

        x, y = self.distribute_normally(*bbox)
        self.mouse_to(x, y)

        return x, y

    def distribute_normally(self, x1, y1, x2, y2):
        centre = x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2

        x = numpy.random.normal(loc=centre[0], scale=(x2 - x1) / 8)
        y = numpy.random.normal(loc=centre[1], scale=(y2 - y1) / 8)

        # failsafe to make sure not out of bounds
        if x < x1:
            x = x1
        if x > x2:
            x = x2
        if y < y1:
            y = y1
        if y > y2:
            y = y2

        return int(x), int(y)

    def press_key(self, key):
        keyboard.press(key)

    def press_hotkey(self, *keys, delay=1):

        for key in keys:
            keyboard.press(key)
            time.sleep(random.random() * delay)
        for key in keys[::-1]:
            keyboard.release(key)
            time.sleep(random.random() * delay)

    def save_img(self, img, path=None):

        if not path:
            path = join(get_root(), 'data', 'test.png')

        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        img = Image.fromarray(img)
        img.save(path)

    def show_img(self, img, name=None):

        name = name or 'test'

        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyWindow(name)

    def draw_mouse(self):

        if 'mouse' in self.client.args.show:
            m = pyautogui.position()
            x, y, _, _ = self.client.localise(m.x, m.y, m.x, m.y)

            # draw a white circle around current mouse position
            cv2.circle(
                self.client.original_img, (x, y),
                self.MOUSE_ORB_RADIUS, WHITE, 1)

            # if mouse has moved, draw a line to more easily see where it went
            if self._mouse_position and self._mouse_position != (x, y):
                ox, oy = self._mouse_position
                cv2.line(self.client.original_img, (x, y), (ox, oy), WHITE)

            self._mouse_position = x, y
