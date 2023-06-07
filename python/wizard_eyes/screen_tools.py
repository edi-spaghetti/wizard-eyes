import random
import time
from os.path import join

import ctypes
import win32gui
import win32ui
import win32con
import numpy
import cv2
import pyautogui
import keyboard
from PIL import Image

from .file_path_utils import get_root
from .constants import WHITE
from .game_objects.game_objects import GameObject


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
        hwin = win32gui.GetDesktopWindow()

        width = x2 - x1 + 1
        height = y2 - y1 + 1

        hwindc = win32gui.GetWindowDC(hwin)
        srcdc = win32ui.CreateDCFromHandle(hwindc)
        memdc = srcdc.CreateCompatibleDC()

        # create a blank bitmap image and bind to DC
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(srcdc, width, height)
        memdc.SelectObject(bmp)

        # copy the screen into buffer
        memdc.BitBlt((0, 0), (width, height), srcdc, (x1, y1), win32con.SRCCOPY)
        signedIntsArray = bmp.GetBitmapBits(True)

        # convert to numpy array
        img = numpy.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (height, width, 4)

        # cleanup
        srcdc.DeleteDC()
        memdc.DeleteDC()
        win32gui.ReleaseDC(hwin, hwindc)
        win32gui.DeleteObject(bmp.GetHandle())

        # NOTE: this image is BGRA
        return img

    def on_off_state(self):
        """
        Uses num lock as an on/off switch to determine state
        :return: 1 if num lock is active else 0
        """
        hllDll = ctypes.WinDLL("User32.dll")
        return (hllDll.GetKeyState(win32con.VK_NUMLOCK)
                or hllDll.GetKeyState(win32con.VK_CAPITAL))

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
                  click=True, right=False, shift=False, multi=1):
        """
        clicks an area of interst

        :param x1:
        :param y1:
        :param x2:
        :param y2:
        :param speed: Define of click as percentage of defaults
        :param pause_before_click:
        :param click:
        :param right:
        :param shift: Press shift before clicking, release on completion
        :param int multi: Click multiple times

        :return: position clicked
        """

        if shift:
            keyboard.press('SHIFT')

        # check if we have actually provided a point instead of area
        if x1 == x2 and y1 == y2:
            x, y = x1, y1
        else:
            x, y = self.distribute_normally(x1, y1, x2, y2)

        if not self.client.is_inside(x, y):
            self.client.logger.debug(f'Out of bounds: {x, y}')
            return None, None

        pyautogui.moveTo(x, y)

        wait_period = self.map_between(random.random(), 0.05, 0.1)
        if pause_before_click:
            time.sleep(wait_period)

        for i in range(multi):
            self.wait_and_click(
                self.CLICK_SPEED_LOWER_BOUND * speed,
                self.CLICK_SPEED_UPPER_BOUND * speed,
                click=click,
                right=right,
            )

            if multi > 1:
                time.sleep(wait_period)

        if shift:
            keyboard.release('SHIFT')

        return x, y

    def mouse_to(self, x, y):
        if not self.client.is_inside(x, y):
            self.client.logger.debug(f'Out of bounds: {x, y}')
            return None, None
        pyautogui.moveTo(x, y)

    def mouse_to_object(self, game_object: GameObject, method=None):
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

    def mouse_away_object(self, game_object: GameObject, max_attempts=99):
        """
        Move the mouse to a random point outside the bounding box of an object
        """

        if game_object is self.client:
            raise ValueError('Cannot mouse out of entire client')

        x1, y1, x2, y2 = self.client.get_bbox()

        def new_xy():
            return int(random.uniform(x1, x2)), int(random.uniform(y1, y2))

        x, y = new_xy()
        attempts = 1
        while game_object.is_inside(x, y):
            if attempts > max_attempts:
                self.client.logger.warning(
                    f'Failed to mouse out of {game_object}')
                return

            x, y = new_xy()
            attempts += 1

        self.client.logger.info(
            f'Mouse out of {game_object} in {attempts} attempts')
        self.mouse_to(x, y)

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
