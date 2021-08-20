import random
import time
from os.path import dirname, join

import ctypes
import win32gui
import win32ui
import win32con
import numpy
import cv2
import pyautogui
import keyboard
from PIL import Image


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

    CLICK_SPEED_LOWER_BOUND = 0.08
    CLICK_SPEED_UPPER_BOUND = 0.15

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
        Uses caps lock as an on/off switch to determine state
        :return: 1 if caps lock is active else 0
        """
        hllDll = ctypes.WinDLL("User32.dll")
        VK_CAPITAL = 0x14
        return hllDll.GetKeyState(VK_CAPITAL)

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
        return (value - start) / (stop - start)

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

        x, y = self.distribute_normally(x1, y1, x2, y2)
        pyautogui.moveTo(x, y)

        if pause_before_click:
            wait_period = self.map_between(random.random(), 0.05, 0.2)
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

    def save_img(self, img, path=None):

        if not path:
            path = join(dirname(__file__), 'data', 'test.png')

        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        img = Image.fromarray(img)
        img.save(path)
