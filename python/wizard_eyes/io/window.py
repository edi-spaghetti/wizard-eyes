import abc
import subprocess
import re

import numpy
try:
    # windows
    import win32gui
    import win32ui
    import win32con
    xdo = None
    Xlib = None
except ImportError:
    # linux
    win32gui = None
    win32ui = None
    win32con = None
    import xdo
    import Xlib


class Window(abc.ABC):

    def __init__(self, name):
        self.name = name
        self._handle = self.get_handle(name)
        self._bbox = None

    @abc.abstractmethod
    def get_handle(self, name):
        """Return a handle to the client window.

        :param name: Name of the client to run, does not need to be exact
        :raises: NotImplementedError if open client not found
        :return: Window handle
        """

    @property
    @abc.abstractmethod
    def bbox(self):
        """Calculate and return bounding box of the window. Bounding box should
        be cached when first used, making the coordinates static. This allows
        us to do things like popping out extra side panels in Runelite without
        having to adjust all the layout.

        :return: global screen coordinates in the format (x1, y1, x2, y2).
        :rtype: tuple[int]
        """

    @abc.abstractmethod
    def activate(self):
        """Set focus on current window."""

    def resize(self, dx, dy):
        """Template method to resize current window"""

        # calculate new dimensions from delta x and y
        x1, y1, x2, y2 = self.bbox
        width = (x2 - x1) + dx
        height = (y2 - y1) + dy

        # call the OS-specific method
        self._resize(x1, y1, width, height)

        # invalidate bounding box cache, it will be re-calculated next time it
        # is needed
        self._bbox = None

    @abc.abstractmethod
    def _resize(self, x, y, width, height):
        """Abstract method to resize, depends on OS implementation."""

    @property
    @abc.abstractmethod
    def img(self):
        """Grab window image at current bounding box.

        :returns: Window image as a numpy array.
        """


class WinWindow(Window):
    """Windows OS implementation of window."""

    def get_handle(self, name):
        """
        Return a handle to the client window.

        :param name: Name of the client to run, does not need to be exact.
        :raises: NotImplementedError if open client not found
        :return: Window object id
        :rtype: int
        """

        # TODO: support multiple game clients.
        handle = win32gui.FindWindow(None, name)
        if handle:
            # if client is not open, win32gui returns 0
            return handle

        raise NotImplementedError(f'No window found with name: {name}')

    @property
    def bbox(self):
        """Get bounding box for current window."""

        if not self._bbox:
            self._bbox = win32gui.GetWindowRect(self._handle)

        return self._bbox

    def activate(self):
        """Activate the current window."""
        win32gui.SetForegroundWindow(self._handle)

    def _resize(self, x, y, width, height):
        """Resize window with win32gui api."""
        win32gui.MoveWindow(self._handle, x, y, width, height, True)

    @property
    def img(self):
        """
        Grab an area of the screen and return as numpy array
        """
        hwin = win32gui.GetDesktopWindow()

        x1, y1, x2, y2 = self.bbox
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
        memdc.BitBlt(
            (0, 0), (width, height), srcdc, (x1, y1), win32con.SRCCOPY
        )
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


class LinWindow(Window):
    """Linux OS implementation of window."""

    DEPTH = 4

    def __init__(self, handle):
        self._xdo = xdo.Xdo()
        self._display = Xlib.display.Display()
        super().__init__(handle)

    def get_handle(self, name):
        """
        Return a handle to the client window

        :param str name: Name of the client to run, can be regex formatted
        :raises: NotImplementedError if window not found.
        :return: Window id
        :rtype: int
        """

        name = name.encode()
        for handle in self._xdo.search_windows(name):

            # TODO: ensure compatibility with other clients
            #       currently runelite will create two windows, one named
            #       'net-runelite-launcher-launcher' and another named
            #       'RuneLite - <player name>'. Xdo searches windows case
            #       insensitively, but we can rely on case sensitive string
            #       membership to find the correct window.
            title = self._xdo.get_window_name(handle)
            title = title.decode()
            if name.decode() in title:
                return handle

        raise NotImplementedError(f'Window with name: {name} not found')

    @property
    def bbox(self):
        """Get bounding box for current window."""

        if not self._bbox:
            # TODO: calculate xdo offsets without xprop
            #       xprop values don't update if window moves/resizes after
            #       after initial launch, and running a subprocess is
            #       inefficient if we ever get round to implementing dynamic
            #       bbox for client window
            # location = self._xdo.get_window_location(self._handle)
            # _x1, _y1 = location.x, location.y
            # size = self._xdo.get_window_size(self._handle)
            # width, height = size.width, size.height

            result = subprocess.check_output(
                ['xprop', '-id', str(self._handle)]
            )
            result = result.decode()
            match = re.search(
                r'program specified location: ([0-9]+), ([0-9]+)',
                result
            )
            x1, y1 = match.groups()
            x1 = int(x1)
            y1 = int(y1)

            match = re.search(
                r'program specified size: ([0-9]+) by ([0-9]+)',
                result
            )
            width, height = match.groups()
            width = int(width)
            height = int(height)

            x2 = x1 + width - 1
            y2 = y1 + height - 1

            self._bbox = (x1, y1, x2, y2)

        return self._bbox

    def activate(self):
        """Activate the current window"""
        self._xdo.activate_window(self._handle)

    def _resize(self, x, y, width, height):
        """Resize current window with Xdo api."""
        self._xdo.set_window_size(self._handle, width, height)

    @property
    def img(self):

        # get dimensions
        x1, y1, x2, y2 = self.bbox
        width = x2 - x1 + 1
        height = y2 - y1 + 1

        # # grab that area of screen
        ximage = self._display.screen().root.get_image(
            x1, y1, width, height, Xlib.X.ZPixmap,
            0xffffffff
        )

        # # convert to numpy array
        img = numpy.frombuffer(ximage.data, dtype=numpy.uint8)
        img = img.reshape((height, width, self.DEPTH))

        return img
