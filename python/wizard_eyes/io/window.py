import abc

try:
    # windows
    import win32gui
    xdo = None
except ImportError:
    # linux
    win32gui = None
    import xdo


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


class LinWindow(Window):
    """Linux OS implementation of window."""

    def __init__(self, handle):
        self._xdo = xdo.Xdo()
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
            location = self._xdo.get_window_location(self._handle)
            x1, y1 = location.x, location.y
            size = self._xdo.get_window_size(self._handle)
            width, height = size.width, size.height
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
