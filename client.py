import json
import win32gui
from os.path import join, dirname

from ahk import AHK


class Client(object):

    def __init__(self, name):
        self.title = None
        self._rect = None
        self._ahk = self._get_ahk()
        self._client = self._get_client(name)

    def set_rect(self):
        """
        Sets bounding box for current client.
        Note, we're using this instead of AHK, because AHK in python has issues
        with dual monitors and I can't find an option to set CoordMode in
        python
        """

        win_handle = win32gui.FindWindow(None, self.title)
        self._rect = win32gui.GetWindowRect(win_handle)

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


def get_config(name):
    path = join(dirname(__file__), 'config', name+'.json')
    with open(path, 'r') as f:
        return json.load(f)
