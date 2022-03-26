import sys
import time
from os import _exit
from abc import ABC, abstractmethod

import cv2
import numpy
import keyboard

from .client import Client
from .file_path_utils import get_root


class Application(ABC):
    """Base class application with methods for implementation."""

    PATH = f'{get_root()}/data/recordings/{{}}.png'

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

            # log run cycle
            t2 = time.time()  # not including show image time
            self.msg.insert(0, f'Cycle {t2 - t1:.3f}')
            msg = ' - '.join(self.msg)
            self.msg_buffer.append(msg)
            if len(self.msg_buffer) > 69:
                self.msg_buffer = self.msg_buffer[1:]  # remove oldest

            sys.stdout.write(f'{msg[:self.msg_length]:{self.msg_length}}')
            sys.stdout.flush()

            self.show()

    def show(self):
        """
        Show images per client args.
        """

        # do image stuff
        images = list()
        if self.client.args.show:
            name = 'Client'
            images.append((name, self.client.original_img))

        if self.client.args.show_map:
            name = 'Map'
            gps = self.client.minimap.minimap.gps
            if gps.current_map is not None:
                images.append((name, gps.current_map.img_colour))

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

            images.append(('Logs', buffer))

        if self.client.args.show_gps:
            name = 'Gielenor Positioning System'
            gps = self.client.minimap.minimap.gps
            if gps.current_map is not None:
                images.append((name, gps.show_img))

        if images:
            for i, (name, image) in enumerate(images):
                cv2.imshow(name, image)
                widths = [im.shape[1] for _, im in images[:i]]
                cv2.moveWindow(name, 5 + sum(widths), 20)
            cv2.waitKey(1)
