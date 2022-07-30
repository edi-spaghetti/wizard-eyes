import sys
import time
import argparse
from random import random
from os import _exit
from abc import ABC, abstractmethod

import cv2
import numpy
import keyboard

from .client import Client
from .file_path_utils import get_root
from .game_objects.game_objects import GameObject


class Application(ABC):
    """Base class application with methods for implementation."""

    PATH = f'{get_root()}/data/recordings/{{}}.png'
    INVENTORY_TEMPLATES = None
    EQUIPMENT_TEMPLATES = None

    @property
    def client_init_args(self):
        return 'RuneLite',

    @property
    def client_init_kwargs(self):
        return {}

    def __init__(self, msg_length=100):
        self.continue_ = True
        self.client = Client(*self.client_init_args, **self.client_init_kwargs)
        self.client.post_init()
        self.msg = list()
        self.msg_length = msg_length
        self.msg_buffer = list()
        self.frame_number = 0
        self.afk_timer = GameObject(self.client, self.client)
        self.parser = None
        self.args = None

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

    def create_parser(self):
        """"""
        parser = argparse.ArgumentParser()

        parser.add_argument(
            '--start-xy', nargs=2, type=int,
            required=True,
            help='Specify starting coordinates'
        )

        parser.add_argument(
            '--run-time', type=int, default=float('inf'),
            help='set a maximum run time for the script in seconds'
        )

        self.parser = parser
        return parser

    def parse_args(self):
        args, _ = self.parser.parse_known_args()
        self.args = args
        return args

    def _setup_game_entity(self, label, map_=None, count=1):
        """
        Helper method to set up any arbitrary game entity based on map nodes.
        """

        mm = self.client.minimap.minimap
        map_ = map_ or mm.gps.current_map

        nodes = map_.find(label=label)
        entities = list()
        for x, y in nodes:
            if len(entities) >= count:
                continue

            key = (int((x - self.args.start_xy[0]) * mm.tile_size),
                   int((y - self.args.start_xy[1]) * mm.tile_size))

            entity = self.client.game_screen.create_game_entity(
                label, label, key, self.client, self.client
            )
            entity.set_global_coordinates(x, y)
            if count == 1:
                return entity
            else:
                entities.append(entity)

        return entities

    @abstractmethod
    def setup(self):
        """
        Run any of the application setup required *before* entering the main
        event loop.
        """

    def _update_game_entities(self, *entities, mapping=None):
        """
        Update a list of game entities relative to player.
        Note, GPS must have been run this cycle or the key will be outdated.

        :param entities: List of game entity objects to update
        :param dict mapping: A mapping of entity name to map name. Used to
            determine if the entities provided should be checked, because
            they're on the current map and therefore positioned relative to
            player, or should be skipped because they're on a different map.
        """

        mm = self.client.minimap.minimap
        gps = self.client.minimap.minimap.gps

        for entity in entities:

            # skip entities not on the current map
            if mapping:
                map_name = mapping.get(entity.name)
                if map_name != gps.current_map.name:
                    continue

            x, y = entity.get_global_coordinates()
            px, py = gps.get_coordinates()
            key = (x - px) * mm.tile_size, (y - py) * mm.tile_size
            entity.update(key=key)

    def _add_afk_timeout(self, min_, max_):
        self.afk_timer.add_timeout(
            self.client.TICK * min_
            + random() * self.client.TICK * max_
        )

    def inventory_icons_loaded(self):
        """
        Common method to check inventory icons have been loaded before we do
        other actions.
        :return: True if all inventory icons slots have been loaded.
        """

        inv = self.client.tabs.inventory
        at = self.client.tabs.active_tab

        if len(inv.interface.icons) < self.client.INVENTORY_SIZE:
            # if inventory is open we can locate as normal
            # if inventory is disabled it means we're in the bank,
            # so the inventory is open anyway
            if at is inv or inv.state == 'disabled':
                inv.interface.locate_icons({
                    'item': {
                        'templates': self.INVENTORY_TEMPLATES,
                        'quantity': self.client.INVENTORY_SIZE},
                })

        return len(inv.interface.icons) < self.client.INVENTORY_SIZE

    def equipment_icons_loaded(self):
        """
        Common method to check icons in the equipment menu have been loaded.
        :return: True if all equipment icons slots have been loaded.
        """

        eq = self.client.tabs.equipment
        at = self.client.tabs.active_tab

        if len(eq.interface.icons) < len(self.EQUIPMENT_TEMPLATES):
            # if equipment tab is open, attempt to locate each piece of
            # equipment one at a time. They should be unique, so quantity 1.
            if at is eq:
                for equipment in self.EQUIPMENT_TEMPLATES:
                    eq.interface.locate_icons({
                        'item': {
                            'templates': [equipment],
                            'quantity': 1
                        },
                    })

        return len(eq.interface.icons) < len(self.EQUIPMENT_TEMPLATES)

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

            self.frame_number += 1

            # set up logging for new cycle
            self.msg = list()
            t1 = time.time()

            # caps lock to pause the script
            # p to exit
            # TODO: convert these to utility functions
            if not self.client.screen.on_off_state():
                msg = f'Sleeping @ {self.client.time}'
                sys.stdout.write('\b' * self.msg_length)
                sys.stdout.write(f'{msg:{self.msg_length}}')
                sys.stdout.flush()
                time.sleep(0.1)
                continue

            # ensure the client is updated every frame and run the
            # application's update method
            self.client.update()
            self.afk_timer.update()
            self.update()

            # do an action (or not, it's your life)
            if self.afk_timer.time_left:
                self.msg.append(f'AFK: {self.afk_timer.time_left:.3f}')
            else:
                self.action()

            # log run cycle
            t2 = time.time()  # not including show image time
            self.msg.insert(0, f'Cycle {self.frame_number} {t2 - t1:.3f}')
            msg = ' - '.join(self.msg)
            self.msg_buffer.append(msg)
            if len(self.msg_buffer) > 69:
                self.msg_buffer = self.msg_buffer[1:]  # remove oldest

            sys.stdout.write('\b' * self.msg_length)
            sys.stdout.write(f'{msg[:self.msg_length]:{self.msg_length}}')
            sys.stdout.flush()

            self.show()

    def show(self):
        """
        Show images per client args.
        """

        # if we're in static mode draw calls will overwrite out static image,
        # take a copy of it and swap the original back at the end.
        img = None
        if self.client.args.static_img:
            img = self.client._original_img
            self.client._original_img = img.copy()

        # all subscribed draw calls can now be executed
        for draw in self.client.draw_calls:
            draw()

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
            w, h = self.client.args.buffer_wh
            buffer = numpy.ones((h, w, 4), dtype=numpy.uint8)

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

        # set the original original image back for next loop update
        if self.client.args.static_img:
            self.client._original_img = img
