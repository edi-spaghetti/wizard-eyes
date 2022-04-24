from uuid import uuid4

import cv2
from cv2 import legacy  # noqa
import numpy

from ..game_objects.game_objects import GameObject
from . import player


class GameEntity(GameObject):
    """
    Base class to represent game entities such as the local player, other
    players and NPCs
    """

    PATH_TEMPLATE = '{root}/data/game_screen/{name}.npy'

    # enums for combat status
    UNKNOWN = -1
    NOT_IN_COMBAT = 0
    LOCAL_ATTACK = 1
    OTHER_ATTACK = 2

    # default attack speed in ticks
    DEFAULT_ATTACK_SPEED = 3

    # default colour for showing client image (note: BGRA)
    DEFAULT_COLOUR = (0, 0, 0, 255)

    def __repr__(self):
        return self.as_string

    def __str__(self):
        return self.as_string

    @property
    def as_string(self):
        return f'{self.__class__.__name__}{self.key[:2]}'

    def __init__(self, name, key, *args, tile_base=1, **kwargs):
        super(GameEntity, self).__init__(*args, **kwargs)
        self.id = uuid4().hex
        self.name = name
        self.key = key
        self._global_coordinates = None
        self.updated_at = -float('inf')
        self._attack_speed = self.DEFAULT_ATTACK_SPEED
        self.combat_status = self.UNKNOWN
        self.combat_status_updated_at = -float('inf')
        self._hit_splats_location = None
        self.colour = self.DEFAULT_COLOUR
        self.checked = False
        self.tile_base = tile_base
        self.state = None
        self.state_changed_at = None

        self._tracker = None
        self._tracker_bbox = None
        self.init_tracker()

    @property
    def tracker(self):
        if self._tracker is None:
            self._tracker = legacy.TrackerCSRT_create()

        return self._tracker

    def init_tracker(self):

        if self.name in self.client.args.tracker:
            win_name = f'{self.name} Bounding Box'
            cv2.imshow(win_name, self.big_img)
            bbox = cv2.selectROI(win_name, self.big_img)
            self.logger.info(f'Init tracker with bbox: {bbox}')
            self.tracker.init(self.big_img, bbox)

    @property
    def big_img(self):
        """Player image at the big bbox"""
        cx1, cy1, cx2, cy2 = self.client.get_bbox()
        x1, y1, x2, y2 = self.big_bbox()
        img = self.client.img
        i_img = img[y1 - cy1:y2 - cy1 + 1, x1 - cx1:x2 - cx1 + 1]

        return i_img

    def big_bbox(self):
        """
        Covers a wider area than the known central area. Useful for
        tracking the player while on the the move as the camera lags behind.
        """
        x1, y1, x2, y2 = self.get_bbox()
        margin = int(self.client.game_screen.tile_size * 1.5)
        return x1 - margin, y1 - margin, x2 + margin, y2 + margin

    def refresh(self):
        self.checked = False

    def tracker_bbox(self):

        if self._tracker_bbox is None:
            return self.get_bbox()

        return self._tracker_bbox

    def mm_bbox(self):
        """
        Get bounding box of this entity on the mini map.
        """

        mm = self.client.minimap.minimap
        player = self.client.game_screen.player

        v, w = self.key[:2]
        px, py, _, _ = player.mm_bbox()

        x1 = px + v
        y1 = py + w
        x2 = x1 + mm.tile_size
        y2 = y1 + mm.tile_size

        return x1, y1, x2, y2

    def get_bbox(self):
        """
        Calculate the bounding box for the current entity on the main screen.
        Bounding box is global, so to get pixel positions relative to client
        image do the following,

        .. code-block:: python

            x1, y1, x2, y2 = entity.ms_bbox()
            cx1, cy1, _, _ = client.get_bbox()

            # relative to image top left
            rx1 = x1 - cx1
            ry1 = y1 - cy1

        """

        # collect components
        player = self.client.game_screen.player
        mm = self.client.minimap.minimap
        cx1, cy1, cx2, cy2 = player.get_bbox()
        # convert relative to static bbox so we can use later
        px, py, _, _ = player.tile_bbox()
        if 'player' in self.client.args.tracker:
            px, py, _, _ = player.tracker_bbox()

        px = px - cx1 + 1
        py = py - cy1 + 1

        t_height, t_width, _ = player.templates['player_marker'].shape
        x, y = self.key[:2]
        x, y = x // mm.tile_size, y // mm.tile_size

        # calculate values
        x1 = cx1 + px + (t_width * x)
        y1 = cy1 + py + (t_height * y)
        x2 = x1 + (t_width * self.tile_base) - x
        y2 = y1 + (t_height * self.tile_base) - y

        return x1, y1, x2, y2

    def get_global_coordinates(self):
        return self._global_coordinates

    def set_global_coordinates(self, x, y):
        self._global_coordinates = x, y

    def set_attack_speed(self, speed):
        self._attack_speed = speed

    @property
    def attack_time(self):
        return self._attack_speed * self.client.TICK

    @property
    def is_on_screen(self):
        """True if the object is fully on screen and not obscured."""

        x1, y1, x2, y2 = self.get_bbox()
        on_screen = True
        for x, y in [(x1, y1), (x2, y2)]:
            on_screen = on_screen and self.client.is_inside(x, y)
            on_screen = on_screen and not self.client.minimap.is_inside(x, y)
            on_screen = on_screen and not self.client.tabs.is_inside(x, y)
            if self.client.tabs.active_tab is not None:
                interface = self.client.tabs.active_tab.interface
                on_screen = on_screen and not interface.is_inside(x, y)
            # TODO: dialog widgets
            # TODO: main screen widgets (bank, etc.)

        return on_screen

    def _draw_hit_splats(self):
        if f'{self.name}_hit_splats' in self.client.args.show:
            try:
                x1, y1, x2, y2 = self._hit_splats_location
                x1, y1, x2, y2 = self.localise(x1, y1, x2, y2)
                x1, y1, x2, y2 = self.client.localise(x1, y1, x2, y2)

                cv2.rectangle(
                    self.client.original_img,
                    (x1, y1), (x2, y2), self.colour, 1)
            except (TypeError, ValueError):
                # no hit splats were found, so either the location is still
                # None, or empty list. Either way, nothing to draw, so move on.
                return

    def check_hit_splats(self):
        """
        Checks the main screen bounding box for hit splats, and returns an
        appropriate enum to represent what it found.
        """

        # reset location
        self._hit_splats_location = list()

        if self.img.size == 0:
            return self.UNKNOWN

        # first collect local player hit splat templates
        splats = list()
        for colour in player.Player.COMBAT_SPLATS:
            splat = self.templates.get(colour)
            splat_mask = self.masks.get(colour)
            splats.append((splat, splat_mask))

        # check if any of local player splats could be found
        for template, mask in splats:
            try:
                matches = cv2.matchTemplate(
                    self.img, template,
                    cv2.TM_CCOEFF_NORMED, mask=mask)
            except cv2.error:
                return self.UNKNOWN
            (my, mx) = numpy.where(matches >= 0.99)
            for y, x in zip(my, mx):
                h, w = template.shape
                # set the position we found hit splat in bbox format
                self._hit_splats_location.append((x, y, x + w - 1, y + h - 1))
                # cache the draw call for later
                self.client.add_draw_call(self._draw_hit_splats)
                return self.LOCAL_ATTACK

        # TODO: other player/NPC hit splats

        return self.NOT_IN_COMBAT

    def show_bounding_boxes(self):

        if f'{self.name}_bbox' in self.client.args.show:

            cx1, cy1, _, _ = self.client.get_bbox()
            x1, y1, x2, y2 = self.mm_bbox()
            x1, y1, x2, y2 = self.client.localise(x1, y1, x2, y2)

            # draw a rect around entity on minimap
            cv2.rectangle(
                self.client.original_img, (x1, y1), (x2, y2), self.colour, 1)

            x1, y1, x2, y2 = self.get_bbox()
            # TODO: method to determine if entity is on screen (and not
            #  obstructed)
            if self.client.is_inside(x1, y1) and self.client.is_inside(x2, y2):

                # convert local to client image
                x1, y1, x2, y2 = self.client.localise(x1, y1, x2, y2)

                # draw a rect around entity on main screen
                cv2.rectangle(
                    self.client.original_img, (x1, y1), (x2, y2),
                    self.colour, 1)

        if f'{self.name}_big_bbox' in self.client.args.show:

            x1, y1, x2, y2 = self.big_bbox()
            # TODO: method to determine if entity is on screen (and not
            #  obstructed)
            if self.client.is_inside(x1, y1) and self.client.is_inside(x2, y2):

                # convert local to client image
                x1, y1, x2, y2 = self.client.localise(x1, y1, x2, y2)

                # draw a rect around entity on main screen
                cv2.rectangle(
                    self.client.original_img, (x1, y1), (x2, y2),
                    self.colour, 1)

        if f'{self.name}_tracker_bbox' in self.client.args.show:

            x1, y1, x2, y2 = self.tracker_bbox()
            # TODO: method to determine if entity is on screen (and not
            #  obstructed)
            if self.client.is_inside(x1, y1) and self.client.is_inside(x2, y2):

                # convert local to client image
                x1, y1, x2, y2 = self.client.localise(x1, y1, x2, y2)

                # draw a rect around entity on main screen
                cv2.rectangle(
                    self.client.original_img, (x1, y1), (x2, y2),
                    self.colour, 1)

        if f'{self.name}_id' in self.client.args.show:
            px, py, _, _ = self.get_bbox()
            x1, y1, _, _ = self.client.get_bbox()

            # TODO: manage this as configuration if we need to add more
            y_display_offset = -8

            cv2.putText(
                self.client.original_img, self.id[:8],
                # convert relative to client image so we can draw
                (px - x1 + 1, py - y1 + 1 + y_display_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                (0, 0, 0, 255), thickness=1
            )

        if f'{self.name}_name' in self.client.args.show:
            px, py, _, _ = self.get_bbox()
            x1, y1, _, _ = self.client.get_bbox()

            # TODO: manage this as configuration if we need to add more
            y_display_offset = 7

            cv2.putText(
                self.client.original_img, str(self.name),
                # convert relative to client image so we can draw
                (px - x1 + 1, py - y1 + 1 + y_display_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                (0, 0, 0, 255), thickness=1
            )

        if f'{self.name}_state' in self.client.args.show:
            px, _, _, py = self.get_bbox()
            x1, y1, _, _ = self.client.get_bbox()

            # TODO: manage this as configuration if we need to add more
            y_display_offset = -7

            cv2.putText(
                self.client.original_img, str(self.state),
                # convert relative to client image so we can draw
                (px - x1 + 1, py - y1 + 1 + y_display_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                self.colour, thickness=1
            )

    def update_tracker(self):

        if self.name not in self.client.args.tracker:
            return

        success, box = self.tracker.update(self.big_img)
        if success:
            x, y, w, h = [int(v) for v in box]

            # convert global coordinates
            x1, y1, x2, y2 = self.big_bbox()
            x = x1 + x - 1
            y = y1 + y - 1

            self._tracker_bbox = x, y, x + w, y + h

    def _show_combat_status(self):
        if f'{self.name}_combat_status' in self.client.args.show:
            px, _, _, py = self.get_bbox()
            x1, y1, _, _ = self.client.get_bbox()

            # TODO: manage this as configuration if we need to add more
            y_display_offset = 10

            cv2.putText(
                self.client.original_img, f'combat: {self.combat_status}',
                # convert relative to client image so we can draw
                (px - x1 + 1, py - y1 + 1 + y_display_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                self.colour, thickness=1
            )

    def update_combat_status(self):

        # pull time stamp that matches the images we'll use
        t = self.client.time

        hit_splats = self.check_hit_splats()
        cs_t = self.combat_status_updated_at
        if hit_splats == self.LOCAL_ATTACK:
            self.combat_status = self.LOCAL_ATTACK
            # hit splats usually last one tick, so if it's been more than
            # a tick since we last saw one, it's probably a new one
            # TODO: tweak this
            if (t - cs_t) > self.client.TICK * 2:
                self.combat_status_updated_at = t

        elif hit_splats == self.NOT_IN_COMBAT:
            # there is a universal 8-tick "in-combat" timer
            if (t - cs_t) > self.client.TICK * 8:
                self.combat_status = self.NOT_IN_COMBAT
                self.combat_status_updated_at = t

        self.client.add_draw_call(self._show_combat_status)

        return self.combat_status

    def calculate_relative_position(self, xy=None):
        """
        Re-calculate relative screen position,
        assuming we already have global coordinates, and assuming those
        global coordinates haven't changed since the last time we checked.
        """

        mm = self.client.minimap.minimap

        if xy is None:
            xy = mm.gps.get_coordinates()

        gxy = self.get_global_coordinates()
        rxy = tuple(map(lambda iv: iv[1] - xy[iv[0]], enumerate(gxy)))
        rxy = tuple(map(lambda v: v * mm.tile_size, rxy))

        return rxy

    def update(self, key=None):
        super().update()

        # set key for locating entity
        if key:
            self.key = key
        self.checked = True
        self.update_tracker()
        self.client.add_draw_call(self.show_bounding_boxes)

        self.updated_at = self.client.time
