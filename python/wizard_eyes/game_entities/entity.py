from enum import Enum
from uuid import uuid4
import math

import cv2
from cv2 import legacy  # noqa
import numpy

from ..constants import YELLOW
from ..game_objects.game_objects import GameObject


class GameEntity(GameObject):
    """
    Base class to represent game entities such as the local player, other
    players and NPCs
    """

    PATH_TEMPLATE = '{root}/data/game_screen/{name}.npy'
    TEMPLATE_THRESHOLD = 0.9
    COMBAT_SPLATS = ('player_blue_splat', 'player_red_splat')

    # enums for combat status
    UNKNOWN = -1
    NOT_IN_COMBAT = 0
    LOCAL_ATTACK = 1
    OTHER_ATTACK = 2

    # default attack speed in ticks
    DEFAULT_ATTACK_SPEED = 3
    DEFAULT_TILE_BASE = 1

    # default colour for showing on client image (note: BGRA)
    DEFAULT_COLOUR = YELLOW
    BIG_BBOX_MARGIN = 1.5

    def __repr__(self):
        return self.as_string

    def __str__(self):
        return self.as_string

    @property
    def as_string(self):
        return f'{self.__class__.__name__}{self.key[:2]}'

    def __init__(self, name, key, *args, tile_base=None, tile_width=None,
                 tile_height=None, **kwargs):
        super(GameEntity, self).__init__(*args, **kwargs)
        self.id = uuid4().hex
        self.name = name
        self.key = key
        """Key is a vector relative to player at (0, 0).
        Keys are usually stored as pixel distance on the minimap."""
        self._global_coordinates = None
        self._attack_speed = self.DEFAULT_ATTACK_SPEED
        self.combat_status = ''
        self.combat_status_updated_at = -float('inf')
        self.last_in_combat = -float('inf')
        self._hit_splats_location = None
        self.colour = self.DEFAULT_COLOUR
        self.checked = False
        if tile_base is None:
            self.tile_base = self.DEFAULT_TILE_BASE
        else:
            self.tile_base = tile_base
            self.DEFAULT_TILE_BASE = tile_base

        self.tile_width = tile_width or self.tile_base
        self.tile_height = tile_height or self.tile_base
        self.state = None
        self.state_changed_at = None

    def reset(self):
        """Reset all attributes to default values"""
        super().reset()
        self.id = 'none'
        self.name = 'None'
        self.key = -float('inf'), -float('inf')
        self._global_coordinates = None
        self._attack_speed = self.DEFAULT_ATTACK_SPEED
        self.combat_status = ''
        self.combat_status_updated_at = -float('inf')
        self.last_in_combat = -float('inf')
        self._hit_splats_location = None
        self.colour = self.DEFAULT_COLOUR
        self.checked = False
        self.tile_base = self.DEFAULT_TILE_BASE
        self.tile_width = self.DEFAULT_TILE_BASE
        self.tile_height = self.DEFAULT_TILE_BASE
        self.state = None
        self.state_changed_at = -float('inf')

    @property
    def big_img(self):
        """Entity image at the big bbox"""
        cx1, cy1, cx2, cy2 = self.client.get_bbox()
        x1, y1, x2, y2 = self.big_bbox()
        img = self.client.img
        i_img = img[y1 - cy1:y2 - cy1 + 1, x1 - cx1:x2 - cx1 + 1]

        return i_img

    @property
    def big_colour_img(self):
        """Entity colour image at the big bbox"""
        cx1, cy1, cx2, cy2 = self.client.get_bbox()
        x1, y1, x2, y2 = self.big_bbox()
        img = self.client.original_img
        i_img = img[y1 - cy1:y2 - cy1 + 1, x1 - cx1:x2 - cx1 + 1]

        return i_img

    def big_bbox(self):
        """
        Covers a wider area than the known central area. Useful for
        tracking the player while on the the move as the camera lags behind.
        """
        x1, y1, x2, y2 = self.get_bbox()
        margin = int(self.client.game_screen.tile_size * self.BIG_BBOX_MARGIN)
        return x1 - margin, y1 - margin, x2 + margin, y2 + margin

    def refresh(self):
        self.checked = False

    def mm_bbox(self):
        """
        Get bounding box of this entity on the mini map.
        """

        mm = self.client.minimap.minimap
        player_ = self.client.game_screen.player

        v, w = self.key[:2]
        px, py, _, _ = player_.mm_bbox()

        x1 = px + v
        y1 = py + w
        x2 = x1 + mm.tile_size
        y2 = y1 + mm.tile_size

        return round(x1), round(y1), round(x2), round(y2)

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
        mm = self.client.minimap.minimap
        tm = self.client.game_screen.tile_marker

        k0, k1 = self.key[:2]
        x = k0 / mm.tile_size
        z = k1 / mm.tile_size

        x1 = x
        z1 = z
        x2 = x1 + self.tile_height
        z2 = z1 + self.tile_width

        top_left = numpy.matrix([[x1, 0, z1, 1.]], dtype=float)
        bottom_right = numpy.matrix([[x2, 0, z2, 1.]], dtype=float)

        x1, y1 = tm.project(top_left)
        x2, y2 = tm.project(bottom_right)

        x1, y1, x2, y2 = self.client.globalise(x1, y1, x2, y2)

        return x1, y1, x2, y2

    def get_global_coordinates(self):
        return self._global_coordinates

    def set_global_coordinates(self, x, y):
        self._global_coordinates = x, y

    def distance_from_player(
            self,
            in_mode: Enum = None,
            out_mode: Enum = None):
        """Calculate current NPC distance from player.
        This is done by first doing a simply trig function on the NPC's key,
        then converting to the desired mode.

        For example if in_mode is tile mode, it means the
        NPC's key is measured in whole map tiles. If out_mode is minimap mode,
        we want to convert whatever distance we calculate into map pixels,
        which is usually 4 pixels per tile.

        Valid enums to use for in and out mode can be found at
        :attr:`wizard_eyes.client.game_screen.dfp`.

        :param in_mode: The mode of the NPC's key.  Defaults to minimap mode.
        :param out_mode: The mode to convert to. Defaults to tile mode.
        """

        gps = self.client.minimap.minimap.gps

        # sanitise modes
        if in_mode is None:
            in_mode = self.client.game_screen.dfp.minimap
        if out_mode is None:
            out_mode = self.client.game_screen.dfp.tile

        # TODO: account for terrain
        try:
            pxy = gps.get_coordinates()
            xy = self.get_global_coordinates()
            if not xy:
                v, w = self.key[:2]
                xy = v + pxy[0], w + pxy[1]
            dist = gps.sum_route(pxy, xy)
        except ValueError:
            # routing failed, fall back to simple trig
            v, w = self.key[:2]
            dist = math.sqrt((abs(v) ** 2 + abs(w) ** 2))

        # first convert distance to tile mode
        modifier = 1 / in_mode.value
        dist = dist * modifier

        # then convert to desired mode
        return dist * out_mode.value

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

    def in_base_contact(self, x, y):
        """
        Return true if the supplied coordinate is base contact.
        Assumes trees record their map coordinates on the north west
        tile and the supplied coordinates are for a 1 tile base
        (e.g. the player)
        """

        tx, ty = self.get_global_coordinates()

        y_adj = {i for i in range(-(self.tile_height - 1), 1)}
        x_adj = {i for i in range(-(self.tile_width - 1), 1)}

        return (
            # west side
            (tx - x == 1 and ty - y in y_adj)
            # north side
            or (tx - x in x_adj and ty - y == 1)
            # east side
            or (tx - x == (-1 * self.tile_width) and ty - y in y_adj)
            # south side
            or (tx - x in x_adj and ty - y == (-1 * self.tile_height))
        )

    def _draw_hit_splats(self):
        if f'{self.name}_hit_splats' in self.client.args.show:
            try:
                x1, y1, x2, y2 = self._hit_splats_location
                x1, y1, x2, y2 = self.localise(x1, y1, x2, y2, draw=True)
                x1, y1, x2, y2 = self.client.localise(
                    x1, y1, x2, y2, draw=True)

                cv2.rectangle(
                    self.client.original_img,
                    (x1, y1), (x2, y2), self.colour, 1)
            except (TypeError, ValueError):
                # no hit splats were found, so either the location is still
                # None, or empty list. Either way, nothing to draw, so move on.
                return

    def show_bounding_boxes(self):

        cboxes = {'*cbox', f'{self.name}_cbox'}
        if self.client.args.show.intersection(cboxes):
            self.draw_click_box()

        bboxes = {'*bbox', f'{self.name}_bbox'}
        if self.client.args.show.intersection(bboxes):

            cx1, cy1, _, _ = self.client.get_bbox()
            x1, y1, x2, y2 = self.mm_bbox()
            x1, y1, x2, y2 = self.client.localise(x1, y1, x2, y2, draw=True)

            # draw a rect around entity on minimap
            cv2.rectangle(
                self.client.original_img, (x1, y1), (x2, y2), self.colour, 1)

            x1, y1, x2, y2 = self.get_bbox()
            x1, y1, x2, y2 = self.client.localise(x1, y1, x2, y2, draw=True)

            # draw a rect around entity on main screen
            cv2.rectangle(
                self.client.original_img, (x1, y1), (x2, y2),
                self.colour, 1)

        big_bboxes = {'*big_bbox', f'{self.name}_big_bbox'}
        if self.client.args.show.intersection(big_bboxes):

            x1, y1, x2, y2 = self.big_bbox()
            # TODO: method to determine if entity is on screen (and not
            #  obstructed)
            if self.client.is_inside(x1, y1) and self.client.is_inside(x2, y2):

                # convert local to client image
                x1, y1, x2, y2 = self.client.localise(
                    x1, y1, x2, y2, draw=True)

                # draw a rect around entity on main screen
                cv2.rectangle(
                    self.client.original_img, (x1, y1), (x2, y2),
                    self.colour, 1)

        ids = {'*id', f'{self.name}_id'}
        if self.client.args.show.intersection(ids):
            px, py, _, _ = self.get_bbox()
            x1, y1, _, _ = self.client.get_bbox()

            # TODO: manage this as configuration if we need to add more
            y_display_offset = -8

            cv2.putText(
                self.client.original_img, self.id[:8],
                # convert relative to client image so we can draw
                (px - x1 + 1, py - y1 + 1 + y_display_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                self.colour, thickness=1
            )

        names = {'*name', f'{self.name}_name'}
        if self.client.args.show.intersection(names):
            px, py, _, _ = self.get_bbox()
            x1, y1, _, _ = self.client.get_bbox()

            # TODO: manage this as configuration if we need to add more
            y_display_offset = 7

            cv2.putText(
                self.client.original_img, str(self.name),
                # convert relative to client image so we can draw
                (px - x1 + 1, py - y1 + 1 + y_display_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                self.colour, thickness=1
            )

        states = {'*state', f'{self.name}_state'}
        if self.client.args.show.intersection(states):
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

        distances = {'*dist', f'{self.name}_distance'}
        if self.client.args.show.intersection(distances):
            px, _, _, py = self.get_bbox()
            x1, y1, _, _ = self.client.get_bbox()

            # TODO: manage this as configuration if we need to add more
            y_display_offset = 18

            cv2.putText(
                self.client.original_img,
                f'distance: {self.distance_from_player():.3f}',
                # convert relative to client image so we can draw
                (px - x1 + 1, py - y1 + 1 + y_display_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                self.colour, thickness=1
            )

    def _show_combat_status(self):
        """Show the combat status of the entity."""

        states = {'*cmb_state', f'{self.name}_cmb_state'}
        if self.client.args.show.intersection(states):
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
        """Check if the entity is in combat, determined by if it has a hit
        splat or not."""
        state = self.identify(.99)
        if state != self.combat_status:
            self.combat_status_updated_at = self.client.time
        if state:
            self.last_in_combat = self.client.time
        self.combat_status = state

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
            xy = mm.gps.get_coordinates(real=True)

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
        self.client.add_draw_call(self.show_bounding_boxes)
