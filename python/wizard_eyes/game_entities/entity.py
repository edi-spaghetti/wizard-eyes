from uuid import uuid4

import cv2
from cv2 import legacy  # noqa
import numpy

from ..constants import YELLOW
from ..game_objects.game_objects import GameObject
from . import player


class GameEntity(GameObject):
    """
    Base class to represent game entities such as the local player, other
    players and NPCs
    """

    PATH_TEMPLATE = '{root}/data/game_screen/{name}.npy'
    TEMPLATE_THRESHOLD = 0.9

    # enums for combat status
    UNKNOWN = -1
    NOT_IN_COMBAT = 0
    LOCAL_ATTACK = 1
    OTHER_ATTACK = 2

    # default attack speed in ticks
    DEFAULT_ATTACK_SPEED = 3

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

    def __init__(self, name, key, *args, tile_base=1, tile_width=None,
                 tile_height=None, **kwargs):
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
        self.tile_width = tile_width or tile_base
        self.tile_height = tile_height or tile_base
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

    def tracker_bbox(self):

        if self._tracker_bbox is None:
            return self.get_bbox()

        return self._tracker_bbox

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
        p = self.client.game_screen.player
        mm = self.client.minimap.minimap
        tm = self.client.game_screen.tile_marker

        cx1, cy1, cx2, cy2 = p.get_bbox()
        h, w, _ = p.templates.get(p.template_name).shape
        cx2 -= w
        cy2 -= h
        # convert relative to static bbox so we can use later
        px, py, _, _ = p.tile_bbox()
        if 'player' in self.client.args.tracker:
            px, py, _, _ = p.tracker_bbox()

        px = max([min([cx2, px]), cx1]) - cx1
        py = max([min([cy2, py]), cy1]) - cy1

        original_x, original_y = self.key[:2]
        remainder_x = ((original_x % mm.tile_size) / mm.tile_size) * w
        remainder_y = ((original_y % mm.tile_size) / mm.tile_size) * h
        x, y = original_x // mm.tile_size, original_y // mm.tile_size

        # calculate camera drag offset
        # assumes player can been updated first
        dx, dy = p.camera_drag
        dx /= 2.5  # why 3? because max 3 tiles? or big bbox size?
        dy /= 2.5
        dx *= w
        dy *= h

        adjust_drag = p.ADJUST_FOR_DRAG

        def top_left_from_grid():
            rx1, ry1 = tm.grid.get((x, y))
            x1 = cx1 + px + rx1 + remainder_x
            y1 = cy1 + py + ry1 + remainder_y

            if adjust_drag:
                x1 += dx
                y1 += dy

            return x1, y1

        def bottom_right_from_grid():
            rx2, ry2 = tm.grid.get((x + self.tile_width, y + self.tile_height))
            x2 = cx1 + px + rx2 + remainder_x
            y2 = cy1 + py + ry2 + remainder_y

            if adjust_drag:
                x2 += dx
                y2 += dy

            return x2, y2

        def old_faithful():
            # if the coords are too far out they'll resolve to None and throw
            # a TypeError when attempting to unpack values.
            # use the old approximation method for estimation.
            x1 = cx1 + px + (w * x) + remainder_x
            y1 = cy1 + py + (h * y) + remainder_y + dy
            x2 = x1 + (w * self.tile_width) - x + remainder_x
            y2 = y1 + (h * self.tile_height) - y + remainder_y

            if adjust_drag:
                x1 += dx
                y1 += dy
                x2 += dx
                y2 += dy

            return x1, y1, x2, y2

        # calculate values
        try:
            # attempt to get values for top left from tile marker grid
            x1, y1 = top_left_from_grid()
        except TypeError:
            # if that fails, see if we can get bottom right, and work back
            try:
                x2, y2 = bottom_right_from_grid()

                x1 = x2 - self.tile_width * w
                y1 = y2 - self.tile_height * h

            except TypeError:
                # if that fails too, then fall back on to the old method
                x1, y1, x2, y2 = old_faithful()
        else:
            try:
                # attempt to get values for bottom right from tile marker grid
                x2, y2 = bottom_right_from_grid()
            except TypeError:
                # if that doesn't work, we already have top left, so we can
                # approximate from there (doesn't really matter since it's
                # off screen anyway)
                x2 = x1 + self.tile_width * w
                y2 = y1 + self.tile_height * h

        return round(x1), round(y1), round(x2), round(y2)

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

        all_bbox = '*bbox' in self.client.args.show

        if f'{self.name}_click_box' in self.client.args.show:

            cx1, cy1, _, _ = self.client.get_bbox()

            x1, y1, x2, y2 = self.click_box()
            if self.client.is_inside(x1, y1) and self.client.is_inside(x2, y2):

                # convert local to client image
                x1, y1, x2, y2 = self.client.localise(x1, y1, x2, y2)

                # draw a rect around entity on main screen
                cv2.rectangle(
                    self.client.original_img, (x1, y1), (x2, y2),
                    self.colour, 1)

        if f'{self.name}_bbox' in self.client.args.show or all_bbox:

            cx1, cy1, _, _ = self.client.get_bbox()
            x1, y1, x2, y2 = self.mm_bbox()
            x1, y1, x2, y2 = self.client.localise(x1, y1, x2, y2)

            # draw a rect around entity on minimap
            cv2.rectangle(
                self.client.original_img, (x1, y1), (x2, y2), self.colour, 1)

            x1, y1, x2, y2 = self.get_bbox()
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

        show_dist2player = (
            self.get_global_coordinates()
            and
            (f'{self.name}_dist_to_player' in self.client.args.show
             or '*dist_to_player' in self.client.args.show)
        )
        if show_dist2player:
            px, _, _, py = self.get_bbox()
            x1, y1, _, _ = self.client.get_bbox()

            # TODO: manage this as configuration if we need to add more
            y_display_offset = -7
            dist = self.client.minimap.minimap.distance_between(
                self.client.minimap.minimap.gps.get_coordinates(),
                self.get_global_coordinates())

            cv2.putText(
                self.client.original_img, f'{dist:.3f}',
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
        self.update_tracker()
        self.client.add_draw_call(self.show_bounding_boxes)

        self.updated_at = self.client.time
