import math
import random
from uuid import uuid4

import cv2
from cv2 import legacy  # noqa
import numpy

from .game_objects.game_objects import GameObject


class GameScreen(object):
    """Container class for anything displayed within the main game screen."""

    def __init__(self, client):
        self.client = client
        self._player = None
        self.default_npc = NPC

    @property
    def player(self):
        if self._player is None:
            names = ['player_marker', 'player_blue_splat', 'player_red_splat']
            player = Player(
                'player', (0, 0), self.client, self, template_names=names)
            player.load_masks(names)
            self._player = player

        return self._player

    @property
    def tile_size(self):
        # assumes 100% top down view at default zoom
        # TODO: set dynamically
        return 48

    def create_game_entity(self, type_, *args,
                           entity_templates=None, **kwargs):
        """Factory method to create entities from this module."""

        if type_ in {'npc', 'npc_tag'}:
            npc = self.default_npc(*args, **kwargs)
            templates = ['player_blue_splat', 'player_red_splat']
            npc.load_templates(templates)
            npc.load_masks(templates)
            return npc
        elif type_ == 'willow':
            tree = Willow(*args, **kwargs)
            return tree
        elif type_ == 'item':
            item = GroundItem(*args, **kwargs)
            if entity_templates:
                item.load_templates(entity_templates)
                item.load_masks(entity_templates)
            return item
        else:
            entity = GameEntity(*args, **kwargs)
            if entity_templates:
                entity.load_templates(entity_templates)
                entity.load_masks(entity_templates)
            return entity


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

        t_height, t_width = player.templates['player_marker'].shape
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
                at = self.client.tabs.active_tab
                on_screen = on_screen and not at.is_inside(x, y)
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
        for colour in Player.COMBAT_SPLATS:
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

    def update(self, key=None):
        super().update()

        # set key for locating entity
        if key:
            self.key = key
        self.checked = True
        self.update_tracker()
        self.client.add_draw_call(self.show_bounding_boxes)

        self.updated_at = self.client.time


class Player(GameEntity):
    """Object to represent the player entity on the main game screen."""

    COMBAT_SPLATS = ('player_blue_splat', 'player_red_splat')

    def __init__(self, *args, **kwargs):
        super(Player, self).__init__(*args, **kwargs)
        self.tile_confidence = None
        self._tile_bbox = None

    def get_bbox(self):
        """
        The bounding box on the main screen when the player is stationary.
        The is a slightly bigger area than one tile, where every time the
        player moves in a certain direction, the player's tile slides over to
        that side of the bounding box.

        Coordinates are global to the computer monitor.
        """
        x1, y1, x2, y2 = self.client.get_bbox()
        x_m, y_m = (x1 + x2) / 2, (y1 + y2) / 2

        # TODO: move these values to config
        # Assumes the game is set up to be facing North, maximum camera
        # height (with detached camera), at default zoom.
        cx1, cy1, cx2, cy2 = (
            int(x_m - 29), int(y_m - 17), int(x_m + 29), int(y_m + 41)
        )

        return cx1, cy1, cx2, cy2

    def mm_bbox(self):
        """
        Get the bounding box of the player's white dot on the minimap.
        Note, game tile are 4x4 pixels, and the player sits in a 3x3 white
        dot in the bottom right of the game tile. Bbox is adjusted accordingly.
        """

        mm = self.client.minimap.minimap
        mx, my, _, _ = mm.get_bbox()

        x1 = int(mx + mm.config['width'] / 2) - 3
        y1 = int(my + mm.config['height'] / 2) - 2

        x2 = x1 + (mm.tile_size - 1)
        y2 = y1 + (mm.tile_size - 1)

        return x1, y1, x2, y2

    def tile_bbox(self):
        """
        Return cached tile bbox, or if it hasn't been set yet, update and
        check.
        """

        if self._tile_bbox:
            return self._tile_bbox

        self.update_tile_marker()
        return self._tile_bbox

    def _draw_tile_marker(self):
        if 'player_tile_marker' in self.client.args.show:
            x1, y1, x2, y2 = self.client.localise(*self.tile_bbox())

            # draw the tile marker where we think it is
            cv2.rectangle(
                self.client.original_img,
                (x1, y1), (x2, y2), self.colour, 1)

            px, _, _, py = self.get_bbox()
            x1, y1, _, _ = self.client.get_bbox()
            y_display_offset = 20

            # write confidence bottom left (under combat status)
            cv2.putText(
                self.client.original_img, f'conf: {self.tile_confidence:.3f}',
                # convert relative to client image so we can draw
                (px - x1 + 1, py - y1 + 1 + y_display_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                self.colour, thickness=1
            )

    def update_tile_marker(self):
        """
        Find the bounding box of the current player tile, which should be
        slided per recent movement. Also assigns a confidence score, because
        the sliding tile area currently only works if the player is stationary.
        This will likely change in the future.

        Coordinates are global.
        """

        x1, y1, x2, y2 = self.client.get_bbox()
        cx1, cy1, cx2, cy2 = self.get_bbox()
        img = self.client.img

        # TODO: find player tile if prayer on
        # TODO: find player tile if moving
        p_img = img[cy1 - y1:cy2 - y1 + 1, cx1 - x1:cx2 - x1 + 1]
        match = cv2.matchTemplate(
            p_img, self.templates['player_marker'], cv2.TM_CCOEFF_NORMED,
            # TODO: convert to self.masks attribute
            mask=self.templates.get('player_marker_mask')
        )
        _, confidence, _, (mx, my) = cv2.minMaxLoc(match)

        self.tile_confidence = confidence

        h, w = self.templates['player_marker'].shape
        # add static bbox back in to make the coordinates global
        tx1 = mx + cx1 - 1  # -1 for static bbox addition
        ty1 = my + cy1 - 1
        tx2 = tx1 + w - 1
        ty2 = ty1 + h - 1

        # add to draw call for later
        self.client.add_draw_call(self._draw_tile_marker)

        # cache and return
        self._tile_bbox = tx1, ty1, tx2, ty2
        return tx1, ty1, tx2, ty2

    def update(self, key=None):
        """
        Runs all update methods, which are currently, combat status and time.
        """

        self.update_combat_status()
        self.update_tile_marker()
        self.update_tracker()
        self.client.add_draw_call(self.show_bounding_boxes)

        self.updated_at = self.client.time


class NPC(GameEntity):

    TAG_COLOUR = [179]

    @property
    def distance_from_player(self):
        # TODO: account for tile base
        # TODO: account for terrain
        v, w = self.key[:2]
        return math.sqrt((abs(v)**2 + abs(w)**2))

    def get_hitbox(self):
        """
        Get a random pixel within the NPCs hitbox.
        Currently relies on NPC tags with fill and border set to 100% cyan.
        TODO: make this more generalised so it would work with untagged NPCs
        TODO: convert to hit*box*, not hit*point*

        Returns global point in format (x, y)
        """

        if self.name != 'npc_tag':
            return

        if self.img.size == 0:
            return

        y, x = numpy.where(self.img == self.TAG_COLOUR)
        zipped = numpy.column_stack((y, x))

        if len(zipped) == 0:
            return

        # TODO: convert to global
        y, x = random.choice(zipped)
        x1, y1, _, _ = self.get_bbox()

        return x1 + x, y1 + y

    def show_bounding_boxes(self):
        super(NPC, self).show_bounding_boxes()

        if f'{self.name}_hitbox' in self.client.args.show:

            try:
                hx, hy = self.get_hitbox()

                if self.client.is_inside(hx, hy):
                    hx, hy, _, _ = self.client.localise(hx, hy, hx, hy)
                    cv2.circle(
                        self.client.original_img, (hx, hy), 3, self.colour,
                        thickness=1)

            except TypeError:
                self.logger.debug(
                    f'not inside: {self.get_bbox()} {self.client.get_bbox()}')

        if f'{self.name}_distance_from_player' in self.client.args.show:
            px, _, _, py = self.get_bbox()
            x1, y1, _, _ = self.client.get_bbox()

            # TODO: manage this as configuration if we need to add more
            y_display_offset = 18

            cv2.putText(
                self.client.original_img,
                f'distance: {self.distance_from_player:.3f}',
                # convert relative to client image so we can draw
                (px - x1 + 1, py - y1 + 1 + y_display_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                (0, 0, 0, 255), thickness=1
            )

    def update(self, key=None):
        super(NPC, self).update(key=key)

        self.update_combat_status()


class Willow(GameEntity):
    """It's a tree."""

    DEFAULT_COLOUR = (0, 200, 55, 255)

    def __init__(self, name, key, *args, tile_base=2, **kwargs):
        super(Willow, self).__init__(name, key, *args, **kwargs)
        self.tile_base = tile_base
        self.load_templates(['willow_stump'])
        self.load_masks(['willow_stump'])
        self.state = None
        self.state_changed_at = None
        self._stump_location = None

    def mm_bbox(self):
        x1, y1, _, _ = super(Willow, self).mm_bbox()

        mm = self.client.minimap.minimap
        return x1, y1, (x1 + mm.tile_size * 2 - 1), (y1 + mm.tile_size * 2 - 1)

    def in_base_contact(self, x, y):
        """
        Return true if the supplied coordinate is base contact.
        Assumes willow trees record their map coordinates on the north west
        tile and the supplied coordinates are for a 1 tile base
        (e.g. the player)
        """

        tx, ty = self.get_global_coordinates()

        return (
            # west side
            (tx - x == 1 and ty - y in {0, -1})
            # north side
            or (tx - x in {0, -1} and ty - y == 1)
            # east side
            or (tx - x == -2 and ty - y in {0, -1})
            # south side
            or (tx - x in {0, -1} and ty - y == -2)
        )

    def _draw_stumps(self):
        if f'{self.name}_stumps' in self.client.args.show:

            try:
                x1, y1, x2, y2 = self._stump_location
            except (ValueError, TypeError):
                # stump has not been set, nothing to draw
                return

            x1, y1, x2, y2 = self.localise(x1, y1, x2, y2)
            x1, y1, x2, y2 = self.client.localise(x1, y1, x2, y2)

            cv2.rectangle(
                self.client.original_img,
                (x1, y1), (x2, y2), self.colour, 1)

            cv2.putText(
                self.client.original_img,
                'stump', (x1, y2 + 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.25, self.colour
            )

    def check_stumps(self):

        # reset stump location
        self._stump_location = None

        for name, template in self.templates.items():
            mask = self.masks.get(name)

            try:
                matches = cv2.matchTemplate(
                    self.img, template,
                    cv2.TM_CCOEFF_NORMED, mask=mask)
            except cv2.error:
                return

            (my, mx) = numpy.where(matches >= 0.8)
            for y, x in zip(my, mx):

                # cache draw call for later
                h, w = template.shape
                self._stump_location = x, y, x + w - 1, y + h - 1
                self.client.add_draw_call(self._draw_stumps)

                # only update the state once, so we can get a time stamp on it
                if name != self.state:
                    self.state = name
                    self.state_changed_at = self.client.time
                    # clear all timeouts, because once the tree has been
                    # felled it can't be clicked anyway
                    self.clear_timeout()

                # return early, because we only need to detect a stump once
                return

        # we didn't find any stumps, so this tree is back to no state
        self.state = None

    def show_bounding_boxes(self):
        super().show_bounding_boxes()

        if f'{self.name}_player_base_contact' in self.client.args.show:
            x1, y1, x2, y2 = self.get_bbox()
            x1, y1, x2, y2 = self.client.localise(x1, y1, x2, y2)

            # TODO: manage this as configuration if we need to add more
            y_display_offset = 13

            gps = self.client.minimap.minimap.gps
            base_contact = self.in_base_contact(*gps.get_coordinates())
            cv2.putText(
                self.client.original_img, f'Contact: {base_contact}',
                # convert relative to client image so we can draw
                (x1, y2 + y_display_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                (0, 0, 0, 255), thickness=1
            )

        if f'{self.name}_distance_to_player' in self.client.args.show:
            x1, y1, x2, y2 = self.get_bbox()
            x1, y1, x2, y2 = self.client.localise(x1, y1, x2, y2)

            # TODO: manage this as configuration if we need to add more
            y_display_offset = 20

            mm = self.client.minimap.minimap
            gps = self.client.minimap.minimap.gps
            if self.get_global_coordinates() is None:
                return
            distance = mm.distance_between(
                self.get_global_coordinates(), gps.get_coordinates())
            cv2.putText(
                self.client.original_img, f'Distance: {distance:.2f}',
                # convert relative to client image so we can draw
                (x1, y2 + y_display_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                (0, 0, 0, 255), thickness=1
            )

    def update(self, key=None):
        super(Willow, self).update(key=key)

        self.check_stumps()


class GroundItem(GameEntity):

    DEFAULT_COLOUR = (0, 0, 255, 255)

    def __repr__(self):
        return f'GroundItem<{self.state} {self.key}>'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hitbox = None
        self._hitbox_info = None

    def _draw_hitbox(self):
        if f'ground_item_hitbox' in self.client.args.show:

            try:
                x1, y1, x2, y2 = self._hitbox
            except (ValueError, TypeError):
                # hitbox has not been set, nothing to draw
                return

            x1, y1, x2, y2 = self.localise(x1, y1, x2, y2)
            x1, y1, x2, y2 = self.client.localise(x1, y1, x2, y2)

            cv2.rectangle(
                self.client.original_img,
                (x1, y1), (x2, y2),
                self.colour, 1)

            cv2.putText(
                self.client.original_img,
                str(self.state), (x1, y2 + 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.25, self.colour
            )

    def update(self, key=None):
        super().update(key=key)

        self._hitbox_info = None

        x1, y1, x2, y2 = self.get_bbox()
        if (not self.client.is_inside(x1, y1)
                or not self.client.is_inside(x2, y2)):
            return

        for name, template in self.templates.items():
            mask = self.masks.get(name)

            try:
                matches = cv2.matchTemplate(
                    self.img, template,
                    cv2.TM_CCOEFF_NORMED, mask=mask)
            except cv2.error:
                return

            # TODO: configurable threshold for ground items
            (my, mx) = numpy.where(matches >= 0.6)
            for y, x in zip(my, mx):

                if name != self.state:
                    self.state = name
                    self.state_changed_at = self.client.time

                h, w = template.shape
                cx1, cy1, _, _ = self.client.get_bbox()
                x1, y1, _, _ = self.get_bbox()

                hx1, hy1, hx2, hy2 = (
                    x1 + x,
                    y1 + y - 1,
                    x1 + x + w,
                    y1 + y + h - 1
                )
                self._hitbox = hx1, hy1, hx2, hy2

                # cache draw call for later
                self.client.add_draw_call(self._draw_hitbox)

                return True

        return False
