import time
import random
from uuid import uuid4

import cv2
import numpy

from game_objects import GameObject


class GameScreen(object):
    """Container class for anything displayed within the main game screen."""

    def __init__(self, client):
        self.client = client
        names = ['player_marker', 'player_blue_splat', 'player_red_splat']
        self.player = Player(
            client, self, template_names=names)
        self.player.load_masks(names)

    def create_game_entity(self, type_, *args, **kwargs):
        """Factory method to create entities from this module."""

        if type_ in {'npc', 'npc_tag'}:
            npc = NPC(*args, **kwargs)
            templates = ['player_blue_splat', 'player_red_splat']
            npc.load_templates(templates)
            npc.load_masks(templates)
            return npc


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

    def __init__(self, *args, **kwargs):
        super(GameEntity, self).__init__(*args, **kwargs)
        self.name = None
        self.updated_at = -float('inf')
        self._attack_speed = self.DEFAULT_ATTACK_SPEED
        self.combat_status = self.UNKNOWN
        self.combat_status_updated_at = -float('inf')

    def set_attack_speed(self, speed):
        self._attack_speed = speed

    @property
    def attack_time(self):
        return self._attack_speed * self.client.TICK

    def check_hit_splats(self):
        """
        Checks the main screen bounding box for hit splats, and returns an
        appropriate enum to represent what it found.
        """

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

                if f'{self.name}_hit_splats' in self.client.args.show:
                    h, w = template.shape
                    cx1, cy1, _, _ = self.client.get_bbox()
                    x1, y1, _, _ = self.get_bbox()

                    cv2.rectangle(
                        self.client.original_img,
                        # convert relative to client image so we can draw
                        ((x1 - cx1) + x, (y1 - cy1) + y - 1),
                        ((x1 - cx1) + x + w, (y1 - cy1) + y + h - 1),
                        (0, 0, 255, 255), 1)

                return self.LOCAL_ATTACK

        # TODO: other player/NPC hit splats

        return self.NOT_IN_COMBAT

    def update_combat_status(self):

        # pull time stamp that matches the images we'll use
        t = self.client.time

        hit_splats = self.check_hit_splats()
        cs_t = self.combat_status_updated_at
        if hit_splats == self.LOCAL_ATTACK:
            self.combat_status = self.LOCAL_ATTACK
            if (t - cs_t) > self.client.TICK * 2:
                self.combat_status_updated_at = t

        elif hit_splats == self.NOT_IN_COMBAT:
            # there is a universal 8-tick "in-combat" timer
            if (t - cs_t) > self.client.TICK * 8:
                self.combat_status = self.NOT_IN_COMBAT
                self.combat_status_updated_at = t

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
                (0, 0, 0, 255), thickness=1
            )

        return self.combat_status


class Player(GameEntity):
    """Object to represent the player entity on the main game screen."""

    COMBAT_SPLATS = ('player_blue_splat', 'player_red_splat')

    def __init__(self, *args, **kwargs):
        super(Player, self).__init__(*args, **kwargs)
        self.name = 'player'
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

    def tile_bbox(self):
        """
        Return cached tile bbox, or if it hasn't been set yet, update and
        check.
        """

        if self._tile_bbox:
            return self._tile_bbox

        self.update_tile_marker()
        return self._tile_bbox

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

        if 'player_tile_marker' in self.client.args.show:

            # draw the tile marker where we think it is
            cv2.rectangle(
                self.client.original_img,
                # convert relative to client image so we can draw
                (tx1 - x1 + 1, ty1 - y1 + 1),
                (tx2 - x1 + 1, ty2 - y1 + 1),
                (255, 255, 255, 255), 1)

            px, _, _, py = self.get_bbox()
            x1, y1, _, _ = self.client.get_bbox()
            y_display_offset = 20

            # write confidence bottom left (under combat status)
            cv2.putText(
                self.client.original_img, f'conf: {self.tile_confidence:.3f}',
                # convert relative to client image so we can draw
                (px - x1 + 1, py - y1 + 1 + y_display_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                (0, 0, 0, 255), thickness=1
            )

        # cache and return
        self._tile_bbox = tx1, ty1, tx2, ty2
        return tx1, ty1, tx2, ty2

    def update(self):
        """
        Runs all update methods, which are currently, combat status and time.
        """

        self.update_combat_status()
        self.update_tile_marker()

        self.updated_at = self.client.time


class NPC(GameEntity):

    TAG_COLOUR = [179]

    def __init__(self, client, parent, name, v, w, x, y, z, tile_base=1):
        super(NPC, self).__init__(client, parent)

        self.id = uuid4().hex
        self.key = v, w, x, y, z
        self.name = name
        self.checked = False
        # TODO: add to configuration
        self.tile_base = tile_base

    # TODO: refactor to mm_bbox

    @property
    def mm_x(self):
        """top left X pixel on the minimap"""
        mm = self.client.minimap.minimap
        mm_x1 = mm.get_bbox()[0]
        x1 = self.client.get_bbox()[0]
        x = self.key[0]

        nx = int(mm_x1 - x1 + mm.config['width'] / 2 + x * mm.tile_size)
        return nx

    @property
    def mm_y(self):
        """top left Y pixel on the minimap"""
        mm = self.client.minimap.minimap
        mm_y1 = mm.get_bbox()[1]
        y1 = self.client.get_bbox()[1]
        y = self.key[1]

        ny = int(mm_y1 - y1 + mm.config['height'] / 2 + y * mm.tile_size)
        return ny

    def get_bbox(self):
        """
        Calculate the bounding box for the current NPC on the main screen.
        Bounding box is global, so to get pixel positions relative to client
        image do the following,

        .. code-block:: python

            x1, y1, x2, y2 = npc.ms_bbox()
            cx1, cy1, _, _ = client.get_bbox()

            # relative to image top left
            rx1 = x1 - cx1
            ry1 = y1 - cy1

        """

        # collect components
        player = self.client.game_screen.player
        cx1, cy1, cx2, cy2 = player.get_bbox()
        px, py, _, _ = player.tile_bbox()
        # convert relative to static bbox so we can use later
        px = px - cx1 + 1
        py = py - cy1 + 1

        t_height, t_width = player.templates['player_marker'].shape
        x, y = self.key[:2]

        # calculate values
        x1 = cx1 + px + (t_width * x)
        y1 = cy1 + py + (t_height * y)
        x2 = x1 + (t_width * 2) - x
        y2 = y1 + (t_height * 2) - y

        return x1, y1, x2, y2

    def get_hitbox(self):
        """
        Get a random pixel within the NPCs hitbox.
        Currently relies on NPC tags with fill and border set to 100% cyan.
        TODO: make this more generalised so it would work with untagged NPCs

        Returns coords in format (y, x) local to NPC hitbox.
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

        return random.choice(zipped)

    def refresh(self):
        self.checked = False

    def update(self, key=None):

        # TODO: set from client global time
        t = time.time()

        # set key for locating NPC
        if key:
            self.key = key

        self.update_combat_status()

        self.updated_at = t
        self.checked = True
