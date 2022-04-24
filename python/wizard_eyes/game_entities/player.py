import cv2

from .entity import GameEntity


class Player(GameEntity):
    """Object to represent the player entity on the main game screen."""

    COMBAT_SPLATS = ('player_blue_splat', 'player_red_splat')
    DEFAULT_COLOUR = (255, 0, 255)

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

        margin = 5
        cx1 -= margin
        cy1 -= margin
        cx2 += margin
        cy2 += margin

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
        # use the original image because the tile border is so thin we need to
        # rely on the colour more heavily.
        img = self.client.original_img

        # TODO: find player tile if prayer on
        # TODO: find player tile if moving
        p_img = img[cy1 - y1:cy2 - y1 + 1, cx1 - x1:cx2 - x1 + 1]
        match = cv2.matchTemplate(
            p_img, self.templates['player_marker'], cv2.TM_CCOEFF_NORMED,
            # TODO: convert to self.masks attribute
            mask=self.masks.get('player_marker')
        )
        _, confidence, _, (mx, my) = cv2.minMaxLoc(match)

        self.tile_confidence = confidence

        h, w, _ = self.templates['player_marker'].shape
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
