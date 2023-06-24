from typing import Tuple, Union

import cv2

from .entity import GameEntity


class Player(GameEntity):
    """Object to represent the player entity on the main game screen."""

    DEFAULT_COLOUR = (255, 0, 255)
    TILE_THRESHOLD_LOWER = (205, 31, 0, 0)
    TILE_THRESHOLD_UPPER = (255, 38, 0, 255)
    TILE_COLOUR = 123
    """int: Colour to threshold greyscale image to find true tile marker.
    Ensure this colour is something unique that won't be found on game screen
    in other objects / menus."""
    BIG_BBOX_MARGIN = 3
    """float: Bigger margin for big bbox, since the true tile marker can be
    several tiles from the static camera position."""

    ADJUST_FOR_DRAG = True

    PATH_TEMPLATE = '{root}/data/game_screen/player/{name}.npy'

    def __init__(self, *args, **kwargs):
        super(Player, self).__init__(*args, **kwargs)
        self.tile_confidence = None
        self._tile_bbox = (0, 0, 0, 0)
        self.camera_drag: Union[Tuple[int, int], None] = None
        self.single_match = False

    def get_bbox(self):
        """
        The bounding box on the main screen when the player is stationary.
        The is a slightly bigger area than one tile, where every time the
        player moves in a certain direction, the player's tile slides over to
        that side of the bounding box.

        Coordinates are global to the computer monitor.
        """

        # find the middle point of client
        # TODO: find middle from left to minimap edge right
        x1, y1, x2, y2 = self.client.get_bbox()
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2

        ch = self.client.height
        ch -= self.client.banner.height
        ch -= self.client.margin_top
        ch -= self.client.margin_bottom

        # these numbers are very magic
        x_offset = int(ch * 0.045)
        y_offset = int(ch * 0.033)

        # TODO: move these values to config
        # Assumes the game is set up to be facing North, maximum camera
        # height (with detached camera), at default zoom.
        cx1, cy1, cx2, cy2 = (
            int(mx - x_offset - 5),
            int(my - y_offset - 5),
            int(mx + x_offset + 5),
            int(my + y_offset * 2 + 5)
        )

        return cx1, cy1, cx2, cy2

    def bbox_offset(self):

        x1, y1, x2, y2 = self.get_bbox()
        tx1, ty1, _, _ = self.tile_bbox()

        w = self.tile_width
        h = self.tile_height
        x2 -= (w - 1)
        y2 -= (h - 1)

        # calculate current offset position
        ox = max([min([x2, tx1]), x1]) - x1 + 1
        oy = max([min([y2, ty1]), y1]) - y1 + 1

        return ox, oy

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

    @property
    def base_width(self):
        # TODO: runelite plugins bar might be open
        tx1, _, tx2, _ = self.tile_bbox()
        width = tx2 - tx1 + 1
        if width:
            return width
        else:
            # estimate
            _, cy1, _, cy2 = self.get_bbox()
            width = cy2 - cy1 + 1
            return int(width / 14.5)

    @property
    def base_height(self):
        _, ty1, _, ty2 = self.tile_bbox()
        height = ty2 - ty1 + 1
        if height:
            return height
        else:
            _, cy1, _, cy2 = self.get_bbox()
            width = cy2 - cy1 + 1
            return int(width / 14.5)

    def tile_bbox(self):
        """
        Return cached tile bbox, or if it hasn't been set yet, update and
        check.
        """

        # return (3399, 574, 3446, 621)

        if self._tile_bbox:
            return self._tile_bbox

        self.update_tile_marker()
        return self._tile_bbox

    def _draw_tile_marker(self):
        if 'player_tile_marker' in self.client.args.show:
            x1, y1, x2, y2 = self.client.localise(
                *self.tile_bbox(), draw=True)

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

        cx1, cy1, cx2, cy2 = self.big_bbox()

        img = self.big_colour_img
        thresh_img = cv2.inRange(
            img, self.TILE_THRESHOLD_LOWER, self.TILE_THRESHOLD_UPPER)

        # TODO: find contours
        non_zero = cv2.findNonZero(thresh_img)
        if non_zero is None:
            return self.get_bbox()

        x1 = y1 = float('inf')
        x2 = y2 = -float('inf')
        for x, y in non_zero.reshape(non_zero.shape[0], 2):
            if x < x1:
                x1 = x
            if y < y1:
                y1 = y
            if x > x2:
                x2 = x
            if y > y2:
                y2 = y

        # globalise tile coordinates to big bbox
        tx1 = x1 + cx1 - 1  # -1 for static bbox addition
        tx2 = x2 + cx1 - 1  # -1 for static bbox addition
        ty1 = y1 + cy1 - 1
        ty2 = y2 + cy1 - 1

        # add to draw call for later
        self.client.add_draw_call(self._draw_tile_marker)

        # cache and return
        self._tile_bbox = tx1, ty1, tx2, ty2
        return tx1, ty1, tx2, ty2

    def show_bounding_boxes(self):
        super().show_bounding_boxes()

        if f'{self.name}_tile_bbox' in self.client.args.show:

            cx1, cy1, _, _ = self.client.get_bbox()

            x1, y1, x2, y2 = self.tile_bbox()
            if self.client.is_inside(x1, y1) and self.client.is_inside(x2, y2):

                # convert local to client image
                x1, y1, x2, y2 = self.client.localise(
                    x1, y1, x2, y2, draw=True)

                # draw a rect around entity on main screen
                cv2.rectangle(
                    self.client.original_img, (x1, y1), (x2, y2),
                    self.colour, 1)

        if 'player_camera_drag' in self.client.args.show:
            px, _, _, py = self.get_bbox()
            x1, y1, _, _ = self.client.get_bbox()

            # TODO: manage this as configuration if we need to add more
            y_display_offset = -10

            cv2.putText(
                self.client.original_img, str(self.camera_drag),
                # convert relative to client image so we can draw
                (px - x1 + 1, py - y1 + 1 + y_display_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                self.colour, thickness=1
            )

    def update_camera_drag(self):
        """Calculate the vector ratio of how far behind the camera is the
        player's true tile."""

        x1, y1, x2, y2 = self.get_bbox()
        tx, ty, _, _ = self.tile_bbox()
        w = self.base_width
        h = self.base_height

        ox, oy = self.bbox_offset()

        # calculate ratio of how many tiles away from fixed position
        rx = (tx - (x1 + ox)) / w
        ry = (ty - (y1 + oy)) / h

        self.camera_drag = (rx, ry)
        return rx, ry

    def update(self, key=None):
        """
        Runs all update methods, which are currently, combat status and time.
        """

        if self.updated_at == self.client.time:
            return

        self.update_combat_status()
        self.update_tile_marker()
        self.update_camera_drag()
        self.client.add_draw_call(self.show_bounding_boxes)

        self.updated_at = self.client.time
