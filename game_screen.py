import cv2

from game_objects import GameObject


class GameScreen(object):
    """Container class for anything displayed within the main game screen."""

    def __init__(self, client):
        self.client = client
        self.player = Player(
            client, self, template_names=[
                'player_marker', 'player_blue_splat', 'player_red_splat'])


class Player(GameObject):
    """Object to represent the player entity on the main game screen."""

    PATH_TEMPLATE = '{root}/data/game_screen/{name}.npy'

    def __init__(self, *args, **kwargs):
        super(Player, self).__init__(*args, **kwargs)
        self.tile_confidence = None

    def static_bbox(self):
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
        Find the bounding box of the current player tile, which should be
        slided per recent movement. Also assigns a confidence score, because
        the sliding tile area currently only works if the player is stationary.
        This will likely change in the future.

        Coordinates are global.
        """

        x1, y1, x2, y2 = self.client.get_bbox()
        cx1, cy1, cx2, cy2 = self.static_bbox()
        img = self.client.img

        # TODO: find player tile if prayer on
        # TODO: find player tile if moving
        p_img = img[cy1 - y1:cy2 - y1 + 1, cx1 - x1:cx2 - x1 + 1]
        match = cv2.matchTemplate(
            p_img, self.templates['player_marker'], cv2.TM_CCOEFF_NORMED,
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

        return tx1, ty1, tx2, ty2
