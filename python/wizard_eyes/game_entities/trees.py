import cv2
import numpy

from .entity import GameEntity


class Tree(GameEntity):
    """It's a tree."""

    # TODO: scale chop timeout with player level
    CHOP_TIMEOUT = 10
    STUMP_THRESHOLD = 0.8

    def __init__(self, name, key, *args, tile_base=2, **kwargs):
        super().__init__(name, key, *args, **kwargs)
        self.tile_base = tile_base
        self.load_templates([f'{name}_stump'])
        self.load_masks([f'{name}_stump'])
        self._stump_location = None

    def mm_bbox(self):
        x1, y1, _, _ = super().mm_bbox()

        mm = self.client.minimap.minimap
        return (x1, y1,
                (x1 + mm.tile_size * self.tile_base - 1),
                (y1 + mm.tile_size * self.tile_base - 1))

    def in_base_contact(self, x, y):
        """
        Return true if the supplied coordinate is base contact.
        Assumes trees record their map coordinates on the north west
        tile and the supplied coordinates are for a 1 tile base
        (e.g. the player)
        # TODO: arbitrary entity base contact
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

            (my, mx) = numpy.where(matches >= self.STUMP_THRESHOLD)
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
                self.colour, thickness=1
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
                self.colour, thickness=1
            )

    def update(self, key=None):
        super().update(key=key)

        self.check_stumps()


class Willow(Tree):
    """It's a sad tree."""

    CHOP_TIMEOUT = 14
    DEFAULT_COLOUR = (0, 200, 55, 255)


class Oak(Tree):
    """Good. Strong. Oak."""

    CHOP_TIMEOUT = 8
    DEFAULT_COLOUR = (25, 200, 5, 255)

    def __init__(self, name, key, *args, tile_base=3, **kwargs):
        super().__init__(name, key, *args, **kwargs)
        self.tile_base = tile_base


class Magic(Tree):
    """Sparkly!"""

    DEFAULT_COLOUR = (255, 15, 0, 255)
    CHOP_TIMEOUT = 180
    STUMP_THRESHOLD = 0.5

    def __init__(self, name, key, *args, tile_base=2, **kwargs):
        super().__init__(name, key, *args, **kwargs)
        self.tile_base = tile_base
        self.stumps = ['magic_stump0', 'magic_stump90',
                       'magic_stump180', 'magic_stump270']
        self.load_templates(self.stumps)
        self.load_masks(self.stumps)
