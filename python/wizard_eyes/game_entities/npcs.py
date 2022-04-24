import math
import random

import cv2
import numpy

from .entity import GameEntity


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
