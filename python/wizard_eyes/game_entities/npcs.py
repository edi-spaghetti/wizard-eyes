from enum import Enum
import math
import random
import re
from typing import Dict, Union

import cv2
import numpy

from .entity import GameEntity


class Action(Enum):
    """Decide what to do with an item once we get it."""
    keep = 1
    alch = 2


class NPC(GameEntity):

    TAG_COLOUR = [179]
    CONSUMABLES: list = []  # subclass in order of priority

    SKIP_TASK = True
    CHAT_REGEX: Union[re.Pattern, None] = None
    """Pattern used to identify chat messages about this NPC"""

    DROPS: Dict[str, Action] = {}
    SEED_DROP_TABLE = {
        'snapdragon seed': Action.keep,
        'snape grass seed': Action.keep,
    }
    GEM_DROP_TABLE = {
        'loop half of key': Action.keep,
        'tooth half of key': Action.keep,
        'rune spear': Action.alch,
        'shield left half': Action.alch,
        'dragon spear': Action.alch,
    }
    RARE_DROP_TABLE = {
        'rune 2h sword': Action.alch,
        'rune battleaxe': Action.alch,
        'rune sq shield': Action.alch,
        'rune kiteshield': Action.alch,
        'dragon med helm': Action.alch,
    }
    IORWERTH_DROP_TABLE = {
        'crystal shard': Action.keep,
    }
    KOUREND_DROP_TABLE = {
        'Ancient shard': Action.keep,
        'Dark totem base': Action.keep,
        'Dark totem middle': Action.keep,
        'Dark totem top': Action.keep,
    }

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
