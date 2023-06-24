from enum import Enum
import math
import random
import re
from typing import Dict, Union, List

import cv2
import numpy

from .entity import GameEntity
from ..consumables import ConsumableSetup
from ..game_objects.template import Template, Action, EquipmentSet


class NPC(GameEntity):

    MAX_HIT: int = 0
    TAG_COLOUR = [179]
    CONSUMABLES: List[ConsumableSetup] = []  # subclass in order of priority

    ATTACK_SPEED = 3
    SKIP_TASK = True
    CHAT_REGEX: Union[re.Pattern, None] = None
    """Pattern used to identify chat messages about this NPC"""

    DROPS: Dict[str, Template] = {}
    SEED_DROP_TABLE = {
        'Snapdragon seed': Template(
            name='snapdragon_seed',
            action=Action.keep,
            noted=False,
            stackable=True,
        ),
        'Snape grass sseed': Template(
            name='snape_grass_seed',
            action=Action.keep,
            noted=False,
            stackable=True,
        ),
    }
    GEM_DROP_TABLE = {
        'Loop half of key': Template(
            name='loop_half_of_key',
            action=Action.keep,
            noted=False,
            stackable=False,
        ),
        'Tooth half of key': Template(
            name='tooth_half_of_key',
            action=Action.keep,
            noted=False,
            stackable=False,
        ),
        'Rune spear': Template(
            name='rune_spear',
            action=Action.alch,
            noted=False,
            stackable=False,
        ),
        'Shield left half': Template(
            name='shield_left_half',
            action=Action.alch,
            noted=False,
            stackable=False,
        ),
        'Dragon spear': Template(
            name='dragon_spear',
            action=Action.alch,
            noted=False,
            stackable=False,
        ),
    }
    RARE_DROP_TABLE = {
        'Rune 2h sword': Template(
            name='rune_2h_sword',
            action=Action.alch,
            noted=False,
            stackable=False,
        ),

        'Rune battleaxe': Template(
            name='rune_battleaxe',
            action=Action.alch,
            noted=False,
            stackable=False,
        ),
        'Rune sq shield': Template(
            name='rune_sq_shield',
            action=Action.alch,
            noted=False,
            stackable=False,
        ),
        'Rune kiteshield': Template(
            name='rune_kiteshield',
            action=Action.alch,
            noted=False,
            stackable=False,
        ),
        'Dragon med helm': Template(
            name='dragon_med_helm',
            action=Action.alch,
            noted=False,
            stackable=False,
        ),
    }
    IORWERTH_DROP_TABLE = {
        'Crystal shard': Template(
            name='crystal_shard',
            action=Action.keep,
            noted=False,
            stackable=False,
        ),
    }
    KOUREND_DROP_TABLE = {
        'Ancient shard': Template(
            name='ancient_shard',
            action=Action.keep,
            noted=False,
            stackable=False,
        ),
        'Dark totem base': Template(
            name='dark_totem_base',
            action=Action.keep,
            noted=False,
            stackable=False,
        ),
        'Dark totem middle': Template(
            name='dark_totem_middle',
            action=Action.keep,
            noted=False,
            stackable=False,
        ),
        'Dark totem top': Template(
            name='dark_totem_top',
            action=Action.keep,
            noted=False,
            stackable=False,
        ),
    }

    EQUIPMENT: EquipmentSet = EquipmentSet()
    """Set of items to be equipped when fighting this NPC."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x1_bbox_offset = 0
        self.z1_bbox_offset = 0
        self.x2_bbox_offset = 0
        self.z2_bbox_offset = 0

    def reset(self):
        super().reset()
        self.x1_bbox_offset = 0
        self.z1_bbox_offset = 0
        self.x2_bbox_offset = 0
        self.z2_bbox_offset = 0

    @classmethod
    def class_name(cls):
        return cls.__name__

    @classmethod
    def equipment_names(cls):
        names = []
        for item in cls.EQUIPMENT.iterate_items(extra=False):
            names.append(item.template.name)
        return names

    @classmethod
    def extra_equipment_names(cls):
        names = []
        for item in cls.EQUIPMENT.extra:
            names.append(item.template.name)
        return names

    @classmethod
    def drop_names(cls):
        names = []
        for item in cls.DROPS.values():
            names.append(item.name)
        return names

    @classmethod
    def get_template_by_name(cls, name: str) -> Union[Template, None]:
        for item in cls.EQUIPMENT.iterate_items(extra=True):
            if item.template.name == name:
                return item.template

        for template in cls.DROPS.values():
            if template.name == name:
                return template

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

        :param in_mode: The mode of the NPC's key.  Defaults to minimap mode.
        :param out_mode: The mode to convert to. Defaults to tile mode.
        """

        # sanitise modes
        if in_mode is None:
            in_mode = self.client.game_screen.dfp.minimap
        if out_mode is None:
            out_mode = self.client.game_screen.dfp.tile

        # TODO: account for tile base
        # TODO: account for terrain
        v, w = self.key[:2]
        dist = math.sqrt((abs(v)**2 + abs(w)**2))

        # first convert distance to tile mode
        modifier = 1 / in_mode.value
        dist = dist * modifier

        # then convert to desired mode
        return dist * out_mode.value


    def get_bbox(self):
        """
        Calculate the bounding box for the current NPC.
        This works the same as for GameEntity, but the NPC's position is
        slightly adjusted left to account for the tile base.

        """

        # collect components
        mm = self.client.minimap.minimap
        tm = self.client.game_screen.tile_marker

        k0, k1 = self.key[:2]
        x = k0 / mm.tile_size
        z = k1 / mm.tile_size

        # TODO: dynamically calculate offset based on tile base
        x1 = (
                x
                # some entities don't line up properly, for various reasons,
                # so we need to adjust their position
                + self.x1_bbox_offset
                # fixed entities are added to map with their position being
                # top left, but npcs seem to have their dot in the middle
                - self.tile_width // 2
        )
        z1 = z + self.z1_bbox_offset
        x2 = x1 + self.tile_height + self.x2_bbox_offset
        z2 = z1 + self.tile_width + self.z2_bbox_offset

        top_left = numpy.matrix([[x1, 0, z1, 1.]], dtype=float)
        bottom_right = numpy.matrix([[x2, 0, z2, 1.]], dtype=float)

        x1, y1 = tm.project(top_left)
        x2, y2 = tm.project(bottom_right)

        x1, y1, x2, y2 = self.client.globalise(x1, y1, x2, y2)

        return x1, y1, x2, y2

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
                    hx, hy, _, _ = self.client.localise(
                        hx, hy, hx, hy, draw=True)
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
                f'distance: {self.distance_from_player(True):.3f}',
                # convert relative to client image so we can draw
                (px - x1 + 1, py - y1 + 1 + y_display_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                (0, 0, 0, 255), thickness=1
            )

    def update(self, key=None):
        super(NPC, self).update(key=key)

        self.update_combat_status()
