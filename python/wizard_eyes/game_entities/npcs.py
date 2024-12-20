import re
from typing import Dict, Union, List

import cv2
import numpy

from .entity import GameEntity
from ..consumables import ConsumableSetup
from ..game_objects.template import Template, Action, EquipmentSet
from ..locomotion.obstacle import Obstacle


class NPC(GameEntity):

    MAX_HIT: int = 0
    TAG_COLOUR = [179]
    CONSUMABLES: List[ConsumableSetup] = []  # subclass in order of priority

    PATH_TEMPLATE = '{root}/data/game_screen/npc/{name}.npy'

    ATTACK_SPEED = 3
    SKIP_TASK = True
    UNLOCKED = True
    CHAT_REGEX: Union[re.Pattern, None] = None
    """Pattern used to identify chat messages about this NPC"""
    MOUSE_REGEX: str = '^$'
    """Pattern used to identify about this NPC type in e.g. attack options"""

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
            name='crystal_shard_1',
            action=Action.keep,
            noted=False,
            stackable=True,
        ),
        'Crystal shard 2': Template(
            name='crystal_shard_2',
            action=Action.keep,
            noted=False,
            stackable=True,
        ),
        'Crystal shard 3': Template(
            name='crystal_shard_3',
            action=Action.keep,
            noted=False,
            stackable=True,
        ),
        'Crystal shard 4': Template(
            name='crystal_shard_4',
            action=Action.keep,
            noted=False,
            stackable=True,
        ),
        'Crystal shard 5': Template(
            name='crystal_shard_5',
            action=Action.keep,
            noted=False,
            stackable=True,
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

    MULTI_COMBAT = False
    """If true, this monster is in multi-combat zones."""

    EQUIPMENT: EquipmentSet = EquipmentSet()
    """Set of items to be equipped when fighting this NPC."""

    PRAYERS: List[str] = []
    """List of prayers to use when fighting this NPC."""

    ROUTE: List[Obstacle] = []
    """List of obstacles to be traversed when walking to this NPC."""

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

    def in_base_contact(self, x, y):

        mm = self.client.gauges.minimap

        dist = mm.distance_between(self.key[:2], (x, y))
        dist = dist / mm.tile_size
        # 1.5 on upper end to account for corners
        if self.tile_base / 2 - 1 < dist < self.tile_base / 2 + 1.5:
            return True
        return False

    def _get_bbox_offset(self):
        mm = self.client.gauges.minimap
        return (mm.tile_size / 2) * self.key_type == self.TOP_LEFT_KEY

    def get_bbox(self):
        """
        Calculate the bounding box for the current NPC.
        This works the same as for GameEntity, but the NPC's position is
        slightly adjusted left to account for the tile base, and applies
        a bbox offset.

        """

        if self.client.game_screen.grid:
            return self._get_bbox_with_grid()
        elif self.client.game_screen.tile_marker:
            return self._get_bbox_with_tile_marker()
        else:
            return super().get_bbox()

    def _get_bbox_with_grid(self):

        mm = self.client.gauges.minimap
        g = self.client.game_screen.grid

        k0, k1 = self.key[:2]
        if k0 == -float('inf') or k1 == -float('inf'):
            return None

        x = k0 + self._get_bbox_offset()
        y = k1 + self._get_bbox_offset()

        # NPCs can also move from tile to tile, and will appear to move by
        # pixel (1/4 of a tile) when they do so - which means there is a slight
        # offset when calculating their bounding box.
        # TODO: remainder
        # remainder_x = (x % mm.tile_size) / mm.tile_size
        # remainder_y = (y % mm.tile_size) / mm.tile_size

        try:
            tx1 = int(x / mm.tile_size) - 1
            # for some reason, y needs a -1 here for NPCs, but not for
            # other entities. this is likely due to the way npc dots are
            # handled being in the bounding box centre, not the top left.
            # TODO: resolve this discrepancy
            ty1 = int(y / mm.tile_size) - 1
            x1, y1, x2, y2 = g.get_tile_bbox(tx1, ty1)
        except ValueError:
            return None

        try:
            tx2 = int((x + self.tile_width) / mm.tile_size) - 1
            ty2 = int((y + self.tile_height) / mm.tile_size) - 1
            x3, y3, x4, y4 = g.get_tile_bbox(tx2, ty2)
        except ValueError:
            return None

        bbox = x1, y1, x4, y4
        return bbox

    def _get_bbox_with_tile_marker(self):

        # collect components
        mm = self.client.gauges.minimap
        tm = self.client.game_screen.tile_marker

        k0, k1 = self.key[:2]
        if k0 == -float('inf') or k1 == -float('inf'):
            return None

        offset = (mm.tile_size / 2) * (self.key_type == self.TOP_LEFT_KEY)
        x = k0 + offset
        z = k1 + offset

        x /= mm.tile_size
        z /= mm.tile_size

        # TODO: dynamically calculate offset based on tile base
        x1 = (
                x
                # some entities don't line up properly, for various reasons,
                # so we need to adjust their position
                + self.x1_bbox_offset
                # # fixed entities are added to map with their position being
                # # top left, but npcs seem to have their dot in the middle
                - self.tile_width // 2
        )
        z1 = z - self.tile_height // 2 + self.z1_bbox_offset
        x2 = x1 + self.tile_height + self.x2_bbox_offset
        z2 = z1 + self.tile_width + self.z2_bbox_offset

        top_left = numpy.matrix([[x1, 0, z1, 1.]], dtype=float)
        bottom_right = numpy.matrix([[x2, 0, z2, 1.]], dtype=float)

        x1, y1 = tm.project(top_left)
        x2, y2 = tm.project(bottom_right)

        x1, y1, x2, y2 = self.client.globalise(x1, y1, x2, y2)

        return x1, y1, x2, y2

    def update(self, key=None):
        super(NPC, self).update(key=key)

        self.update_combat_status()
