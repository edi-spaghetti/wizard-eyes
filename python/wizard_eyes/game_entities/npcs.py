import re
from typing import Dict, Union, List

import numpy

from .entity import GameEntity
from ..consumables import ConsumableSetup
from ..game_objects.template import Template, Action, EquipmentSet


class NPC(GameEntity):

    MAX_HIT: int = 0
    TAG_COLOUR = [179]
    CONSUMABLES: List[ConsumableSetup] = []  # subclass in order of priority

    PATH_TEMPLATE = '{root}/data/game_screen/npc/{name}.npy'

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

    def in_base_contact(self, x, y):

        mm = self.client.minimap.minimap

        dist = mm.distance_between(self.key[:2], (0, 0))
        dist = dist / mm.tile_size
        # 1.5 on upper end to account for corners
        if self.tile_base / 2 - 1 < dist < self.tile_base / 2 + 1.5:
            return True
        return False


    def get_bbox(self):
        """
        Calculate the bounding box for the current NPC.
        This works the same as for GameEntity, but the NPC's position is
        slightly adjusted left to account for the tile base, and applies
        a bbox offset.

        """

        # collect components
        mm = self.client.minimap.minimap
        tm = self.client.game_screen.tile_marker

        k0, k1 = self.key[:2]
        if k0 == -float('inf') or k1 == -float('inf'):
            return None

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

    def update(self, key=None):
        super(NPC, self).update(key=key)

        self.update_combat_status()
