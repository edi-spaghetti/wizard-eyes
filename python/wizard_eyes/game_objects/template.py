from dataclasses import dataclass
from enum import Enum
from typing import List

from numpy import ndarray
import cv2

from ..constants import REDA


class Action(Enum):
    """Represents an action that can be performed on a game object."""
    drop = 'drop'
    eat = 'eat'
    equip = '(equip|wear|wield)'
    examine = 'examine'
    alch = 'alch'
    keep = 'keep'


@dataclass
class InterfaceItem:
    """Item that can be equipped or just held in inventory."""

    template: 'Template'
    """Template name of the interface item."""
    quantity: int = 1
    """How many of that item we need."""
    slot: str = ''
    """Equipment slot the template represents, empty if not equip-able 
    (or you don't want it to be equipped)."""
    pre_pot: bool = False
    """Should we take a cheeky sip before heading off on our travels?"""


@dataclass
class EquipmentSet:
    """Set of items to be equipped when fighting a particular NPC."""

    cape: InterfaceItem = None
    helmet: InterfaceItem = None
    ammo: InterfaceItem = None
    weapon: InterfaceItem = None
    amulet: InterfaceItem = None
    shield: InterfaceItem = None
    body: InterfaceItem = None
    legs: InterfaceItem = None
    gloves: InterfaceItem = None
    boots: InterfaceItem = None
    ring: InterfaceItem = None

    extra: List[InterfaceItem] = None
    """Extra items not to be equipped, e.g. teleports."""

    def iterate_items(self, extra=True):
        """Iterates over all items in the equipment set."""
        for item in (self.cape, self.helmet, self.ammo, self.weapon,
                     self.amulet, self.shield, self.body, self.legs,
                     self.gloves, self.boots, self.ring):
            if item is not None:
                yield item
        if extra:
            for item in self.extra or []:
                yield item

    @staticmethod
    def slot_attributes():
        """Returns the equipment slot names."""
        return ('cape', 'helmet', 'ammo', 'weapon', 'amulet', 'shield',
                     'body', 'legs', 'gloves', 'boots', 'ring')


@dataclass
class TemplateGroup:
    """Represents a group of game object images that have a common purpose,
    e.g. coin_1, coin_2, coin_3, coin_4, coin_5, coin_25, coin_100, coin_250,
    coin_1000 and coin_10000 might all be grouped under the name 'coin'."""
    name: str
    templates: List['Template']
    colour: tuple = REDA
    quantity: int = 1


@dataclass
class Template:
    """Represents a single game object image."""

    # item stuff
    name: str
    """Name of template, used to load image and mask from file."""
    alias: str = None
    """Mask alias, used to load mask with a different name from template."""
    action: Enum = Action.alch
    """Action to perform on the item when we get it."""
    noted: bool = False
    stackable: bool = False

    # image stuff
    image: ndarray = None
    mask: ndarray = None
    threshold: float = .99
    method: int = cv2.TM_CCOEFF_NORMED
    invert: bool = False
