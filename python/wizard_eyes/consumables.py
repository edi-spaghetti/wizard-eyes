from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from random import random, uniform, choice
from typing import Tuple, SupportsIndex, Type, Union


@dataclass
class ConsumableSetup:

    consumable: Type['AbstractConsumable']
    """Class of consumable to set up."""
    quantity: Union[int, float]
    """When re-gearing in the bank we should withdraw this many."""
    kwargs: dict
    """Keyword arguments to pass to the consumable on init."""
    recalculate_on_init: bool = False
    """When this consumable is set up should we recalculate it's threshold.
    For some things like food/prayer we want to recalculate, but for others,
    like anti-fire we want to take a dose immediately, but recalc would set
    it's timer before we've take the dose."""


class ConsumableType(Enum):
    potion = 'potion'
    food = 'food'


@dataclass
class AbstractConsumable(ABC):

    application: 'wizard_eyes.application.Application'
    """Reference to application class to access attributes, client etc."""

    name: SupportsIndex = 'abstract'
    """Used to find consumable in application.consumables dict."""

    type: str = 'type'
    """Used to check if some post init actions are required
    e.g. mask aliasing for potions."""

    value: object = None
    """Value against which the condition check is evaluated. Will be updated
    by recalculation function. Subclasses should implement a default value."""

    templates: Tuple[str] = ()
    """Template names that represent this consumable."""

    @property
    def out_of_supply(self):
        inv = self.application.client.tabs.inventory
        return inv.interface.choose_target_icon(*self.templates) is None

    @property
    @abstractmethod
    def target(self):
        """The value against which we calculate the condition.
        Not to be confused with application targets which define a game object
        to be clicked."""

    @abstractmethod
    def condition(self):
        """check if the consumable should be consumed."""

    @abstractmethod
    def recalculate(self, state):
        """Re-calculate the value used to check the condition.

        This method must be able to recalculate with state as None, since
        this will be done on initialisation, but it can implement recalculation
        based on state as other values too.
        """

    @abstractmethod
    def regear_condition(self):
        """If we run out of this consumable, what are the criteria for
        triggering a regear."""


@dataclass
class Food(AbstractConsumable):

    MAPPING = {
        'shark': 20,
        'bass': 13,
    }

    name: str = 'food'
    type: Enum = ConsumableType.food
    max_hp: int = 99
    value: float = 99.
    recalculate_on_init: bool = True

    @property
    def target(self):
        return self.application.client.minimap.orb.hitpoints.value

    def condition(self):
        return self.target < self.value

    def recalculate(self, state):
        food = state or choice(self.templates)
        heal = self.MAPPING[food]
        self.value = self.max_hp - (heal + random() * heal)

    def regear_condition(self):
        max_hit = self.application.client.game_screen.default_npc.MAX_HIT
        hp = self.application.client.minimap.orb.hitpoints
        return hp.value < max_hit * 2


@dataclass
class PrayerPotion(AbstractConsumable):
    name: str = 'prayer'
    type: Enum = ConsumableType.potion
    templates: Tuple[str] = (
        'prayer_potion_1',
        'prayer_potion_2',
        'prayer_potion_3',
        'prayer_potion_4',
    )
    max_prayer: int = 99
    value: float = 99.
    blessed: bool = False
    recalculate_on_init: bool = True

    @property
    def target(self):
        return self.application.client.minimap.orb.prayer.value

    def condition(self):
        return self.target < self.value

    def recalculate(self, state):
        if self.blessed:
            restore = int(self.max_prayer * .27) + 7
        else:
            restore = int(self.max_prayer * .25) + 7

        self.value = self.max_prayer - (restore + random() * restore)

    def regear_condition(self):
        max_hit = self.application.client.game_screen.default_npc.MAX_HIT
        hp = self.application.client.minimap.orb.hitpoints
        return hp.value < max_hit * 2


@dataclass
class SuperAntiPoisonPotion(AbstractConsumable):
    name: str = 'antipoison'
    type: Enum = ConsumableType.potion
    templates: Tuple[str] = (
        'super_antipoison_1',
        'super_antipoison_2',
        'super_antipoison_3',
        'super_antipoison_4',
    )
    value: float = -float('inf')

    @property
    def target(self):
        return self.application.client.time

    def condition(self):
        return self.target > self.value

    def recalculate(self, state):
        self.value = self.application.client.time + uniform(.9, 1.2) * 6 * 60

    def regear_condition(self):
        max_hit = self.application.client.game_screen.default_npc.MAX_HIT
        hp = self.application.client.minimap.orb.hitpoints
        return hp.value < max_hit * 2


@dataclass
class AntiFirePotion(AbstractConsumable):
    name: str = 'antifire'
    type: Enum = ConsumableType.potion
    templates: Tuple[str] = (
        'antifire_potion_1',
        'antifire_potion_2',
        'antifire_potion_3',
        'antifire_potion_4',
    )
    value: float = -float('inf')

    @property
    def target(self):
        return self.application.client.time

    def condition(self):
        return self.target > self.value

    def recalculate(self, state):
        self.value = self.application.client.time + uniform(.9, 1) * 6 * 60

    def regear_condition(self):
        """Don't f around with dragonfire."""
        return True
