from .minimap import MiniMap
from .orbs import PrayerOrb, HitPointsOrb, RunEnergyOrb, SpecialAttackOrb
from .xp_tracker import XPTracker
from ..game_objects import GameObject


class OrbsContainer:

    def __init__(self, client, parent):
        self.prayer: PrayerOrb = PrayerOrb(client, parent)
        self.hitpoints: HitPointsOrb = HitPointsOrb(client, parent)
        self.run_energy: RunEnergyOrb = RunEnergyOrb(client, parent)
        self.special_attack: SpecialAttackOrb = SpecialAttackOrb(
            client, parent
        )


class MiniMapWidget(GameObject):

    def __init__(self, client):
        self.minimap = MiniMap(client, self)
        self.xp_tracker = XPTracker(client, self)
        self.orb: OrbsContainer = OrbsContainer(client, self)
        super(MiniMapWidget, self).__init__(
            client, client, config_path='minimap',
            container_name='minimap',
        )
