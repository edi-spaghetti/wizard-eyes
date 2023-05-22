from .minimap import MiniMap
from .orbs import PrayerOrb, HitPointsOrb
from .xp_tracker import XPTracker
from ..game_objects import GameObject
from ..personal_menu import LogoutButton


class OrbsContainer:

    def __init__(self, client, parent):
        self.prayer = PrayerOrb(client, parent)
        self.hitpoints = HitPointsOrb(client, parent)


class MiniMapWidget(GameObject):

    def __init__(self, client):
        self.minimap = MiniMap(client, self)
        self.logout = LogoutButton(client, self)
        self.xp_tracker = XPTracker(client, self)
        self.orb = OrbsContainer(client, self)
        super(MiniMapWidget, self).__init__(
            client, client, config_path='minimap',
            container_name='minimap',
        )
