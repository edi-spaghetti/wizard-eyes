"""Teleporting through the house: Construction cape "Tele to POH", then the
ornate jewellery box's menu.

    tele = HouseTeleport(app, method)   # a method that needs 'house'
    ...  # every frame:
    tele.update()
    result = tele.step()   # None: working on it, True: landed, False: failed

Needs the account's house layout with the jewellery box recorded
(``scripts/house_map.py --until "Jewellery box space"``).
"""
import time
from typing import Optional

from .bank_pin import account_from_title
from .icon_teleport import IconTeleport
from .menu_teleport import ObjectMenuTeleport
from .navigation import map_labels
from .navigation.house import HouseLayout, HousePosition
from .teleports import Method, Teleports

CAPE = 'construction_cape_house'
HOUSE_SECONDS = 10.0
ROOM_SECONDS = 5.0
SETTLE_SECONDS = 1.5
"""The scene loads after the cape."""


class HouseTeleport:

    def __init__(self, app, method: Method,
                 catalogue: Optional[Teleports] = None):
        if method.needs != 'house' or not method.hotkey:
            raise ValueError(f'{method.key}: not a jewellery box method')
        self.app = app
        self.client = app.client
        self.method = method
        catalogue = catalogue or Teleports()
        self.menu = catalogue.menu_for(method)
        self.layout = HouseLayout.load(account_from_title(
            getattr(self.client, 'title', '')))
        self.tracker = HousePosition(self.layout)
        self.cape = IconTeleport(app, catalogue.methods[CAPE])
        self.state = 'cape'
        self.since = time.time()
        self.box = None
        self.tele: Optional[ObjectMenuTeleport] = None
        self.failed: Optional[str] = None

    @property
    def log(self):
        return self.client.logger

    def _fail(self, message: str) -> bool:
        self.failed = f'{self.method.key}: {message}'
        self.log.warning(f'Teleport {self.failed}')
        return False

    def update(self):
        self.tracker.update(self.client.gauges.minimap.gps)
        if self.state == 'cape':
            self.cape.update()
        if self.box is not None:
            self.app._update_game_entities(self.box)

    def _box_entity(self):
        found = self.layout.hotspots(map_labels.object_map(),
                                     'Jewellery box space')
        if not found:
            return None
        _, tiles = found[0]
        box = self.client.game_screen.create_game_entity(
            'box', 'box', (0, 0), self.client, self.client,
            tile_width=1, tile_height=1)
        box.set_global_coordinates(min(t[0] for t in tiles),
                                   max(t[1] for t in tiles), tiles[0][2])
        self.app._update_game_entities(box)
        return box

    def step(self) -> Optional[bool]:
        if self.failed is not None:
            return False
        now = time.time()

        if self.state == 'cape':
            if self.tracker.in_house:
                self.state, self.since = 'house', now
                return None
            result = self.cape.step()
            if result is False:
                return self._fail(f'the cape: {self.cape.failed}')
            if result:
                self.state, self.since = 'arriving', now
            return None

        if self.state == 'arriving':
            if self.tracker.in_house:
                self.state, self.since = 'house', now
            elif now - self.since > HOUSE_SECONDS:
                return self._fail('not in the house after the cape')
            return None

        if self.state == 'house':
            if self.tracker.house is None:
                if now - self.since > ROOM_SECONDS:
                    return self._fail('not in a known room (overlay '
                                      f'{self.tracker.template})')
                return None
            if now - self.since < SETTLE_SECONDS:
                return None
            self.box = self._box_entity()
            if self.box is None:
                return self._fail('no jewellery box in the house layout')
            self.tele = ObjectMenuTeleport(self.app, self.box, self.menu,
                                           self.method)
            self.state = 'box'
            return None

        result = self.tele.step()
        if result is False:
            return self._fail('the jewellery box')
        if result:
            self.box = None
        return result
