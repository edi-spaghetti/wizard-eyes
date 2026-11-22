"""Teleport to the Grand Exchange the preferred way: Construction cape
"Tele to POH" (if not in the house yet), then the house's ornate jewellery
box, hotkey L. Needs the account's house layout with the jewellery box
recorded (``scripts/house_map.py --until "Jewellery box space"``).

    venv\\Scripts\\python scripts\\ge_trip.py

Each step has an expectation; a mismatch stops the trip (never retried
more than once).
"""
import re
import time
from typing import Optional

from wizard_eyes.application import Application
from wizard_eyes.bank_pin import account_from_title
from wizard_eyes.interactions import mouse_text_pattern
from wizard_eyes.menu_teleport import ObjectMenuTeleport
from wizard_eyes.navigation.house import (HouseLayout, HousePosition,
                                          in_template_area)
from wizard_eyes.navigation.object_map import ObjectMap
from wizard_eyes.teleports import Teleports

HOUSE_SECONDS = 10.0
HOVER_TRIES = 6


class GETrip(Application):

    def create_parser(self):
        parser = super().create_parser()
        parser.add_argument('--account')
        parser.add_argument('--method', default='house_jewellery_box_ge')
        return parser

    def setup(self):
        self.args = getattr(self, 'args', None) or self.parse_args()
        account = self.args.account or account_from_title(
            getattr(self.client, 'title', ''))
        self.layout = HouseLayout.load(account)
        self.tracker = HousePosition(self.layout)
        self.teleports = Teleports()
        self.method = self.teleports.methods[self.args.method]
        self.menu = self.teleports.menu_for(self.method)
        self.om = ObjectMap()
        inv = self.client.tabs.inventory
        inv.auto_locate = True
        i = inv.interface
        for colour, (key, _) in i.ALPHA_MAPPING.get(i.widget.name,
                                                    {}).items():
            i.create_template_group(key, colour, 28)
        self.client.mouse_options.add_colours('white', 'cyan', 'orange')
        self.state = 'start'
        self.since = time.time()
        self.box = None
        self.tele: Optional[ObjectMenuTeleport] = None
        self.cape_clicked = False

    @property
    def log(self):
        return self.client.logger

    def update(self):
        self.client.gauges.update()
        self.client.game_screen.update()
        self.client.tabs.update()
        self.client.tabs.inventory.interface.update(True)
        self.client.mouse_options.update()
        self.tracker.update(self.client.gauges.minimap.gps)
        if self.box is not None:
            self._update_game_entities(self.box)

    def give_up(self, why: str):
        self.log.error(f'GE TRIP stopped: {why}')
        self.stop()

    def refresh(self, wait=0.4):
        time.sleep(wait)
        self.client.update()
        self.update()

    # steps

    def tele_to_house(self) -> bool:
        """Left-click the Construction cape in the inventory ("Tele to
        POH" is its left-click option; the option is only on that cape,
        so the option alone identifies it when the orange name isn't
        read)."""
        pat = mouse_text_pattern('Tele to POH')
        for icon in self.client.tabs.inventory.interface.icons.values():
            if icon.state in (None, 'nothing'):
                continue
            x1, y1, x2, y2 = icon.get_bbox()
            self.client.screen.mouse_to((x1 + x2) // 2, (y1 + y2) // 2)
            for _ in range(HOVER_TRIES):
                self.refresh(0.4)
                text = self.client.mouse_options.state or ''
                if re.search(pat, text):
                    self.client.screen.click_aoi(x1 + 4, y1 + 4,
                                                 x2 - 4, y2 - 4)
                    self.log.info(f'GE TRIP cape: {text!r}')
                    return True
        return False

    def box_entity(self):
        found = self.layout.hotspots(self.om, 'Jewellery box space')
        if not found:
            return None
        room, tiles = found[0]
        x = min(t[0] for t in tiles)
        y = max(t[1] for t in tiles)
        e = self.client.game_screen.create_game_entity(
            'box', 'box', (0, 0), self.client, self.client,
            tile_width=1, tile_height=1)
        e.set_global_coordinates(x, y, tiles[0][2])
        self._update_game_entities(e)
        return e

    def action(self):
        now = time.time()
        pos = self.client.gauges.minimap.gps.get_coordinates()
        if pos[0] == -1:
            self.msg.append('no position yet')
            return

        if self.state == 'start':
            if self.tracker.in_house:
                self.state = 'house'
            else:
                if not self.client.tabs.inventory.interface.located:
                    self.msg.append('waiting for the inventory')
                    return
                if not self.tele_to_house():
                    return self.give_up('no Construction cape with "Tele '
                                        'to POH" in the inventory')
                self.state, self.since = 'to_house', now
            return

        if self.state == 'to_house':
            if self.tracker.in_house:
                self.log.info(f'GE TRIP in the house at '
                              f'{self.tracker.template}')
                self.state, self.since = 'house', now
            elif now - self.since > HOUSE_SECONDS:
                return self.give_up(f'the house within {HOUSE_SECONDS:.0f}s '
                                    f'of the cape (position {pos})')
            return

        if self.state == 'house':
            if self.tracker.house is None:
                if now - self.since > 5:
                    return self.give_up(f'a known room of the house '
                                        f'(overlay {self.tracker.template})')
                self.msg.append('placing ourselves in the house')
                return
            # let the landing settle (the scene loads) before clicking
            if now - self.since < 1.5:
                return
            self.box = self.box_entity()
            if self.box is None:
                return self.give_up('no jewellery box in the house layout')
            self.tele = ObjectMenuTeleport(self, self.box, self.menu,
                                           self.method)
            self.state = 'teleport'
            return

        if self.state == 'teleport':
            result = self.tele.step()
            if result is None:
                self.msg.append(f'jewellery box: {self.tele.state}')
                return
            if result is False:
                return self.give_up('jewellery box teleport failed')
            self.box = None
            self.log.info('GE TRIP at the Grand Exchange')
            self.stop()


def main():
    app = GETrip()
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
