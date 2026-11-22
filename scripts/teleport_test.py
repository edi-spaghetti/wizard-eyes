"""Test a teleport method from config/teleports.json, item based ones.

Run with the bank open (``walk_test.py --object "Bank booth" --action
Bank``)::

    venv\\Scripts\\python scripts\\teleport_test.py --method varrock_tablet --slot 9
    venv\\Scripts\\python scripts\\teleport_test.py --method varrock_tablet --slot 9 --go

Steps, each with an expectation (a mismatch stops the test):

1. select the bank tag tab (``--tag``, default potion_maker)
2. deposit the inventory with the "Deposit inventory" button: inventory
   empty
3. withdraw the item from its tag slot: something in the inventory
4. close the bank: bank closed
5. find the item in the inventory by hovering (mouse text names it)
6. right-click it: the teleport option is in the menu

Without ``--go`` it stops there and moves the mouse off the menu. With
``--go`` it clicks the option and checks the landing against the wiki
landing points (within a few seconds the position must be in range of one
of them).
"""
import re
import time
from typing import Optional

from wizard_eyes.application import Application
from wizard_eyes.bank_pin import BankPinEntry
from wizard_eyes.interactions import best_menu_line, mouse_text_pattern
from wizard_eyes.teleports import Teleports, name_matches

STEP_SECONDS = 3.0
LAND_SECONDS = 8.0


class TeleportTest(Application):

    def create_parser(self):
        parser = super().create_parser()
        parser.add_argument('--method', required=True)
        parser.add_argument('--slot', type=int, required=True,
                            help='Bank tag slot of the item.')
        parser.add_argument('--tag', default='potion_maker')
        parser.add_argument('--go', action='store_true',
                            help='Actually teleport.')
        return parser

    def setup(self):
        self.args = getattr(self, 'args', None) or self.parse_args()
        self.method = Teleports().methods[self.args.method]
        if not self.method.item or 'inventory' not in self.method.where:
            raise SystemExit(f'{self.method.key}: only item methods used '
                             f'from the inventory can be tested here')
        bank = self.client.bank
        bank.tabINF.auto_locate = True
        bank.load_tag_templates([self.args.tag])
        inv = self.client.tabs.inventory
        inv.auto_locate = True
        interface = inv.interface
        for colour, (key, _) in interface.ALPHA_MAPPING.get(
                interface.widget.name, {}).items():
            interface.create_template_group(key, colour, 28)
        self.client.mouse_options.add_colours('white', 'orange', 'cyan')
        self.done = False
        self.pin: Optional[BankPinEntry] = None

    @property
    def log(self):
        return self.client.logger

    def update(self):
        self.client.bank.update()
        self.client.tabs.update()
        self.client.tabs.inventory.interface.update(True)
        self.client.mouse_options.update()
        self.client.gauges.update()

    def refresh(self, wait=0.3):
        time.sleep(wait)
        self.client.update()
        self.update()

    def wait_for(self, what: str, check, seconds=STEP_SECONDS) -> bool:
        end = time.time() + seconds
        while time.time() < end:
            self.refresh(0.25)
            if check():
                return True
        self.log.error(f'TELEPORT expected {what} within {seconds:.0f}s; '
                       f'stopping')
        return False

    def inventory_items(self):
        icons = self.client.tabs.inventory.interface.icons.values()
        return [i for i in icons if i.state not in (None, 'nothing')]

    def position(self):
        x, y, z = self.client.gauges.minimap.gps.get_coordinates()
        return None if x == -1 else (int(x), int(y), int(z))

    def run_test(self) -> bool:
        bank = self.client.bank
        m = self.method

        # 1. tag tab
        for _ in range(40):
            r = bank.select_tag(self.args.tag)
            if r:
                break
            if r is False:
                self.log.error('TELEPORT tag tab not selectable')
                return False
            self.refresh()
        self.refresh(1.0)

        # 2. empty the inventory
        if self.inventory_items():
            btn = bank.button('deposit_inventory')
            x1, y1, x2, y2 = btn.get_bbox()
            self.client.screen.click_aoi(x1 + 8, y1 + 8, x2 - 8, y2 - 8,
                                         pause_before_click=True)
            if not self.wait_for('an empty inventory',
                                 lambda: not self.inventory_items()):
                return False
        self.log.info('TELEPORT inventory empty')

        # 3. withdraw from the tag slot (check its name first)
        slot = bank.tabINF.interface.slot(self.args.slot)
        if slot is None:
            self.log.error(f'TELEPORT slot {self.args.slot} not visible')
            return False
        x1, y1, x2, y2 = slot.get_bbox()
        self.client.screen.mouse_to((x1 + x2) // 2, (y1 + y2) // 2)
        self.refresh(0.6)
        text = self.client.mouse_options.state or ''
        if not name_matches(text, m.item) or text.lower().startswith(
                'release'):
            self.log.error(f'TELEPORT slot {self.args.slot} is {text!r}, '
                           f'not {m.item[0]} (or only a placeholder)')
            return False
        self.client.screen.click_aoi(x1 + 6, y1 + 6, x2 - 6, y2 - 6)
        if not self.wait_for('the item in the inventory',
                             lambda: bool(self.inventory_items())):
            return False
        self.log.info(f'TELEPORT withdrew {text!r}')

        # 4. close the bank
        self.client.screen.press_key(bank.CLOSE_HOTKEY)
        if not self.wait_for('the bank closed', lambda: not bank.open):
            return False

        # 5. the side panel must show the inventory (another tab, e.g.
        # equipment, may have been left open), then find the item
        inv = self.client.tabs.inventory
        if not (inv.state or '').endswith('selected'):
            self._click_tab(inv)
            if not self.wait_for('the inventory tab selected',
                                 lambda: (inv.state or '').endswith(
                                     'selected')):
                return False
            self.refresh(0.6)
        item = None
        for icon in self.inventory_items():
            ix1, iy1, ix2, iy2 = icon.get_bbox()
            self.client.screen.mouse_to((ix1 + ix2) // 2, (iy1 + iy2) // 2)
            self.refresh(0.6)
            if name_matches(self.client.mouse_options.state or '', m.item):
                item = icon
                break
        if item is None:
            interface = self.client.tabs.inventory.interface
            states = {i.name: i.state for i in interface.icons.values()
                      if i.state != 'nothing'}
            self.log.error(f'TELEPORT {m.item[0]} not found in the '
                           f'inventory (located={interface.located}, '
                           f'tab state={self.client.tabs.inventory.state!r}, '
                           f'non-empty slots={states}, last mouse text='
                           f'{self.client.mouse_options.state!r})')
            return False
        self.log.info(f'TELEPORT found it in {item.name}: '
                      f'{self.client.mouse_options.state!r}')

        # 6. the option: left click if it's the default (mouse text is read
        # better than menus), else it must be in the right-click menu
        hover = self.client.mouse_options.state or ''
        if re.search(mouse_text_pattern(m.option, m.item), hover):
            self.log.info(f'TELEPORT {m.option!r} is the left-click option '
                          f'({hover!r})')
            if not self.args.go:
                self.client.screen.mouse_away_object(item)
                self.log.info('TELEPORT dry run OK (not teleporting)')
                return True
            start = self.position()
            ix1, iy1, ix2, iy2 = item.get_bbox()
            self.client.screen.click_aoi(ix1 + 4, iy1 + 4, ix2 - 4, iy2 - 4)
            return self.check_landing(start)
        if not self._right_click(item, set_ocr=True):
            self.log.error('TELEPORT right-click failed')
            return False
        menu_item = None
        end = time.time() + STEP_SECONDS
        while time.time() < end and menu_item is None:
            self.refresh(0.3)
            lines = item.context_menu.items if item.context_menu else []
            index = best_menu_line([i.value for i in lines], m.option,
                                   m.item) if lines else None
            if index is not None:
                menu_item = lines[index]
        values = [i.value for i in (item.context_menu.items
                                    if item.context_menu else [])]
        if menu_item is None:
            self.log.error(f'TELEPORT no {m.option!r} in the menu: '
                           f'{values}')
            if item.context_menu:
                self.client.screen.mouse_away_object(item.context_menu)
            return False
        self.log.info(f'TELEPORT menu has {menu_item.value!r} '
                      f'(menu: {values})')

        if not self.args.go:
            self.client.screen.mouse_away_object(item.context_menu)
            self.log.info('TELEPORT dry run OK (not teleporting)')
            return True

        start = self.position()
        menu_item.click(tmin=1.0, tmax=1.5, pause_before_click=True)
        return self.check_landing(start)

    def check_landing(self, start) -> bool:
        m = self.method
        landed: Optional[tuple] = None
        end = time.time() + LAND_SECONDS
        while time.time() < end:
            self.refresh(0.5)
            pos = self.position()
            if pos and start and max(abs(pos[0] - start[0]),
                                     abs(pos[1] - start[1])) > 20:
                time.sleep(1.5)
                self.refresh()
                landed = self.position()
                break
        if landed is None:
            self.log.error(f'TELEPORT no teleport within {LAND_SECONDS}s')
            return False
        point = m.landing_point(landed)
        if point is None:
            self.log.error(f'TELEPORT landed at {landed}, not near any '
                           f'landing point {m.lands.get("tiles")}')
            return False
        self.log.info(f'TELEPORT landed at {landed}, near {point}')
        return True

    def action(self):
        if self.done:
            return
        b = self.client.bank
        if not b.open and self.pin is None and b.pin_requested():
            self.pin = BankPinEntry(self.client)
        if self.pin is not None:
            result = self.pin.step()
            if result is None:
                self.msg.append('entering Bank PIN')
                return
            self.pin = None
            if result is False:
                self.stop()
                return
        if not (b.open and b.tags and b.tabINF.interface.located):
            self.msg.append('waiting for the bank to be open')
            return
        self.done = True
        self.run_test()
        self.stop()


def main():
    app = TeleportTest()
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
