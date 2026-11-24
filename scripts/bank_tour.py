"""Test getting to a bank with each of several teleports, one after the other:
teleport, walk to the bank, open it, close it (the teleport items must be in
the inventory). A failed one is logged and the next one tried::

    venv\\Scripts\\python scripts\\bank_tour.py --teleports strength_cape crafting_cape

Landings are logged: copy them into ``config/teleports.json``.
"""
import time

from wizard_eyes.application import Application
from wizard_eyes.interactions import UseOption
from wizard_eyes.navigation import banks
from wizard_eyes.navigation.banks import BankTrip
from wizard_eyes.teleports import Teleports

OPEN_SECONDS = 12.0
NAMES = ['Bank booth', 'Bank chest', 'Banker', 'Bank table', 'Bank counter']


class BankTour(Application):

    def create_parser(self):
        parser = super().create_parser()
        parser.add_argument('--teleports', nargs='+', required=True,
                            choices=Teleports().usable())
        return parser

    def setup(self):
        self.args = getattr(self, 'args', None) or self.parse_args()
        banks.NEAR_STEPS = -1
        banks.SAME_PLACE = -1
        self.client.tabs.inventory.auto_locate = True
        self.client.bank.tabINF.auto_locate = True
        interface = self.client.tabs.inventory.interface
        for colour, (key, _) in interface.ALPHA_MAPPING.get(
                interface.widget.name, {}).items():
            interface.create_template_group(key, colour, 28)
        self.client.mouse_options.add_colours('white', 'cyan', 'orange')
        self.todo = list(self.args.teleports)
        self.results = {}
        self.trip = self.use = None
        self.name, self.state, self.since = '', 'next', time.time()

    def update(self):
        self.client.gauges.update()
        self.client.game_screen.update()
        self.client.mouse_options.update()
        if self.state == 'trip':
            self.trip.update()
        if self.state in ('open', 'opening', 'close'):
            self.client.bank.update()
            if self.state == 'open':
                self._update_game_entities(self.trip.entity)

    def done(self, result):
        self.results[self.name] = result
        self.client.logger.info(f'TOUR {self.name}: {result}')
        self.state, self.since = 'next', time.time() + 1.5

    def action(self):
        now = time.time()
        if now < self.since:
            return
        if self.state == 'next':
            if not self.todo:
                for name, result in self.results.items():
                    self.client.logger.info(f'TOUR RESULT {name}: {result}')
                return self.stop()
            self.name = self.todo.pop(0)
            self.trip = BankTrip(self, [self.name])
            self.state, self.since = 'trip', now
            return
        if self.state == 'trip':
            result = self.trip.step()
            if self.trip.tele is None and not self.trip.teleported and \
                    self.trip.method is not None:
                return self.done('FAILED teleport')
            if result is False:
                return self.done(f'FAILED {self.trip.failed}')
            if result:
                self.use = UseOption(self, self.trip.entity, 'Bank', NAMES) \
                    if 'Bank' in self.trip.obj.actions else UseOption(
                        self, self.trip.entity, 'Use', NAMES)
                self.state, self.since = 'open', now
            return
        if self.state == 'open':
            result = self.use.step()
            if result is False:
                return self.done(f'FAILED to use {self.trip.obj}')
            if result:
                self.state = 'opening'
                self.since = now
                self.deadline = now + OPEN_SECONDS + self.trip.seconds_left
            return
        if self.state == 'opening':
            if self.client.bank.open:
                self.client.screen.press_key('esc')
                return self.done(f'ok, opened {self.trip.obj.name} at '
                                 f'{self.trip.obj.tile}')
            if self.client.bank.pin_requested():
                return self.done('FAILED: bank PIN asked')
            if now > self.deadline:
                return self.done(f'FAILED: bank not open ({self.trip.obj})')


def main():
    app = BankTour()
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
