"""Test getting to a bank (``wizard_eyes.navigation.banks.BankTrip``).

    venv\\Scripts\\python scripts\\bank_trip.py --username <name>
    venv\\Scripts\\python scripts\\bank_trip.py --teleports minigame_soul_wars --force

``--force`` uses the teleport even if a bank is close. Stops once a bank is
in view (doesn't click it).
"""
from wizard_eyes.application import Application
from wizard_eyes.navigation import banks
from wizard_eyes.navigation.banks import BankTrip
from wizard_eyes.teleports import Teleports


class BankTripTest(Application):

    def create_parser(self):
        parser = super().create_parser()
        parser.add_argument('--teleports', nargs='+', default=(),
                            choices=Teleports().usable())
        parser.add_argument('--force', action='store_true')
        return parser

    def setup(self):
        self.args = getattr(self, 'args', None) or self.parse_args()
        if self.args.force:
            banks.NEAR_STEPS = -1
        self.client.tabs.inventory.auto_locate = True
        interface = self.client.tabs.inventory.interface
        for colour, (key, _) in interface.ALPHA_MAPPING.get(
                interface.widget.name, {}).items():
            interface.create_template_group(key, colour, 28)
        self.client.mouse_options.add_colours('white', 'cyan', 'orange')
        self.trip = BankTrip(self, self.args.teleports)

    def update(self):
        self.client.gauges.update()
        self.client.game_screen.update()
        self.client.mouse_options.update()
        self.trip.update()

    def action(self):
        result = self.trip.step()
        if result is None:
            return
        self.client.logger.info(
            f'BANK TRIP {"OK: " + str(self.trip.obj) if result else "FAILED: " + str(self.trip.failed)}')
        self.stop()


def main():
    app = BankTripTest()
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
