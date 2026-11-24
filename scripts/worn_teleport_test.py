"""Teleport with a worn item (``wizard_eyes.worn_teleport``), then open the
Grand Exchange with ``--ge``::

    venv\Scripts\python scripts\worn_teleport_test.py --method ring_of_wealth_ge --ge
"""
from wizard_eyes.application import Application
from wizard_eyes.navigation.ge_visit import OpenGE
from wizard_eyes.teleports import Teleports
from wizard_eyes.worn_teleport import WornTeleport


class WornTeleportTest(Application):

    def create_parser(self):
        parser = super().create_parser()
        parser.add_argument('--method', default='ring_of_wealth_ge')
        parser.add_argument('--ge', action='store_true')
        return parser

    def setup(self):
        self.args = getattr(self, 'args', None) or self.parse_args()
        self.client.tabs.inventory.auto_locate = True
        self.client.mouse_options.add_colours('white', 'cyan', 'yellow')
        self.tele = WornTeleport(
            self, Teleports().methods[self.args.method])
        self.visit = None

    def update(self):
        self.client.gauges.update()
        self.client.game_screen.update()
        self.client.mouse_options.update()
        (self.visit or self.tele).update()

    def action(self):
        step = self.visit or self.tele
        result = step.step()
        if result is None:
            return
        name = 'GE' if self.visit else 'TELEPORT'
        self.client.logger.info(
            f'WORN TEST {name}: {"ok" if result else "FAILED " + str(step.failed)}')
        if result and self.args.ge and self.visit is None:
            self.visit = OpenGE(self)
            return
        self.stop()


def main():
    app = WornTeleportTest()
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
