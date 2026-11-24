"""Walk to the Grand Exchange, open it and leave it open
(``wizard_eyes.navigation.ge_visit``)::

    venv\\Scripts\\python scripts\\ge_visit.py --username NAME --run-time 150
"""
from wizard_eyes.application import Application
from wizard_eyes.navigation.ge_visit import OpenGE


class GEVisit(Application):

    def setup(self):
        self.args = getattr(self, 'args', None) or self.parse_args()
        self.client.mouse_options.add_colours('white', 'cyan', 'yellow')
        self.visit = OpenGE(self)

    def update(self):
        self.client.gauges.update()
        self.client.game_screen.update()
        self.client.mouse_options.update()
        self.visit.update()

    def action(self):
        result = self.visit.step()
        if result is None:
            return
        self.client.logger.info(
            f'GE VISIT: {"open" if result else "FAILED"}'
            + (f', empty slots {self.visit.ge.empty_slots()}' if result
               else f': {self.visit.failed}'))
        self.stop()


def main():
    app = GEVisit()
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
