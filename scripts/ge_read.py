"""Read the Grand Exchange's offer slots (read-only; the GE must be open)::

    venv\\Scripts\\python scripts\\ge_read.py --username NAME --items "Grimy toadflax" "Toadflax"
"""
import time

from wizard_eyes.application import Application
from wizard_eyes.grand_exchange import GrandExchange


class GERead(Application):

    def create_parser(self):
        parser = super().create_parser()
        parser.add_argument('--items', nargs='+', required=True)
        return parser

    def setup(self):
        self.args = getattr(self, 'args', None) or self.parse_args()
        self.ge = GrandExchange(self.client)
        self.started = time.time()

    def update(self):
        pass

    def action(self):
        if self.ge.update() != 'main':
            if time.time() - self.started > 10:
                self.client.logger.error('GE READ: the GE is not open')
                self.stop()
            return
        self.client.logger.info(f'GE READ: origin {self.ge.origin}, empty '
                                f'{self.ge.empty_slots()}')
        for view in self.ge.read_slots(self.args.items):
            self.client.logger.info(f'GE READ: {view}')
        self.stop()


def main():
    app = GERead()
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
