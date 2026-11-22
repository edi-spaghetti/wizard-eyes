"""Set up a Grand Exchange buy offer with every step checked
(``wizard_eyes.grand_exchange.PlaceOffer``). Start with the GE open
(``walk_test.py --object "Grand Exchange booth" --action Exchange``).

    venv\\Scripts\\python scripts\\ge_offer_test.py --item "Grimy ranarr weed" --quantity 100 --price 4500

Dry run unless ``--confirm``: it goes back without placing the offer.
"""
import time

from wizard_eyes.application import Application
from wizard_eyes.grand_exchange import GrandExchange, PlaceOffer


class GEOfferTest(Application):

    def create_parser(self):
        parser = super().create_parser()
        parser.add_argument('--item', required=True)
        parser.add_argument('--quantity', type=int, required=True)
        parser.add_argument('--price', type=int, required=True)
        parser.add_argument('--slot', type=int)
        parser.add_argument('--confirm', action='store_true',
                            help='Actually place the offer.')
        return parser

    def setup(self):
        self.args = getattr(self, 'args', None) or self.parse_args()
        self.offer = None
        self.started = time.time()

    def update(self):
        pass

    def action(self):
        if self.offer is None:
            ge = GrandExchange(self.client)
            if ge.update() != 'main':
                self.msg.append(f'waiting for the GE (screen {ge.screen})')
                if time.time() - self.started > 10:
                    self.client.logger.error('GE OFFER: the GE is not open')
                    self.stop()
                return
            self.client.logger.info(f'GE OFFER: empty slots '
                                    f'{ge.empty_slots()}')
            a = self.args
            self.offer = PlaceOffer(self, 'buy', a.item, a.quantity,
                                    a.price, confirm=a.confirm,
                                    slot=a.slot)
        result = self.offer.step()
        if result is None:
            self.msg.append(f'offer: {self.offer.state}')
            return
        self.client.logger.info(f'GE OFFER: {"done" if result else "FAILED"}'
                                f' at state {self.offer.state}')
        self.stop()


def main():
    app = GEOfferTest()
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
