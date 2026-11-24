"""Hop to a friend's world and / or trade with a player
(``wizard_eyes.friends``, ``wizard_eyes.player_trade``). For a trade, run one
on each side with the same ``--id``; one of them asks::

    player_trade_test.py --partner NAME --hop --id t1 --role cleaner --other mixer --ask --give Cadantine 10 0
    player_trade_test.py --partner NAME --id t1 --role mixer --other cleaner

``--give ITEM QUANTITY SLOT``: the stack's inventory slot (0-27).
"""
import time

from wizard_eyes.application import Application
from wizard_eyes.coordination import Coordinator
from wizard_eyes.friends import HopToFriend
from wizard_eyes.player_trade import Give, PlayerTrade, TradeSignal


class PlayerTradeTest(Application):

    def create_parser(self):
        parser = super().create_parser()
        parser.add_argument('--partner', required=True)
        parser.add_argument('--hop', action='store_true')
        parser.add_argument('--id', help='Trade id: do a trade.')
        parser.add_argument('--role', default='cleaner')
        parser.add_argument('--other', default='mixer')
        parser.add_argument('--coordinator',
                            default='http://127.0.0.1:8765')
        parser.add_argument('--ask', action='store_true')
        parser.add_argument('--give', nargs=3, action='append', default=[],
                            metavar=('ITEM', 'QUANTITY', 'SLOT'))
        return parser

    def setup(self):
        self.args = getattr(self, 'args', None) or self.parse_args()
        inv = self.client.tabs.inventory
        inv.auto_locate = True
        i = inv.interface
        for colour, (key, _) in i.ALPHA_MAPPING.get(i.widget.name,
                                                    {}).items():
            i.create_template_group(key, colour, 28)
        self.client.mouse_options.add_colours('white', 'cyan')
        self.hop = HopToFriend(self, self.args.partner) \
            if self.args.hop else None
        self.trade = None
        self.ready_at = 0.0

    def update(self):
        self.client.gauges.update()
        self.client.game_screen.update()
        self.client.mouse_options.update()
        self.client.tabs.update()
        if self.trade is not None:
            self.trade.update()

    def end(self, message):
        self.client.logger.info(f'TRADE TEST: {message}')
        self.stop()

    def action(self):
        if self.hop is not None:
            result = self.hop.step()
            if result is None:
                self.msg.append(f'hop: {self.hop.state}')
                return
            if result is False:
                return self.end(f'FAILED to hop: {self.hop.failed}')
            self.hop = None
            self.ready_at = time.time() + 3
        a = self.args
        if not a.id:
            return self.end('hop done')
        if time.time() < self.ready_at:
            return
        inv = self.client.tabs.inventory
        if self.trade is None:
            if not inv.state.endswith('selected'):
                self._click_tab(inv)
                return
            if not inv.interface.located:
                return
            give = [Give(item, int(q), inv.interface.icons[f'item{slot}'])
                    for item, q, slot in a.give]
            signal = TradeSignal(Coordinator(a.coordinator, a.role), a.id,
                                 a.other)
            self.trade = PlayerTrade(self, a.partner, signal, a.ask, give)
            return
        result = self.trade.step()
        if result is None:
            self.msg.append(f'trade: {self.trade.state}')
            return
        self.end('traded' if result else f'FAILED: {self.trade.failed}')


def main():
    app = PlayerTradeTest()
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
