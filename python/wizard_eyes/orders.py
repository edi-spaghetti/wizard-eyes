"""Grand Exchange orders from a planner, and what a bot does with them.

A planner posts full targets (one :class:`Order` per item and side); the bot
compares them with what its GE slots show and works out the difference with
:func:`reconcile`. The bot also checks each order for itself
(:func:`check_order`) and ignores orders that are too old (:func:`is_stale`).

Nothing here touches the game or knows which items are traded.
"""
from dataclasses import asdict, dataclass, field
from typing import Iterable, List, Optional

BUY, SELL = 'buy', 'sell'
PRICE_TOLERANCE = 0.02
"""An offer is only re-listed if its price is off by more than this."""
MAX_DEVIATION = 0.10
"""Orders further than this outside the market's low-high range are
refused."""


@dataclass
class Order:
    item: str
    side: str
    price: int
    quantity: int
    why: str = ''

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'Order':
        return cls(d['item'], d['side'], int(d['price']), int(d['quantity']),
                   d.get('why', ''))


@dataclass
class Slot:
    """An offer as the GE shows it."""
    item: str
    side: str
    price: int
    quantity: int
    filled: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.quantity - self.filled)

    @property
    def complete(self) -> bool:
        return self.filled >= self.quantity

    @classmethod
    def from_dict(cls, d: dict) -> 'Slot':
        return cls(d['item'], d['side'], int(d['price']), int(d['quantity']),
                   int(d.get('filled', 0)))


@dataclass
class Difference:
    collect: List[Slot] = field(default_factory=list)   # complete
    abort: List[Slot] = field(default_factory=list)     # then collect
    keep: List[Slot] = field(default_factory=list)
    place: List[Order] = field(default_factory=list)
    no_slot: List[Order] = field(default_factory=list)


def price_off(price: int, target: int) -> float:
    """How far ``price`` is from ``target``, as a share of the target."""
    return abs(price - target) / target if target else float('inf')


def reconcile(slots: Iterable[Slot], orders: Iterable[Order],
              priced: Iterable[str], total_slots: int = 8,
              tolerance: float = PRICE_TOLERANCE) -> Difference:
    """What to change so ``slots`` match ``orders``.

    Complete offers are collected. An open offer is kept if an order for the
    same item and side has a price within ``tolerance``, aborted if its
    price is further off or the planner prices the item but doesn't order
    that side, and left alone if the planner doesn't price it (``priced``).
    Quantities never cause a re-list. Orders not already standing are placed
    in the order given while slots last."""
    orders = list(orders)
    wanted = {(o.item, o.side): o for o in orders}
    if len(wanted) != len(orders):
        raise ValueError('two orders for the same item and side')
    priced = set(priced)
    diff = Difference()
    standing = set()
    for s in slots:
        order = wanted.get((s.item, s.side))
        if s.complete:
            diff.collect.append(s)
        elif order is None:
            (diff.abort if s.item in priced else diff.keep).append(s)
        elif price_off(s.price, order.price) > tolerance:
            diff.abort.append(s)
        else:
            diff.keep.append(s)
            standing.add((s.item, s.side))
    free = total_slots - len(diff.keep)
    for o in orders:
        if (o.item, o.side) in standing:
            continue
        if free > 0:
            diff.place.append(o)
            free -= 1
        else:
            diff.no_slot.append(o)
    return diff


def check_order(order: Order, low: Optional[int], high: Optional[int],
                max_quantity: Optional[int] = None,
                max_deviation: float = MAX_DEVIATION) -> Optional[str]:
    """Why a bot must refuse ``order`` (None: it's fine). ``low`` / ``high``
    are the market's prices, read by the bot itself."""
    if order.side not in (BUY, SELL):
        return f'unknown side {order.side!r}'
    if order.price < 1 or order.quantity < 1:
        return 'price and quantity must be at least 1'
    if max_quantity is not None and order.quantity > max_quantity:
        return f'quantity {order.quantity} > limit {max_quantity}'
    if not low or not high:
        return 'no market price to compare with'
    low, high = min(low, high), max(low, high)
    if not (low * (1 - max_deviation) <= order.price
            <= high * (1 + max_deviation)):
        return (f'price {order.price} more than {max_deviation:.0%} outside '
                f'the market ({low}-{high})')
    return None


def is_stale(issued: float, now: float, max_age: float) -> bool:
    """Orders too old (or from the future: clocks disagree) to follow."""
    return not (-60 <= now - issued <= max_age)
