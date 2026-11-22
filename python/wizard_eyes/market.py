"""Pricing and scheduling for GE arbitrage on a conversion (e.g. grimy herb
-> clean herb), from the OSRS Wiki real-time prices API.

Three parts:

* **Reference prices** (:func:`reference`): the median of the last hour of
  5-minute average lows / highs. Steadier than the single latest 5m point
  (which is often empty for thin items) but still current.
* **Selection** (:func:`evaluate`, :func:`select`): margin statistics over
  the last day of 5m intervals; a conversion is traded only if its margin is
  a large enough share of the price, reliably positive, not too noisy and
  liquid enough. This reproduces the hand-made avoid list (ranarr,
  snapdragon, huasca...) without hard-coding it.
* **Adaptation** (:class:`OfferBook`): the base rule is bid = low + 5,
  ask = high - 5. When the collection box is checked, each offer's fill
  (fraction and age) moves that item's extra offset: slow buys bid higher,
  slow sells ask lower, instant fills back off. Prices never cross the
  margin floor; if the market can't pay the floor, the item isn't bought.
* **Trips** (:func:`next_trip_seconds`): GE trips are 4-12 hours apart,
  drawn at random each time; purchases are sized for that gap.

Nothing here touches the game; the GE automation reads fills and places the
resulting offers.
"""
import json
import random
import statistics
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

API = 'https://prices.runescape.wiki/api/v1/osrs'
# generic UA: never the user's account or contact details
USER_AGENT = 'OSRSHerbArbitrageScanner/1.0 - Personal market analysis'

GE_TAX = 0.02
GE_TAX_CAP = 5_000_000
BASE_OFFSET = 5            # gp above the low / below the high
REFERENCE_INTERVALS = 12   # one hour of 5m intervals

# adaptation
SLOW_FILL = 0.5            # less than this filled ...
SLOW_HOURS = 2.0           # ... after this long counts as slow
STEP_FRACTION = 0.01       # one step is 1% of the price ...
MIN_STEP = 5               # ... but at least this many gp
MAX_STEPS = 6              # never more than this many steps of chasing
MIN_EXTRA = -BASE_OFFSET   # backing off never goes past the plain low/high

# selection
MIN_MARGIN = 50            # gp per conversion, after tax, at our prices
MIN_MARGIN_SHARE = 0.05    # margin / input price
MIN_POSITIVE = 0.95        # share of 5m intervals with a positive margin
MAX_NOISE = 0.5            # margin sd / median margin
MIN_VOLUME = 50_000        # daily volume of input and output

MIN_TRIP_HOURS = 4.0
MAX_TRIP_HOURS = 12.0


def tax(price: int) -> int:
    return min(int(price * GE_TAX), GE_TAX_CAP)


def margin(buy: int, sell: int, ratio: float = 1.0) -> float:
    """Profit per input item bought at ``buy``, converted (``ratio`` outputs
    per input) and sold at ``sell``."""
    return ratio * (sell - tax(sell)) - buy


def next_trip_seconds(rng: Optional[random.Random] = None,
                      low: float = MIN_TRIP_HOURS,
                      high: float = MAX_TRIP_HOURS) -> float:
    """Seconds until the next GE trip: uniform between 4 and 12 hours."""
    rng = rng or random
    return rng.uniform(low, high) * 3600


# ---------------------------------------------------------------- prices

class WikiPrices:
    """Minimal client for the Wiki prices API (``requests`` only when
    used)."""

    def __init__(self, session=None):
        self.session = session
        self._mapping = None

    def get(self, endpoint: str, **params):
        if self.session is None:
            import requests
            self.session = requests.Session()
            self.session.headers['User-Agent'] = USER_AGENT
        r = self.session.get(f'{API}/{endpoint}', params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def mapping(self) -> Dict[str, dict]:
        """Item name (lower case) -> mapping entry (id, limit, ...)."""
        if self._mapping is None:
            self._mapping = {m['name'].lower(): m for m in self.get('mapping')}
        return self._mapping

    def item(self, name: str) -> dict:
        return self.mapping()[name.lower()]

    def series(self, name: str) -> List[dict]:
        """The last ~day of 5m intervals (oldest first)."""
        return self.get('timeseries', timestep='5m',
                        id=self.item(name)['id'])['data']


def reference(series: Sequence[dict], key: str,
              intervals: int = REFERENCE_INTERVALS) -> Optional[int]:
    """Median of the last ``intervals`` non-empty values of ``key``
    (``avgLowPrice`` / ``avgHighPrice``)."""
    values = [p[key] for p in series if p.get(key)][-intervals:]
    return int(statistics.median(values)) if values else None


def volume(series: Sequence[dict]) -> int:
    return sum((p.get('highPriceVolume') or 0) + (p.get('lowPriceVolume') or 0)
               for p in series)


@dataclass
class Conversion:
    source: str            # bought, e.g. 'Grimy toadflax'
    product: str           # sold, e.g. 'Toadflax'
    ratio: float = 1.0     # products per source item


@dataclass
class Evaluation:
    conversion: Conversion
    buy_ref: Optional[int]
    sell_ref: Optional[int]
    median_margin: Optional[float]
    positive: float
    noise: Optional[float]   # sd / median margin
    source_volume: int
    product_volume: int
    reasons: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.reasons


def evaluate(conv: Conversion, source: Sequence[dict],
             product: Sequence[dict]) -> Evaluation:
    """Margin statistics at our base prices (low + 5 / high - 5) over every
    5m interval where both sides traded, plus the selection verdict."""
    highs = {p['timestamp']: p['avgHighPrice'] for p in product
             if p.get('avgHighPrice')}
    margins = [margin(p['avgLowPrice'] + BASE_OFFSET,
                      highs[p['timestamp']] - BASE_OFFSET, conv.ratio)
               for p in source
               if p.get('avgLowPrice') and p['timestamp'] in highs]
    buy_ref = reference(source, 'avgLowPrice')
    sell_ref = reference(product, 'avgHighPrice')
    med = statistics.median(margins) if margins else None
    positive = sum(m > 0 for m in margins) / len(margins) if margins else 0.0
    noise = (statistics.pstdev(margins) / med
             if med and med > 0 and len(margins) > 1 else None)
    ev = Evaluation(conv, buy_ref, sell_ref, med, positive, noise,
                    volume(source), volume(product))
    r = ev.reasons
    if med is None or buy_ref is None or sell_ref is None:
        r.append('no prices')
        return ev
    if med < MIN_MARGIN:
        r.append(f'margin {med:.0f} < {MIN_MARGIN}')
    if med < MIN_MARGIN_SHARE * buy_ref:
        r.append(f'margin {med / buy_ref:.1%} of price < '
                 f'{MIN_MARGIN_SHARE:.0%}')
    if positive < MIN_POSITIVE:
        r.append(f'positive {positive:.0%} < {MIN_POSITIVE:.0%}')
    if noise is None or noise > MAX_NOISE:
        r.append(f'noise {noise if noise is None else round(noise, 2)} > '
                 f'{MAX_NOISE}')
    if min(ev.source_volume, ev.product_volume) < MIN_VOLUME:
        r.append(f'volume {min(ev.source_volume, ev.product_volume)} < '
                 f'{MIN_VOLUME}')
    return ev


def select(evaluations: Sequence[Evaluation], limits: Dict[str, int]
           ) -> List[Evaluation]:
    """The tradeable conversions, best first by margin x buy limit (what
    one 4h limit window can earn)."""
    good = [e for e in evaluations if e.ok]
    return sorted(good, key=lambda e: -e.median_margin *
                  (limits.get(e.conversion.source) or 0))


# ---------------------------------------------------------------- offers

def step_size(price: int) -> int:
    return max(MIN_STEP, int(round(price * STEP_FRACTION)))


@dataclass
class Fill:
    """What the collection box showed for one offer."""
    item: str
    side: str          # 'buy' or 'sell'
    price: int
    quantity: int
    filled: int
    hours: float       # since the offer was placed

    @property
    def fraction(self) -> float:
        return self.filled / self.quantity if self.quantity else 1.0


@dataclass
class Offer:
    item: str
    side: str
    price: Optional[int]   # None: don't trade it now
    why: str = ''


@dataclass
class OfferBook:
    """Per-item extra offsets (gp beyond the base 5), learned from how
    offers fill. Saved per account (``*.local.json``: never published)."""

    extra: Dict[str, int] = field(default_factory=dict)  # 'buy:Grimy x'
    path: Optional[str] = None

    @classmethod
    def load(cls, path: str) -> 'OfferBook':
        p = Path(path)
        data = json.loads(p.read_text()) if p.exists() else {}
        return cls(extra=data.get('extra', {}), path=path)

    def save(self):
        if self.path:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.path).write_text(json.dumps(
                {'extra': self.extra, 'saved': int(time.time())}, indent=1))

    def get(self, side: str, item: str) -> int:
        return self.extra.get(f'{side}:{item}', 0)

    def learn(self, fill: Fill, reference_price: int) -> int:
        """Adjust the item's extra offset from an offer's fill; returns the
        new extra. Slow (under half filled after 2h): one step more
        aggressive. Complete: half a step back. Anything else: unchanged.
        """
        key = f'{fill.side}:{fill.item}'
        extra = self.extra.get(key, 0)
        step = step_size(reference_price)
        if fill.fraction >= 1.0:
            extra -= step // 2
        elif fill.fraction < SLOW_FILL and fill.hours >= SLOW_HOURS:
            extra += step
        extra = max(MIN_EXTRA, min(extra, MAX_STEPS * step))
        self.extra[key] = extra
        return extra

    def prices(self, ev: Evaluation) -> (Offer, Offer):
        """Bid for the source and ask for the product. The bid is held back
        so the margin at our own prices stays >= ``MIN_MARGIN``; when even
        the base bid can't, the source isn't bought (the market has failed
        for now). The product is always offered: stock is sold rather than
        held, at no less than the plain reference high minus the learned
        extra."""
        conv = ev.conversion
        ask = ev.sell_ref - BASE_OFFSET - self.get('sell', conv.product)
        sell = Offer(conv.product, 'sell', ask)
        bid = ev.buy_ref + BASE_OFFSET + self.get('buy', conv.source)
        ceiling = int(conv.ratio * (ask - tax(ask)) - MIN_MARGIN)
        if ceiling < ev.buy_ref + BASE_OFFSET:
            return Offer(conv.source, 'buy', None,
                         f'margin below {MIN_MARGIN} even at the base bid'
                         ), sell
        why = ''
        if bid > ceiling:
            bid, why = ceiling, 'capped at the margin floor'
        return Offer(conv.source, 'buy', bid, why), sell


def plan_quantities(ranked: Sequence[Evaluation], limits: Dict[str, int],
                    needed: int, slots: int) -> Dict[str, int]:
    """Split ``needed`` source items over the best conversions, one GE slot
    each, at most each item's buy limit."""
    out = {}
    for ev in ranked[:slots]:
        if needed <= 0:
            break
        n = min(needed, limits.get(ev.conversion.source) or 0)
        if n > 0:
            out[ev.conversion.source] = n
            needed -= n
    return out


def supplies_needed(rate_per_hour: float, hours: float, stock: int) -> int:
    """Source items to buy so that ``stock`` lasts ``hours`` at
    ``rate_per_hour``."""
    return max(0, int(rate_per_hour * hours) - stock)


def to_json(obj) -> str:
    return json.dumps(asdict(obj), default=str)
