"""General GE trading tools, from the OSRS Wiki real-time prices API.

* :class:`WikiPrices`: item mapping (ids, buy limits) and 5m time series.
* :class:`PriceHistory`: a rolling day of prices from one bulk request now and
  then, instead of a series per item.
* :func:`reference`: a steady current price, the volume-weighted median of
  the last half hour (the single latest 5m point is often empty).
* :func:`evaluate` / :func:`select`: margin statistics for a conversion
  (buy one item, turn it into another, sell that), judged against
  :class:`Rules` the bot supplies.
* :class:`OfferBook`: offsets learned per item from how offers fill, and
  bid / ask prices that respect a margin floor.
* :func:`plan_quantities`: split a purchase over the best conversions.

Which items to trade, the thresholds and when to visit the GE are the
bot's policy and live with the bot; nothing here touches the game.
"""
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

API = 'https://prices.runescape.wiki/api/v1/osrs'
# generic UA: never the user's account or contact details
USER_AGENT = 'OSRSHerbArbitrageScanner/1.0 - Personal market analysis'

GE_TAX = 0.02
GE_TAX_CAP = 5_000_000
REFERENCE_INTERVALS = 12   # one hour of 5m intervals
WINDOWS = (30, 60, 120, 360, 1440)     # minutes
MIN_SHARE = 0.005


def tax(price: int) -> int:
    return min(int(price * GE_TAX), GE_TAX_CAP)


def margin(buy: int, sell: int, ratio: float = 1.0) -> float:
    """Profit per input item bought at ``buy``, converted (``ratio`` outputs
    per input) and sold at ``sell``, after GE tax."""
    return ratio * (sell - tax(sell)) - buy


# ---------------------------------------------------------------- prices

class WikiPrices:
    """Minimal client for the Wiki prices API (``requests`` imported only
    when used)."""

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

    def limit(self, name: str) -> Optional[int]:
        return self.item(name).get('limit')

    def series(self, name: str) -> List[dict]:
        """The last ~day of 5m intervals (oldest first)."""
        return self.get('timeseries', timestep='5m',
                        id=self.item(name)['id'])['data']


class PriceHistory:
    """A rolling day of prices for some items, kept on disk, from one bulk
    request (``5m``: every item) per ``interval`` seconds, restarts included.

    The single-item series are only fetched to begin with (or after hours
    without any samples), the mapping once a week. A sample stands for the
    time since the one before (up to ``interval``): its volumes are scaled
    up, so a day of them adds up to about a day's volume.

    Use :meth:`refresh`, then :meth:`series` / :meth:`limit` as with
    :class:`WikiPrices`, which then cost nothing."""

    KEEP_SECONDS = 30 * 3600
    SEED_SPAN = 6 * 3600
    """Less of the last day than this covered: fetch the series."""
    MAPPING_SECONDS = 7 * 24 * 3600
    STEP = 300

    def __init__(self, items: Sequence[str], path: str, interval: int = 900,
                 wiki: Optional[WikiPrices] = None, clock=time.time):
        self.items = list(items)
        self.path = Path(path)
        self.interval = interval
        self.wiki = wiki or WikiPrices()
        self.clock = clock
        self.data = {'mapped': 0, 'mapping': {}, 'tried': 0, 'fetched': 0,
                     'series': {}}
        try:
            self.data.update(json.loads(self.path.read_text()))
        except (OSError, ValueError):
            pass

    @property
    def fetched(self) -> float:
        """When the latest prices arrived (0: never)."""
        return self.data['fetched']

    def due(self) -> bool:
        return self.clock() - self.data['tried'] >= self.interval

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.data))
        tmp.replace(self.path)

    def refresh(self):
        """Fetch what's due (nothing, before ``interval`` is up; a failed
        request isn't repeated before that either)."""
        if not self.due():
            return
        now = self.clock()
        self.data['tried'] = now
        self.save()
        if now - self.data['mapped'] > self.MAPPING_SECONDS or any(
                i not in self.data['mapping'] for i in self.items):
            self.data['mapping'] = {
                i: {'id': e['id'], 'limit': e.get('limit')}
                for i in self.items for e in [self.wiki.item(i)]}
            self.data['mapped'] = now
            self.save()
        for item in self.items:
            if not self._covered(item, now):
                self.data['series'][item] = self.wiki.get(
                    'timeseries', timestep='5m',
                    id=self.data['mapping'][item]['id'])['data']
                self.save()
        bulk = self.wiki.get('5m')
        stamp, data = int(bulk['timestamp']), bulk['data']
        for item in self.items:
            self._add(item, stamp, data.get(
                str(self.data['mapping'][item]['id'])) or {}, now)
        self.data['fetched'] = now
        self.save()

    def _covered(self, item: str, now: float) -> bool:
        times = [p['timestamp'] for p in self.data['series'].get(item, [])
                 if now - p['timestamp'] < 24 * 3600]
        return bool(times) and max(times) - min(times) >= self.SEED_SPAN

    def _add(self, item: str, stamp: int, entry: dict, now: float):
        points = self.data['series'].setdefault(item, [])
        last = max((p['timestamp'] for p in points), default=0)
        if stamp > last:
            share = max(1, min(self.interval // self.STEP,
                               (stamp - last) // self.STEP))
            point = {'timestamp': stamp}
            for side in ('High', 'Low'):
                point[f'avg{side}Price'] = entry.get(f'avg{side}Price')
                volume = f'{side.lower()}PriceVolume'
                point[volume] = (entry.get(volume) or 0) * share
            points.append(point)
        self.data['series'][item] = [
            p for p in points if now - p['timestamp'] < self.KEEP_SECONDS]

    def series(self, name: str) -> List[dict]:
        return self.data['series'].get(name, [])

    def limit(self, name: str) -> Optional[int]:
        return (self.data['mapping'].get(name) or {}).get('limit')


def reference(series: Sequence[dict], key: str,
              intervals: int = REFERENCE_INTERVALS) -> Optional[int]:
    """The current ``avgLowPrice`` / ``avgHighPrice``: the median price an
    item recently changed hands at, every item counting once (a single one
    sold at a fancy price doesn't count as much as a thousand).

    Recent is the last half hour; longer if hardly anything was traded then
    (under ``MIN_SHARE`` of the day's volume). Without times and volumes:
    the median of the last ``intervals`` values."""
    points = [p for p in series if p.get(key)]
    if not points:
        return None
    vol = key.replace('avg', '').replace('High', 'high').replace(
        'Low', 'low') + 'Volume'
    if not all('timestamp' in p and vol in p for p in points):
        return int(statistics.median(
            [p[key] for p in points][-intervals:]))
    now = max(p['timestamp'] for p in series)
    total = sum(p[vol] or 0 for p in points
                if now - p['timestamp'] < 24 * 3600)
    for minutes in WINDOWS:
        recent = [(p[key], p[vol] or 0) for p in points
                  if now - p['timestamp'] < minutes * 60]
        traded = sum(v for _, v in recent)
        if traded and traded >= MIN_SHARE * total:
            return weighted_median(recent)
    return int(points[-1][key])


def weighted_median(values: Sequence[Tuple[int, int]]) -> int:
    """Of (value, weight)s: the lowest value with half the weight at or
    below it."""
    half, seen = sum(w for _, w in values) / 2, 0
    for value, weight in sorted(values):
        seen += weight
        if seen >= half:
            return int(value)
    raise ValueError('no values')


def volume(series: Sequence[dict]) -> int:
    return sum((p.get('highPriceVolume') or 0) + (p.get('lowPriceVolume') or 0)
               for p in series)


@dataclass
class Conversion:
    source: str            # bought
    product: str           # sold
    ratio: float = 1.0     # products per source item


@dataclass
class Rules:
    """What makes a conversion worth trading; set by the bot."""
    offset: int             # gp above the low to bid / below the high to ask
    min_margin: float       # gp per source item at our prices
    min_margin_share: float  # margin / source price
    min_positive: float     # share of 5m intervals with a positive margin
    max_noise: float        # margin sd / median margin
    min_volume: int         # daily volume of source and of product


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
             product: Sequence[dict], rules: Rules) -> Evaluation:
    """Margin statistics at the base prices (low + offset / high - offset)
    over every 5m interval where both sides traded, and the reasons (if
    any) the conversion fails ``rules``."""
    o = rules.offset
    highs = {p['timestamp']: p['avgHighPrice'] for p in product
             if p.get('avgHighPrice')}
    margins = [margin(p['avgLowPrice'] + o, highs[p['timestamp']] - o,
                      conv.ratio)
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
    if med < rules.min_margin:
        r.append(f'margin {med:.0f} < {rules.min_margin}')
    if med < rules.min_margin_share * buy_ref:
        r.append(f'margin {med / buy_ref:.1%} of price < '
                 f'{rules.min_margin_share:.0%}')
    if positive < rules.min_positive:
        r.append(f'positive {positive:.0%} < {rules.min_positive:.0%}')
    if noise is None or noise > rules.max_noise:
        r.append(f'noise {noise if noise is None else round(noise, 2)} > '
                 f'{rules.max_noise}')
    low_volume = min(ev.source_volume, ev.product_volume)
    if low_volume < rules.min_volume:
        r.append(f'volume {low_volume} < {rules.min_volume}')
    return ev


def select(evaluations: Sequence[Evaluation], limits: Dict[str, int]
           ) -> List[Evaluation]:
    """The conversions that passed, best first by margin x buy limit (what
    one limit window can earn)."""
    good = [e for e in evaluations if e.ok]
    return sorted(good, key=lambda e: -e.median_margin *
                  (limits.get(e.conversion.source) or 0))


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


# ---------------------------------------------------------------- offers

@dataclass
class Fill:
    """What the GE showed for one offer when it was checked."""
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
    """Per-item extra offsets (gp beyond ``Rules.offset``), learned from how
    offers fill:

    * slow (under ``slow_fill`` filled after ``slow_hours``): one step more
      aggressive (bid up / ask down)
    * complete: half a step back
    * anything else: unchanged

    A step is ``step_fraction`` of the price, at least ``min_step`` gp; the
    extra stays within ``max_steps`` steps, and backing off never goes past
    the plain offset (``extra >= -offset``). Keep the file per account and
    out of the repo (``*.local.json``)."""

    extra: Dict[str, int] = field(default_factory=dict)  # 'buy:<item>'
    path: Optional[str] = None
    slow_fill: float = 0.5
    slow_hours: float = 2.0
    step_fraction: float = 0.01
    min_step: int = 5
    max_steps: int = 6

    @classmethod
    def load(cls, path: str, **settings) -> 'OfferBook':
        p = Path(path)
        data = json.loads(p.read_text()) if p.exists() else {}
        return cls(extra=data.get('extra', {}), path=path, **settings)

    def save(self):
        if self.path:
            p = Path(self.path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(
                {'extra': self.extra, 'saved': int(time.time())}, indent=1))

    def step(self, price: int) -> int:
        return max(self.min_step, int(round(price * self.step_fraction)))

    def get(self, side: str, item: str) -> int:
        return self.extra.get(f'{side}:{item}', 0)

    def learn(self, fill: Fill, reference_price: int, offset: int) -> int:
        """Adjust the item's extra from an offer's fill; returns it."""
        key = f'{fill.side}:{fill.item}'
        extra = self.extra.get(key, 0)
        step = self.step(reference_price)
        if fill.fraction >= 1.0:
            extra -= step // 2
        elif fill.fraction < self.slow_fill and fill.hours >= self.slow_hours:
            extra += step
        extra = max(-offset, min(extra, self.max_steps * step))
        self.extra[key] = extra
        return extra

    def prices(self, ev: Evaluation, rules: Rules) -> Tuple[Offer, Offer]:
        """Bid for the source and ask for the product. The bid is held back
        so the margin at our own prices stays >= ``rules.min_margin``; if
        even the base bid can't, the source isn't bought (the market has
        failed for now). The product is always offered."""
        conv = ev.conversion
        ask = ev.sell_ref - rules.offset - self.get('sell', conv.product)
        sell = Offer(conv.product, 'sell', ask)
        base = ev.buy_ref + rules.offset
        ceiling = int(conv.ratio * (ask - tax(ask)) - rules.min_margin)
        if ceiling < base:
            return Offer(conv.source, 'buy', None,
                         f'margin below {rules.min_margin} even at the '
                         f'base bid'), sell
        bid, why = base + self.get('buy', conv.source), ''
        if bid > ceiling:
            bid, why = ceiling, 'capped at the margin floor'
        return Offer(conv.source, 'buy', bid, why), sell
