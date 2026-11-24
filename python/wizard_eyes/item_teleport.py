"""Teleporting with an item in the inventory, e.g. the Strength cape:
find it by its mouse-over text, use its option, check the landing.

    tele = ItemTeleport(app, method)
    ...  # every frame:
    tele.update()
    result = tele.step()   # None: working on it, True: landed, False: failed

``method`` comes from ``config/teleports.json``. Needs
``mouse_options.add_colours('white', 'orange')`` and the inventory tab set
up (``inventory.auto_locate`` and a template group for its slots).

Expectations: the inventory is open and the item found within one pass over
its slots; the position jumps within ``LAND_SECONDS`` of using the option;
the landing is near one of the method's landing points. Anything else fails
(never retried).
"""
import time
from typing import List, Optional

from .interactions import UseOption
from .teleports import Method, name_matches

TAB_SECONDS = 6.0
HOVER_FRAMES = 4
"""Frames the mouse rests on a slot for its mouse-over text to show."""
LAND_SECONDS = 10.0
SETTLE_SECONDS = 1.2
JUMP_TILES = 20


class ItemTeleport:

    def __init__(self, app, method: Method):
        if not method.item or 'inventory' not in method.where:
            raise ValueError(f'{method.key}: not an inventory item')
        self.app = app
        self.client = app.client
        self.method = method
        self.state = 'tab'
        self.since = time.time()
        self.slots: Optional[List] = None
        self.hover_frames = 0
        self.use: Optional[UseOption] = None
        self.start: Optional[tuple] = None
        self.left_at = 0.0
        self.failed: Optional[str] = None

    @property
    def log(self):
        return self.client.logger

    def _position(self):
        x, y, z = self.client.gauges.minimap.gps.get_coordinates()
        return None if x == -1 else (int(x), int(y), int(z))

    def _fail(self, message: str) -> bool:
        self.failed = f'{self.method.key}: {message}'
        self.log.warning(f'Teleport {self.failed}')
        return False

    def update(self):
        self.client.tabs.update()
        self.client.tabs.inventory.interface.update(True)

    def step(self) -> Optional[bool]:
        if self.failed is not None:
            return False
        return getattr(self, f'_{self.state}')(time.time())

    def _tab(self, now):
        inv = self.client.tabs.inventory
        if (inv.state or '').endswith('selected') and inv.interface.located:
            self.slots = [i for i in inv.interface.icons.values()
                          if i.state not in (None, 'nothing')]
            self.state = 'find'
            return None
        if now - self.since > TAB_SECONDS:
            return self._fail('the inventory did not open')
        if inv.located and not inv.clicked and \
                not (inv.state or '').endswith('selected'):
            self.app._click_tab(inv)
        return None

    def _find(self, now):
        if not self.slots:
            return self._fail(f'no {self.method.item[0]} in the inventory')
        icon = self.slots[0]
        if self.hover_frames == 0:
            self.client.screen.mouse_to_object(icon)
        elif name_matches(self.client.mouse_options.state or '',
                          self.method.item):
            self.use = UseOption(self.app, icon, self.method.option,
                                 self.method.item)
            self.start = self._position()
            self.state = 'using'
            return None
        self.hover_frames += 1
        if self.hover_frames > HOVER_FRAMES:
            self.slots.pop(0)
            self.hover_frames = 0
        return None

    def _using(self, now):
        result = self.use.step()
        if result is False:
            return self._fail(f'could not use {self.method.option!r}')
        if result:
            self.state, self.since = 'leaving', now
        return None

    def _leaving(self, now):
        pos, start = self._position(), self.start
        if pos is not None and start is not None and not self.left_at and (
                pos[2] != start[2] or max(abs(pos[0] - start[0]),
                                          abs(pos[1] - start[1]))
                > JUMP_TILES):
            self.left_at = now
        if self.left_at:
            if now - self.left_at < SETTLE_SECONDS:
                return None
            if pos is None or self.method.landing_point(pos) is None:
                return self._fail(f'landed at {pos}, not near '
                                  f'{self.method.lands.get("tiles")}')
            self.log.info(f'Teleport {self.method.key}: landed at {pos}')
            return True
        if now - self.since > LAND_SECONDS:
            return self._fail(f'no teleport within {LAND_SECONDS:.0f}s '
                              f'(at {pos})')
        return None
