"""Walking to a tile with the cache-built collision map.

    walker = Walker(app, [tile, ...])
    ...  # every frame, after updating gauges, game_screen, mouse_options
    walker.update()
    result = walker.step()   # None: walking, True: there, False: gave up

By default it stops on exactly that tile. To use something in the game that
isn't needed: pass ``done`` (e.g. :func:`in_view` of its entity), and walking
ends, still moving, as soon as that's true; clicking the entity makes the
game walk the rest. ``spread`` varies which tile is clicked at the end.

Plans a path with ``CollisionMap.find_path`` and follows it by clicking the
furthest path tile that's safely inside the minimap, re-clicking shortly
before reaching it. At a closed door it stops on the tile before, hovers the
door's game entity on the game screen and clicks only if the mouse text says
"Open"; it walks on only once the door shows "Close" (open).

Every click comes with an expectation of what happens next (see
:class:`Expectation`): the player reaches the clicked tile in time, without
straying from a shortest route there. When that fails the walker stops,
re-checks the situation (e.g. hovers the door again) before doing anything
else, and gives up after a couple of failures instead of retrying blindly.

Needs the camera north / top-down (start-up checks do this), the grid info
overlay for coordinates and ``mouse_options.add_colours('white', 'cyan')``.
"""
import math
import random
import re
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from ..game_entities.entity import GameEntity
from .collision_map import DOOR_COST, CollisionMap

Tile = Tuple[int, int, int]
DoorKey = Tuple[Tile, Tile]


def chebyshev(a: Tile, b: Tile) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


VIEW_CORNER = 0.3
"""The corners of the client (this part of its width and height) don't count
as "in view": the chat box, side panel and minimap are there, and not all of
them are always located. The middle of each edge is fine."""


def in_view(client, entity: GameEntity) -> bool:
    """Whether an entity is comfortably on screen, clear of the interface,
    so it can be hovered and clicked."""
    try:
        x1, y1, x2, y2 = entity.get_bbox()
    except TypeError:  # no bounding box: off screen
        return False
    cx1, cy1, cx2, cy2 = client.get_bbox()
    mx, my = (cx2 - cx1) * VIEW_CORNER, (cy2 - cy1) * VIEW_CORNER
    if (x1 < cx1 + mx or x2 > cx2 - mx) and \
            (y1 < cy1 + my or y2 > cy2 - my):
        return False
    return bool(client.game_screen.is_clickable(x1, y1, x2, y2))


class InView:
    """:func:`in_view` of an entity as a ``done`` for :class:`Walker`."""

    def __init__(self, client, entity: GameEntity):
        self.client = client
        self.entity = entity

    def __call__(self) -> bool:
        return in_view(self.client, self.entity)

    def reason(self) -> str:
        try:
            bbox = tuple(int(v) for v in self.entity.get_bbox())
        except TypeError:
            return ': no bounding box'
        return (f': {bbox} in client {tuple(self.client.get_bbox())}, '
                f'clickable {self.client.game_screen.is_clickable(*bbox)}')


@dataclass
class Expectation:
    """What a click should lead to."""
    what: str
    start: Tile
    goal: Tile
    cost: int
    """Planned route cost start -> goal (steps, plus DOOR_COST per door)."""
    deadline: float
    final: bool
    """True if the goal is where we stop (target or door); otherwise the
    next click is made while still walking."""
    door: Optional[DoorKey] = None
    """Door the route goes through, if any: failures are put down to it."""
    reached: bool = False
    """Passed over the goal while moving (must stop on it to count)."""


class Walker:

    RECLICK_DISTANCE = 4
    """Click the next waypoint once this close (tiles) to the current one."""

    MOVING_SECONDS = 1.5
    """Counted as stopped once the position hasn't changed for this long."""

    SECONDS_PER_TILE = 0.7
    """Walking pace allowance (walking is 0.6s per tile, running 0.3s)."""

    MAX_DETOUR = 2
    """A position further than this (route cost) from any shortest route
    to the clicked tile means we're not going where we expected."""

    MAX_FAILURES = 2
    """Failed expectations without getting closer before giving up."""

    MAX_OVERSHOOTS = 2
    """Clicks that carry us past the clicked tile before giving up (the
    minimap scale is wrong, e.g. it's been zoomed with the mouse wheel)."""

    OBSTACLE_SECONDS = 3.0
    """Time allowed to get past a door once we start walking through it;
    after that we stop and check the door again (never retry blindly)."""

    MAX_NO_PROGRESS = 3
    """Expectations (met or not) in a row that don't bring us closer to
    the target before giving up (e.g. walking back and forth)."""

    MAX_DOOR_FAILURES = 2
    """Failures at one door (didn't open, or couldn't get through) before
    giving up."""

    DOOR_HOVERS = 6
    """Hovers over a door's tiles without finding Open / Close before
    trying (once) to walk through."""

    DOOR_OPEN_SECONDS = 2.0
    """Wait after clicking "Open" before checking the door again (the
    player may have to step next to it, then it takes a tick to swing)."""

    MINIMAP_MARGIN = 8
    """Pixels kept clear of the minimap's safe radius."""

    DONE_STEPS = 15
    """With ``done``: the route left can't be longer than this (nor go
    through a closed door): something on screen can be a long way round."""

    MAX_BLIND_SECONDS = 20.0
    """Time without a position (e.g. the overlay is hidden) before giving
    up."""

    def __init__(self, app, goals: Iterable[Tile],
                 cm: Optional[CollisionMap] = None, doors: bool = True,
                 done: Optional[Callable[[], bool]] = None, spread: int = 0):
        """
        :param goals: where to go: whichever is closest to walk to.
        :param done: close enough once this is true (the exact tile is then
            only the fallback).
        :param spread: click up to this many tiles before the last one.
        """
        self.done = done
        self.spread = spread
        self.app = app
        self.client = app.client
        self.goals = list(goals)
        self.doors = doors
        self.cm = cm or CollisionMap()
        self.target: Optional[Tile] = None
        self.path: List[Tile] = []
        self.clicks = 0
        self.clicked_at = 0.0
        self.last_pos: Optional[Tile] = None
        self.last_moved = 0.0
        self.started = time.time()
        self.blind_since: Optional[float] = None
        self.arrived = False
        self._not_done_at: Optional[Tile] = None
        self.settling = False
        """Told to stop after a failure, not stopped yet."""
        self.failed: Optional[str] = None
        """Why we gave up."""
        # expectations
        self.expect: Optional[Expectation] = None
        self.failures = 0
        self.overshoots = 0
        self.best_remaining: Optional[int] = None
        self.no_progress = 0
        self.detour_cache: Dict[Tile, int] = {}
        self.obstacle: Optional[DoorKey] = None
        """Door we're currently trying to get through."""
        self.obstacle_since = 0.0
        self.crossed_cache: Dict[Tile, bool] = {}
        # doors
        self.door: Optional[GameEntity] = None
        self.door_key: Optional[DoorKey] = None
        self.door_hovers = 0
        self.door_clicked_at = 0.0
        self.door_open: Optional[DoorKey] = None
        """Door seen open (mouse text "Close"): we may walk through it."""
        self.door_blind: Optional[DoorKey] = None
        """Door we couldn't find on screen and are trying to walk through
        anyway (once)."""
        self.door_failures: Dict[DoorKey, int] = {}

    @property
    def log(self):
        return self.client.logger

    # helpers

    def give_up(self, reason: str):
        """Stop walking for good; :meth:`step` returns False from now on."""
        self.log.warning(f'Walk: giving up: {reason} '
                         f'(after {time.time() - self.started:.0f}s, '
                         f'{self.clicks} clicks)')
        self.failed = reason

    def position(self) -> Optional[Tile]:
        x, y, z = self.client.gauges.minimap.gps.get_coordinates()
        if x == -1:
            return None
        return int(x), int(y), int(z)

    def route_cost(self, a: Tile, b: Tile) -> Optional[int]:
        path = self.cm.find_path(a, b, doors=self.doors,
                                 max_nodes=20_000)
        if path is None:
            return None
        return len(path) - 1 + DOOR_COST * len(self.cm.doors_on_path(path))

    def _done_logged(self, pos: Tile) -> bool:
        done = self.done()
        if not done and chebyshev(pos, self.target) <= 6 and \
                pos != self._not_done_at:
            self._not_done_at = pos
            self.log.info(f'Walk: {chebyshev(pos, self.target)} tiles from '
                          f'{self.target} and not in view yet'
                          f'{getattr(self.done, "reason", lambda: "")()}')
        return done

    def close_enough(self, pos: Tile) -> bool:
        if self.done is None or self.target is None or \
                chebyshev(pos, self.target) > self.DONE_STEPS or \
                not self._done_logged(pos):
            return False
        path = self.cm.find_path(pos, self.target, doors=False,
                                 max_nodes=5_000)
        return path is not None and len(path) - 1 <= self.DONE_STEPS

    def plan(self, pos: Tile) -> bool:
        t = time.perf_counter()
        # several goals: the closest to walk to, then stick to it
        path = self.cm.find_path(pos, self.target or self.goals,
                                 doors=self.doors)
        ms = (time.perf_counter() - t) * 1000
        if path is None:
            self.log.warning(f'Walk: no path {pos} -> '
                             f'{self.target or self.goals}')
            return False
        self.target = path[-1]
        doors = [path[i] for i in self.cm.doors_on_path(path)]
        self.log.info(f'Walk: path {pos} -> {self.target}: '
                      f'{len(path) - 1} steps, doors at {doors}, '
                      f'planned in {ms:.0f}ms')
        self.path = path
        if self.best_remaining is None:
            # so a first click that gets us nowhere isn't "progress"
            self.best_remaining = len(path) - 1 + DOOR_COST * len(doors)
        return True

    def max_click_tiles(self) -> float:
        mm = self.client.gauges.minimap
        return (mm.safe_radius - self.MINIMAP_MARGIN) / mm.tile_size

    def minimap_xy(self, pos: Tile, tile: Tile) -> Tuple[int, int]:
        """Screen point of a tile's centre on the minimap."""
        mm = self.client.gauges.minimap
        x1, y1, _, _ = mm.get_bbox()
        ox, oy = mm.orb_xy
        dx, dy = tile[0] - pos[0], tile[1] - pos[1]
        return (x1 + ox + dx * mm.tile_size,
                y1 + oy - dy * mm.tile_size)

    def click(self, x: int, y: int):
        self.client.screen.click_aoi(x - 1, y - 1, x + 1, y + 1, speed=0.5)
        self.clicked_at = time.time()

    def stop_walking(self, pos: Tile):
        """Click our own tile on the minimap, so the game stops walking us
        somewhere we didn't expect."""
        self.click(*self.minimap_xy(pos, pos))
        self.log.info(f'Walk: stopping at {pos}')

    # expectations

    def walk_to(self, pos: Tile, tile: Tile, cost: int, final: bool,
                door: Optional[DoorKey] = None):
        """Click a tile on the minimap and record what should happen."""
        x, y = self.minimap_xy(pos, tile)
        what = f'walk {pos} -> {tile}' + (f' through door {door}'
                                           if door else '')
        self.log.info(f'Walk: expect {what} (cost {cost})')
        self.click(x, y)
        self.clicks += 1
        self.expect = Expectation(
            what, pos, tile, cost,
            deadline=time.time() + 2 + cost * self.SECONDS_PER_TILE,
            final=final, door=door)
        self.detour_cache = {}
        if door is not None and self.obstacle != door:
            self.obstacle = door
            self.obstacle_since = time.time()
            self.crossed_cache = {}

    def crossed(self, pos: Tile, door: DoorKey) -> bool:
        """True once we're on the far side of a door: the way from here to
        the near side goes through it (or there's none without it)."""
        if pos not in self.crossed_cache:
            near = door[0]
            path = self.cm.find_path(pos, near, doors=False,
                                     max_nodes=5_000)
            self.crossed_cache[pos] = path is None
        return self.crossed_cache[pos]

    def note_progress(self, pos: Tile) -> bool:
        """After each expectation (met or failed): are we closer to the
        target than ever before? Gives up (returns False) after a few
        rounds without getting closer, e.g. walking back and forth."""
        remaining = self.route_cost(pos, self.target)
        if remaining is not None and (self.best_remaining is None or
                                      remaining < self.best_remaining):
            self.best_remaining = remaining
            self.no_progress = 0
            self.failures = 0
            return True
        self.no_progress += 1
        if self.no_progress >= self.MAX_NO_PROGRESS:
            self.give_up(f'no progress toward {self.target} in '
                         f'{self.no_progress} tries (best remaining '
                         f'{self.best_remaining}, now {remaining} at {pos})')
            return False
        return True

    def detour(self, pos: Tile) -> Optional[int]:
        """How far off a shortest route start -> goal a position is (0 on
        one), or None if the goal can't be reached from it."""
        e = self.expect
        if pos not in self.detour_cache:
            a = self.route_cost(e.start, pos)
            b = self.route_cost(pos, e.goal)
            self.detour_cache[pos] = (None if a is None or b is None
                                      else a + b - e.cost)
        return self.detour_cache[pos]

    def check_expectation(self, pos: Tile, moving: bool) -> Optional[bool]:
        """True if met, False if failed (already handled), None if still in
        progress."""
        e = self.expect
        now = time.time()
        if pos == e.goal and not (e.final and moving):
            return True
        if pos == e.goal:
            # passing over the goal doesn't count: we must stop on it
            e.reached = True
        if not e.final and moving and \
                chebyshev(pos, e.goal) <= self.RECLICK_DISTANCE:
            return True
        stopped = not moving and now - self.clicked_at > self.MOVING_SECONDS
        if stopped:
            # stopped somewhere else: did the click carry us too far?
            planned = chebyshev(e.start, e.goal)
            travelled = chebyshev(e.start, pos)
            if e.reached or (planned >= 1 and
                             travelled > planned + 1):
                self.overshoots += 1
                self.fail(pos, moving, (
                    f'expected {e.what}, but overshot: clicked {planned} '
                    f'tiles away, travelled {travelled} and stopped at '
                    f'{pos} (overshoot {self.overshoots}/'
                    f'{self.MAX_OVERSHOOTS})'))
                if self.overshoots >= self.MAX_OVERSHOOTS:
                    self.give_up('clicks keep carrying us past the clicked '
                                 'tile (minimap zoom? click offset? the '
                                 'tile itself?)')
                return False
        if e.door is not None and self.obstacle == e.door:
            if self.crossed(pos, e.door):
                self.obstacle = None
            elif now - self.obstacle_since > self.OBSTACLE_SECONDS:
                self.fail(pos, moving, (
                    f'expected {e.what}, but not past the door after '
                    f'{now - self.obstacle_since:.1f}s, at {pos}'))
                return False
        reason = None
        detour = self.detour(pos)
        if detour is None:
            # standing where the map data says we can't: data mismatch,
            # not evidence we're lost; the other checks still apply
            self.app.msg.append(f'{pos} not walkable in map data')
        elif detour > self.MAX_DETOUR:
            reason = (f'went off route (detour {detour}), e.g. a door or '
                      f'gate on the way is shut')
        elif stopped:
            reason = f'stopped {chebyshev(pos, e.goal)} tiles short'
        elif now > e.deadline:
            reason = 'took too long'
        if reason is None:
            return None
        self.fail(pos, moving, f'expected {e.what}, but {reason}, at {pos}')
        return False

    def fail(self, pos: Tile, moving: bool, message: str):
        e = self.expect
        self.expect = None
        self.failures += 1
        if e is not None and e.door is not None:
            # the door gets a fresh time budget only after it's been
            # checked again
            self.obstacle = None
        self.log.warning(f'Walk: {message} '
                         f'(failure {self.failures}/{self.MAX_FAILURES})')
        if moving:
            self.stop_walking(pos)
            self.settling = True
        if e is not None and e.door is not None:
            # the door is the likely cause: look at it again before
            # trusting it's open
            n = self.door_failures.get(e.door, 0) + 1
            self.door_failures[e.door] = n
            if self.door_open == e.door:
                self.door_open = None
            if self.door_blind == e.door:
                self.door_blind = None
            if n >= self.MAX_DOOR_FAILURES:
                self.give_up(f"can't get through door {e.door} ({n} "
                             f"failures)")
                return
            self.log.info(f'Walk: re-checking door {e.door}')
        if self.failures >= self.MAX_FAILURES:
            self.give_up(f'{self.failures} failed expectations without '
                         f'getting closer')
            return
        self.note_progress(pos)

    # door handling

    def door_entity(self, a: Tile, b: Tile) -> GameEntity:
        """Game entity covering both tiles either side of a door, so the
        door (on the edge between them) is inside its screen box.

        Entity keys put the anchor at the north-west tile; width extends
        east and height extends south."""
        key = (a, b)
        if self.door is not None and self.door_key == key:
            return self.door
        x = min(a[0], b[0])
        y = max(a[1], b[1])
        width = abs(a[0] - b[0]) + 1
        height = abs(a[1] - b[1]) + 1
        entity = self.client.game_screen.create_game_entity(
            'door', 'door', (0, 0), self.client, self.client,
            tile_width=width, tile_height=height,
        )
        entity.set_global_coordinates(x, y, a[2])
        self.door, self.door_key = entity, key
        self.door_hovers = 0
        self.door_clicked_at = 0.0
        self.app._update_game_entities(entity)
        return entity

    def check_door(self, pos: Tile, nxt: Tile) -> None:
        """Standing next to a door: find out whether it's open (mouse text
        "Close ...") and open it if it's shut ("Open ..."). Sets
        ``door_open`` once it's seen open."""
        door = self.door_entity(pos, nxt)
        key = self.door_key
        now = time.time()
        if now - self.door_clicked_at < self.DOOR_OPEN_SECONDS:
            self.app.msg.append('waiting for door to open')
            return
        text = self.client.mouse_options.state or ''
        hovering = door.is_inside(*self.client.screen.mouse_xy)
        if hovering and re.search('^[Cc]lose', text):
            self.log.info(f'Walk: door {key} is open ({text!r})')
            self.door_open = key
            return
        if self.door_clicked_at and hovering and \
                re.search('^[Oo]pen', text):
            # clicked "Open" earlier and it still says "Open"
            n = self.door_failures.get(key, 0) + 1
            self.door_failures[key] = n
            self.door_clicked_at = 0.0
            self.log.warning(f'Walk: door {key} still shut after '
                             f'clicking Open ({text!r}), failure {n}/'
                             f'{self.MAX_DOOR_FAILURES}')
            if n >= self.MAX_DOOR_FAILURES:
                self.give_up(f"door {key} won't open")
            return
        if self.app._click_entity(door, 1.2, 1.8, '^[Oo]pen'):
            self.log.info(f'Walk: clicked Open on door {key}')
            self.door_clicked_at = time.time()
            self.door_hovers = 0
            return
        if self.door_clicked_at and hovering:
            # clicked "Open" and the option is gone from the door's tiles:
            # it swung open, out of the way (its "Close" is elsewhere)
            self.log.info(f'Walk: door {key} opened ("Open" gone '
                          f'after clicking, mouse text {text!r})')
            self.door_open = key
            return
        self.door_hovers += 1
        if self.door_hovers >= self.DOOR_HOVERS:
            # no "Open" / "Close" on the door's tiles: maybe someone opened
            # it and it swung out of the way. Walk through once, with the
            # usual expectation; a failure brings us back here.
            self.door_hovers = 0
            if self.door_failures.get(key, 0):
                self.give_up(f"can't find door {key} on screen (mouse "
                             f"text {text!r}) and walking through failed")
                return
            self.log.info(f'Walk: door {key} not found after '
                          f'{self.DOOR_HOVERS} hovers (mouse text '
                          f'{text!r}), trying to walk through once')
            self.door_blind = key

    # main loop

    def update(self):
        """Keep the door's place on screen up to date (every frame)."""
        if self.door is not None:
            self.app._update_game_entities(self.door)

    def step(self) -> Optional[bool]:
        """One frame: None while walking, True once stopped on the target,
        False if we gave up (see ``failed``)."""
        if not self.arrived and self.failed is None:
            self._step()
        if self.failed is not None:
            return False
        return True if self.arrived else None

    def _step(self) -> None:
        now = time.time()
        pos = self.position()
        if pos is None:
            self.app.msg.append('no position')
            if self.blind_since is None:
                self.blind_since = now
            elif now - self.blind_since > self.MAX_BLIND_SECONDS:
                self.give_up(f'no position for {now - self.blind_since:.0f}s')
            return
        self.blind_since = None

        if pos != self.last_pos:
            self.last_pos = pos
            self.last_moved = now
        moving = now - self.last_moved < self.MOVING_SECONDS

        if self.target is None and not self.plan(pos):
            self.give_up('no path')
            return
        if self.close_enough(pos):
            self.log.info(f'Walk: close enough to {self.target} at {pos} in '
                          f'{now - self.started:.0f}s, {self.clicks} clicks')
            self.arrived = True
            return

        # check the last click did what we expected before anything else
        if self.expect is not None:
            met = self.check_expectation(pos, moving)
            if met is None:
                self.app.msg.append(f'{self.expect.what}: '
                                f'{chebyshev(pos, self.expect.goal)} left')
                return
            if not met:
                return
            self.log.info(f'Walk: done: {self.expect.what}')
            self.expect = None
            # a met expectation only counts if it got us closer
            if not self.note_progress(pos):
                return

        if self.settling:
            # a click while still running back would be measured from a
            # position we've already left
            if moving or now - self.clicked_at < self.MOVING_SECONDS:
                self.app.msg.append('waiting to stop')
                return
            self.settling = False

        if pos == self.target:
            if moving:
                self.app.msg.append('waiting to stop at the target')
                return
            self.log.info(f'Walk: arrived at {pos} in '
                          f'{now - self.started:.0f}s, {self.clicks} clicks')
            self.arrived = True
            return

        if not self.plan(pos):
            self.give_up('no path')
            return
        if pos == self.target:
            return  # already there: next frame
        remaining = self.path
        doors = self.cm.doors_on_path(remaining)
        through: Optional[DoorKey] = None
        if doors and doors[0] == 0:
            key = (remaining[0], remaining[1])
            if self.door_open == key or self.door_blind == key:
                through = key
                doors = doors[1:]
            elif moving:
                self.app.msg.append('waiting to stop at door')
                return
            else:
                self.check_door(pos, remaining[1])
                return

        # furthest tile we can click on the minimap without passing a
        # (closed) door
        limit = doors[0] if doors else len(remaining) - 1
        reach = self.max_click_tiles()
        best = 0
        for k in range(1, limit + 1):
            t = remaining[k]
            if math.hypot(t[0] - pos[0], t[1] - pos[1]) > reach:
                break
            best = k
        if best == 0:
            self.app.msg.append('nothing in reach')
            return
        final = best == limit
        if final and not doors and self.spread:
            best = random.randint(max(1, best - self.spread), best)
        cost = best + (DOOR_COST if through else 0)
        self.walk_to(pos, remaining[best], cost, final=final,
                     door=through)
        if through:
            # past this door (or failed at it): look again next time
            self.door_open = self.door_blind = None
