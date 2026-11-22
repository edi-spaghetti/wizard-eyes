"""Map the player-owned house of the logged-in account: which room template
sits in each cell of the house grid and how it's turned (see
``wizard_eyes.navigation.house``). Saved to
``data/maps/house/<account>.local.json``.

Start it standing in the house (camera north, top-down)::

    venv\\Scripts\\python scripts\\house_map.py
    venv\\Scripts\\python scripts\\house_map.py --hotspot "Jewellery box space"

It maps the floor the player is on:

1. The first room becomes cell (0, 0). Its turn comes from a short walk
   in a known direction (minimap click), checked with a second walk.
2. For every doorway of every known room that leads to an unknown cell,
   it walks two tiles through the doorway. If the overlay then reads a
   different template chunk, the new room is in that cell; its turn is
   the one that puts the player where the click was aimed, checked with
   one more walk (a mismatch isn't recorded). No change: nothing there.
3. With ``--hotspot``, it hovers that hotspot in every room that has one
   and records what's built there (mouse-over text).

Every walk has an expectation; a mismatch is logged and the doorway is
skipped, and the mapper gives up after a few mismatches in a row.
"""
import re
import time
from typing import List, Optional, Set, Tuple

from wizard_eyes.application import Application
from wizard_eyes.bank_pin import account_from_title
from wizard_eyes.navigation.collision_map import (CollisionMap, Tile,
                                                  _STEP_BITS)
from wizard_eyes.navigation.house import (CHUNK, SIDES,
                                          HouseLayout, HousePosition, Room,
                                          in_template_area,
                                          is_empty_template, rotate_mask,
                                          rotation_from_step, template_chunk)
from wizard_eyes.navigation.object_map import ObjectMap

STOP_SECONDS = 1.2
"""The player has stopped once the tile hasn't changed for this long."""
START_SECONDS = 1.8
"""...and, if it hasn't moved at all, not before this long after the
click."""
WALK_SECONDS = 2.0
SECONDS_PER_TILE = 0.35
MAX_MISMATCHES = 3


def chebyshev(a, b) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


class HouseMap(Application):

    def create_parser(self):
        parser = super().create_parser()
        parser.add_argument('--account',
                            help='Account name (default: from the window '
                                 'title).')
        parser.add_argument('--fresh', action='store_true',
                            help='Forget the saved layout and start over.')
        parser.add_argument('--hotspot', action='append', default=[],
                            help='Hotspot to record furniture for, e.g. '
                                 '"Jewellery box space" (repeatable).')
        parser.add_argument('--until', action='append', default=[],
                            help='Stop exploring once a room with this '
                                 'hotspot is known, then keep only the '
                                 'first room, that room and rooms between '
                                 '(repeatable). Also recorded like '
                                 '--hotspot.')
        parser.add_argument('--no-explore', action='store_true',
                            help='Only record hotspots for known rooms.')
        return parser

    def setup(self):
        self.args = getattr(self, 'args', None) or self.parse_args()
        self.client.mouse_options.add_colours('white', 'cyan', 'orange')
        self.done = False
        self.mismatches = 0

    @property
    def log(self):
        return self.client.logger

    # reading the game

    def update(self):
        self.client.gauges.update()
        self.client.game_screen.update()
        self.client.mouse_options.update()

    def refresh(self, wait=0.25):
        time.sleep(wait)
        self.client.update()
        self.update()
        if getattr(self, 'tracker', None) is not None:
            self.tracker.update(self.client.gauges.minimap.gps)

    def template(self) -> Optional[Tile]:
        """The overlay's tile (a template tile in the house)."""
        t = getattr(self, 'tracker', None)
        if t is not None and t.template is not None:
            return t.template
        x, y, z = self.client.gauges.minimap.gps.get_coordinates()
        return None if x == -1 else (int(x), int(y), int(z))

    def wait_stopped(self, seconds: float) -> Optional[Tile]:
        """Template tile once the player has stopped (or after
        ``seconds``)."""
        start = time.time()
        end = start + seconds
        last, since, first, moved = None, start, None, False
        while time.time() < end:
            self.refresh()
            t = self.template()
            if first is None:
                first = t
            if t != last:
                moved = moved or (last is not None)
                last, since = t, time.time()
            elif t is not None and time.time() - since >= STOP_SECONDS                     and (moved or time.time() - start >= START_SECONDS):
                # (the player can take a tick or two to start moving)
                return t
        return last

    # acting

    def reach(self) -> float:
        mm = self.client.gauges.minimap
        return (mm.safe_radius - 12) / mm.tile_size

    def click_offset(self, dx: int, dy: int):
        """Click the minimap ``dx, dy`` tiles from the player (house
        directions: the minimap is north up)."""
        mm = self.client.gauges.minimap
        x1, y1, _, _ = mm.get_bbox()
        ox, oy = mm.orb_xy
        x = x1 + ox + dx * mm.tile_size
        y = y1 + oy - dy * mm.tile_size
        self.client.screen.click_aoi(x - 1, y - 1, x + 1, y + 1, speed=0.5)

    def walk(self, pos: Tile, target: Tile) -> Optional[Tile]:
        """Click ``target`` (house tile) from ``pos``; the template tile
        once stopped."""
        d = chebyshev(pos, target)
        self.click_offset(target[0] - pos[0], target[1] - pos[1])
        return self.wait_stopped(WALK_SECONDS + d * SECONDS_PER_TILE)

    def mismatch(self, message: str) -> bool:
        """Log a mismatch; True if it's time to give up."""
        self.mismatches += 1
        self.log.warning(f'HOUSE expected {message} '
                         f'({self.mismatches}/{MAX_MISMATCHES})')
        return self.mismatches >= MAX_MISMATCHES

    # mapping

    def first_room(self, start: Tile) -> Optional[Room]:
        """The room the player is in, as cell (0, 0) of a new layout: its
        turn from a walk of two tiles, checked with a walk back."""
        chunk = template_chunk(start)
        z = start[2]
        for side, (dx, dy) in SIDES.items():
            # stay in the room whichever way it's turned: only sides with
            # room for two steps (in house directions we don't know the
            # local tile yet, so just try and see)
            pos0 = start
            house_step = (2 * dx, 2 * dy)
            self.click_offset(*house_step)
            pos1 = self.wait_stopped(WALK_SECONDS + 2 * SECONDS_PER_TILE)
            if pos1 is None or template_chunk(pos1) != chunk:
                self.log.info(f'HOUSE first room: walking {side} left the '
                              f'room ({pos1}); back and try another way')
                self.click_offset(-2 * dx, -2 * dy)
                back = self.wait_stopped(WALK_SECONDS + 2 * SECONDS_PER_TILE)
                if back is None or template_chunk(back) != chunk:
                    self.mismatch(f'to be back in the first room, at {back}')
                    return None
                start = back
                continue
            step = (pos1[0] - pos0[0], pos1[1] - pos0[1])
            r = rotation_from_step(step, house_step)
            if r is None:
                self.log.info(f'HOUSE first room: no movement {side}')
                continue
            room = Room((0, 0, z), chunk, r)
            # check: a walk back must land where the turn says
            here = room.to_house(pos1)
            goal = (here[0] - dx * 2, here[1] - dy * 2, z)
            got = self.walk(here, goal)
            if got is None or room.to_house(got) != goal:
                if self.mismatch(f'walking back {side} to template '
                                 f'{room.to_template(goal)}, got {got}'):
                    return None
                start = got or start
                continue
            self.log.info(f'HOUSE first room: template {chunk}, turned '
                          f'{r} (template step {step} for house {side})')
            return room
        return None

    def doorways(self, room: Room, side: str) -> List[Tile]:
        """House tiles on a room's edge where its template opens to
        ``side``."""
        dx, dy = SIDES[side]
        bit = _STEP_BITS[(dx, dy)]
        ox, oy, z = room.origin
        out = []
        for k in range(CHUNK):
            if dx:
                lx, ly = (CHUNK - 1 if dx > 0 else 0), k
            else:
                lx, ly = k, (CHUNK - 1 if dy > 0 else 0)
            tile = (ox + lx, oy + ly, z)
            m = rotate_mask(self.base_cm.mask(*room.to_template(tile)),
                            room.rotation)
            if m & bit:
                out.append(tile)
        return out

    def room_route(self, start: Tuple[int, int, int],
                   goal: Tuple[int, int, int]) -> Optional[list]:
        """Cells from start to goal through neighbouring known rooms."""
        prev = {start: None}
        queue = [start]
        while queue:
            cell = queue.pop(0)
            if cell == goal:
                route = []
                while cell is not None:
                    route.append(cell)
                    cell = prev[cell]
                return route[::-1]
            for dx, dy in SIDES.values():
                nxt = (cell[0] + dx, cell[1] + dy, cell[2])
                if nxt in self.layout.rooms and nxt not in prev:
                    prev[nxt] = cell
                    queue.append(nxt)
        return None

    def free_tile(self, room: Room) -> Tile:
        """A tile near the middle of a room that the template leaves free
        (hotspots, where furniture may be built, count as blocked), so a
        click there doesn't walk us to the nearest reachable tile, which
        could be outside."""
        ox, oy, z = room.origin
        tiles = [(ox + x, oy + y, z) for x in range(1, CHUNK - 1)
                 for y in range(1, CHUNK - 1)]
        free = [t for t in tiles
                if self.base_cm.mask(*room.to_template(t)) == 0xFF]
        return min(free or tiles, key=lambda t: abs(t[0] - ox - 3.5)
                   + abs(t[1] - oy - 3.5))

    def go_to_room(self, pos: Tile, room: Room) -> Optional[Tile]:
        """Walk into a known room, one neighbouring room at a time (click
        the middle of the next room; the game finds the way through the
        doorway). The house tile once there."""
        if room.contains(pos):
            return pos
        here = self.layout.room_at(pos)
        route = self.room_route(here.cell, room.cell) if here else None
        if not route:
            self.mismatch(f'a route of known rooms from {pos} to room '
                          f'{room.cell}')
            return None
        for cell in route[1:]:
            mid = self.free_tile(self.layout.rooms[cell])
            got = self.walk(pos, mid)
            now = self.tracker.house
            if got is None or now is None or                     not self.layout.rooms[cell].contains(now):
                self.mismatch(f'to be in room {cell} after walking to its '
                              f'middle {mid}; overlay {got}, house {now}')
                return None
            pos = now
        return pos

    def try_side(self, pos: Tile, room: Room, side: str
                 ) -> Tuple[Optional[Tile], str]:
        """Walk through a doorway of ``room`` (the player is in it).
        Returns (house position, outcome)."""
        dx, dy = SIDES[side]
        doors = self.doorways(room, side)
        if not doors:
            return pos, 'wall'
        door = min(doors, key=lambda t: chebyshev(t, pos))
        target = (door[0] + 2 * dx, door[1] + 2 * dy, door[2])
        if chebyshev(pos, target) > self.reach():
            got = self.walk(pos, door)
            pos = self.tracker.house
            if pos is None or not room.contains(pos):
                self.mismatch(f'to reach doorway {door}, overlay {got}')
                return pos, 'lost'
        got = self.walk(pos, target)
        if got is None:
            return None, 'lost'
        cell = (room.cell[0] + dx, room.cell[1] + dy, room.cell[2])
        same = room.contains_template(got)
        if same and chebyshev(room.to_house(got), target) <= 3:
            # still in the room, stopped at or near the doorway (a
            # neighbour with the same template can't be told apart from
            # that; it would need the player well away from the doorway)
            back = self.tracker.house
            self.log.info(f'HOUSE {room.cell} {side}: still in the room at '
                          f'{back}: no room there')
            return back, 'none'
        chunk = template_chunk(got)
        if is_empty_template(got):
            self.log.info(f'HOUSE {room.cell} {side}: an empty cell (grass) '
                          f'there; back inside')
            return self.back_out(dx, dy), 'none'
        if not in_template_area(got):
            self.mismatch(f'to stay in the house, overlay {got}')
            return None, 'lost'
        options = []
        for r in range(4):
            cand = Room(cell, chunk, r)
            options.append((chebyshev(cand.to_house(got), target), r))
        dist, r = min(options)
        if dist > 2:
            self.mismatch(f'to be near {target} in cell {cell} after '
                          f'walking {side}; template {got} fits no turn '
                          f'(best {dist} tiles off)')
            return None, 'unclear'
        new = Room(cell, chunk, r)
        here = new.to_house(got)
        # check: walk on towards the middle of the new room
        mid = (new.origin[0] + 3 + (dx < 0), new.origin[1] + 3 + (dy < 0),
               cell[2])
        # (furniture may stop us short, so check the way we moved, not
        # where we stopped)
        check = self.walk(here, mid)
        moved = (check[0] - got[0], check[1] - got[1]) if check else (0, 0)
        aim = (mid[0] - here[0], mid[1] - here[1])
        if check is None or not new.contains_template(check) or (
                moved != (0, 0) and rotation_from_step(moved, aim) != r):
            self.mismatch(f'walking to {mid} in the new room (template '
                          f'{new.to_template(mid)}), overlay {check}')
            return self.back_out(dx, dy), 'unclear'
        if moved == (0, 0) and aim != (0, 0):
            self.log.info(f'HOUSE {cell}: could not move to check the turn '
                          f'(furniture?); keeping it from the doorway')
        self.layout.add(new)
        self.layout.save()
        self.mismatches = 0
        self.refresh(0.1)
        self.log.info(f'HOUSE new room {cell}: template {chunk}, turned '
                      f'{r}')
        return self.tracker.house, 'room'

    def explore(self, pos: Tile) -> bool:
        tried: Set[Tuple[Tuple[int, int, int], str]] = set()
        while True:
            todo = []
            for room in self.layout.rooms.values():
                if room.cell[2] != pos[2]:
                    continue
                for side, (dx, dy) in SIDES.items():
                    cell = (room.cell[0] + dx, room.cell[1] + dy,
                            room.cell[2])
                    if cell in self.layout.rooms or (room.cell, side) in \
                            tried:
                        continue
                    todo.append((room, side))
            if not todo:
                return True
            if self.found_targets():
                return True
            # rooms nearest the first room (the portal room) first, so the
            # search spreads out from it
            room, side = min(todo, key=lambda rs: (
                max(abs(rs[0].cell[0]), abs(rs[0].cell[1])),
                chebyshev(rs[0].origin, pos)))
            tried.add((room.cell, side))
            if not self.doorways(room, side):
                continue
            pos = self.go_to_room(pos, room)
            if pos is None:
                return False
            self.log.info(f'HOUSE trying {room.cell} {side}')
            pos, outcome = self.try_side(pos, room, side)
            if pos is None:
                pos = self.recover()
                if pos is None:
                    return False
            if self.mismatches >= MAX_MISMATCHES:
                return False

    def back_out(self, dx: int, dy: int) -> Optional[Tile]:
        """Walk back the way we came (into a known room); the house tile
        or None."""
        self.click_offset(-3 * dx, -3 * dy)
        self.wait_stopped(WALK_SECONDS + 3 * SECONDS_PER_TILE)
        return self.recover()

    def found_targets(self) -> bool:
        """With ``--until``: every hotspot asked for is in a known room."""
        return bool(self.args.until) and all(
            self.layout.hotspots(self.om, name) for name in self.args.until)

    def prune(self):
        """Keep only the first room, the rooms with the ``--until``
        hotspots, and rooms joining them (fewest rooms between)."""
        keep = {(0, 0, z) for (_, _, z) in self.layout.rooms}
        keep &= set(self.layout.rooms)
        targets = {room.cell for name in self.args.until
                   for room, _ in self.layout.hotspots(self.om, name)}
        for goal in targets:
            # breadth first over neighbouring known rooms
            start = (0, 0, goal[2])
            prev = {start: None}
            queue = [start]
            while queue:
                cell = queue.pop(0)
                if cell == goal:
                    break
                for dx, dy in SIDES.values():
                    nxt = (cell[0] + dx, cell[1] + dy, cell[2])
                    if nxt in self.layout.rooms and nxt not in prev:
                        prev[nxt] = cell
                        queue.append(nxt)
            cell = goal if goal in prev else None
            while cell is not None:
                keep.add(cell)
                cell = prev[cell]
            keep.add(goal)
        dropped = sorted(set(self.layout.rooms) - keep)
        for cell in dropped:
            del self.layout.rooms[cell]
        self.layout.save()
        self.log.info(f'HOUSE kept rooms {sorted(keep)}, dropped {dropped}')

    def recover(self) -> Optional[Tile]:
        """After a mismatch: find out where we are (a known room), if
        anywhere."""
        got = self.wait_stopped(3.0)
        self.tracker.house = None
        self.refresh()
        known = self.layout.rooms_with_template(got) if got else []
        if len(known) == 1:
            pos = known[0].to_house(got)
            self.tracker.house = pos
            self.log.info(f'HOUSE recovered: in room {known[0].cell} at '
                          f'{pos}')
            return pos
        self.log.error(f'HOUSE lost: overlay {got} is in '
                       f'{len(known)} known rooms; stopping')
        return None

    # furniture

    def record_hotspot(self, pos: Tile, name: str) -> Tile:
        om = self.om
        for room, tiles in self.layout.hotspots(om, name, floor=pos[2]):
            got = self.go_to_room(pos, room)
            if got is None:
                return pos
            pos = got
            self.refresh(0.5)
            x = min(t[0] for t in tiles)
            y = max(t[1] for t in tiles)
            w = max(t[0] for t in tiles) - x + 1
            h = y - min(t[1] for t in tiles) + 1
            entity = self.client.game_screen.create_game_entity(
                'hotspot', 'hotspot', (0, 0), self.client, self.client,
                tile_width=w, tile_height=h)
            entity.set_global_coordinates(x, y, pos[2])
            self._update_game_entities(entity)
            x1, y1, x2, y2 = entity.get_bbox()
            self.client.screen.mouse_to((x1 + x2) // 2, (y1 + y2) // 2)
            self.refresh(0.8)
            text = self.client.mouse_options.state or ''
            pieces = [p.strip() for p in text.split('|')[1:] if p.strip()]
            built = pieces[0] if pieces else ''
            built = re.sub(r'\s+', ' ', built)
            self.log.info(f'HOUSE {name} in room {room.cell} at {tiles}: '
                          f'mouse text {text!r}')
            if built:
                room.furniture[name] = built
                self.layout.save()
            else:
                self.log.warning(f'HOUSE nothing readable on {name} in '
                                 f'room {room.cell}')
        self.client.screen.mouse_away_object(self.client.game_screen)
        return pos

    # main

    def run_map(self):
        account = self.args.account or account_from_title(
            getattr(self.client, 'title', ''))
        if not account:
            self.log.error('HOUSE no account name (use --account)')
            return
        self.base_cm = CollisionMap()
        self.om = ObjectMap()
        self.layout = (HouseLayout(account) if self.args.fresh
                       else HouseLayout.load(account))
        self.tracker = HousePosition(self.layout)
        # the grid info overlay takes a few frames to be located
        end = time.time() + 20
        while time.time() < end and self.template() is None:
            self.refresh(0.5)
        start = self.wait_stopped(3.0)
        if not in_template_area(start):
            self.log.error(f'HOUSE not in a house (overlay {start})')
            return
        pos = self.tracker.house
        if pos is None:
            if self.layout.rooms_with_template(start):
                self.log.error(f'HOUSE template {start} is in more than one '
                               f'known room; walk somewhere unique')
                return
            for dx, dy in SIDES.values() if self.layout.rooms else ():
                self.log.info(f'HOUSE in an unmapped room ({start}); '
                              f'walking {dx, dy} to find a mapped one')
                self.click_offset(8 * dx, 8 * dy)
                self.wait_stopped(WALK_SECONDS + 10 * SECONDS_PER_TILE)
                if self.tracker.house is not None and \
                        self.layout.rooms_with_template(self.template()):
                    break
            if self.tracker.house is None and self.layout.rooms:
                self.log.error(f'HOUSE standing in an unmapped room '
                               f'({start}) of a mapped house; walk into a '
                               f'mapped room or use --fresh')
                return
            if not self.layout.rooms:
                room = self.first_room(start)
                if room is None:
                    self.log.error('HOUSE could not work out the first room')
                    return
                self.layout.add(room)
                self.layout.save()
            self.refresh()
            pos = self.tracker.house
        self.log.info(f'HOUSE at {pos} (template {self.template()})')
        if not self.args.no_explore:
            ok = self.explore(pos)
            pos = self.tracker.house or pos
            self.log.info(f'HOUSE exploring {"done" if ok else "stopped"}: '
                          f'{len(self.layout.rooms)} rooms')
        if self.args.until and self.found_targets():
            self.prune()
        for name in self.args.until + self.args.hotspot:
            pos = self.record_hotspot(pos, name)
        self.log.info(f'HOUSE layout ({self.layout.path}):\n'
                      f'{self.layout.describe()}')

    def action(self):
        if self.done:
            return
        self.done = True
        try:
            self.run_map()
        finally:
            self.stop()


def main():
    app = HouseMap()
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
