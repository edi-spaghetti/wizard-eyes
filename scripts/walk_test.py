"""Test traveller: walk to a tile using the cache-built collision map (see
``wizard_eyes.navigation.walker``).

With ``--object`` it walks to the nearest object of that name (by walking
distance, from the cache-built object index) instead, to a tile the object
can be used from, then hovers the object and clicks it once the mouse text
shows the option given with ``--action`` (default: its first option).

Needs the camera north / top-down (start-up checks do this) and the grid
info overlay for coordinates. Run::

    venv\\Scripts\\python scripts\\walk_test.py --to 3253 3266 --run-time 300
    venv\\Scripts\\python scripts\\walk_test.py --object "Bank booth" --action Bank
"""
import time
from typing import Optional

from wizard_eyes.application import Application
from wizard_eyes.game_entities.entity import GameEntity
from wizard_eyes.interactions import UseOption
from wizard_eyes.navigation.collision_map import CollisionMap
from wizard_eyes.navigation.object_map import MapObject, ObjectMap, describe
from wizard_eyes.navigation.walker import (  # noqa: F401 (transport_test)
    Tile, Walker, chebyshev,
)


class WalkTest(Application):

    def create_parser(self):
        parser = super().create_parser()
        target = parser.add_mutually_exclusive_group(required=True)
        target.add_argument('--to', type=int, nargs='+',
                            help='Target tile: x y [z] (z defaults to the '
                                 'current plane).')
        target.add_argument('--object',
                            help='Walk to the nearest object with this '
                                 'name and use it.')
        parser.add_argument('--action',
                            help='With --object: the option to click '
                                 '(default: the first one).')
        parser.add_argument('--radius', type=int, default=60,
                            help='With --object: search radius in tiles.')
        parser.add_argument('--no-doors', action='store_true',
                            help="Don't route through doors.")
        return parser

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cm: Optional[CollisionMap] = None
        self.walker: Optional[Walker] = None
        self.last_pos: Optional[Tile] = None
        self.last_moved = 0.0
        self.started = time.time()
        # objects
        self.om: Optional[ObjectMap] = None
        self.obj: Optional[MapObject] = None
        self.obj_entity: Optional[GameEntity] = None
        self.obj_clicked_at = 0.0
        self.use: Optional[UseOption] = None

    def setup(self):
        self.args = getattr(self, 'args', None) or self.parse_args()
        self.cm = CollisionMap()
        if self.args.object:
            self.om = ObjectMap()
        # mouse text: white action ("Open") and cyan object name ("Gate")
        self.client.mouse_options.add_colours('white', 'cyan')
        log = self.client.logger
        log.info(f'Walk test: collision data {self.cm.meta}')

    @property
    def log(self):
        return self.client.logger

    def give_up(self, reason: str):
        self.log.warning(f'Walk test: giving up: {reason} '
                         f'(after {time.time() - self.started:.0f}s)')
        self.stop()

    def position(self) -> Optional[Tile]:
        x, y, z = self.client.gauges.minimap.gps.get_coordinates()
        if x == -1:
            return None
        return int(x), int(y), int(z)

    # objects

    def choose_object(self, pos: Tile) -> bool:
        found = self.om.nearest_reachable(
            pos, self.cm, name=self.args.object, action=self.args.action,
            radius=self.args.radius, doors=not self.args.no_doors)
        if found is None:
            nearby = self.om.find(self.args.object, self.args.action,
                                  near=pos, radius=self.args.radius)
            self.log.warning(
                f'Walk test: no reachable {self.args.object!r} within '
                f'{self.args.radius} tiles of {pos}; in range:\n'
                + describe(nearby[:10], pos))
            return False
        self.obj, path = found
        self.walker = Walker(self, [path[-1]], cm=self.cm,
                             doors=not self.args.no_doors)
        tiles = self.om.interaction_tiles(self.obj, self.cm)
        self.log.info(f'Walk test: object {self.obj}, size {self.obj.size}, '
                      f'used from {tiles}; walking to {path[-1]} '
                      f'({len(path) - 1} steps)')
        return True

    def object_entity(self) -> GameEntity:
        """Game entity over the object's tiles. Entity keys put the anchor
        at the north-west tile; width extends east, height south."""
        if self.obj_entity is None:
            o = self.obj
            sx, sy = o.size
            entity = self.client.game_screen.create_game_entity(
                'object', 'object', (0, 0), self.client, self.client,
                tile_width=sx, tile_height=sy,
            )
            entity.set_global_coordinates(o.x, o.y + sy - 1, o.plane)
            self.obj_entity = entity
            self._update_game_entities(entity)
        return self.obj_entity

    def use_object(self, pos: Tile):
        """Use the option on the object (left click or right-click menu,
        see ``wizard_eyes.interactions.UseOption``), then stop."""
        action = self.args.action or self.obj.actions[0]
        if self.obj_clicked_at:
            if time.time() - self.obj_clicked_at > 3:
                self.log.info(f'Walk test: used {self.obj.name!r} '
                              f'({action}) from {pos} in '
                              f'{time.time() - self.started:.0f}s')
                self.stop()
            return
        if self.use is None:
            self.use = UseOption(self, self.object_entity(), action,
                                 self.obj.names or [self.obj.name])
        result = self.use.step()
        if result:
            self.obj_clicked_at = time.time()
        elif result is False:
            self.give_up(f'could not use {action!r} on {self.obj}')

    # main loop

    def update(self):
        self.client.gauges.update()
        self.client.game_screen.update()
        self.client.mouse_options.update()
        if self.walker is not None:
            self.walker.update()
        if self.obj_entity is not None:
            self._update_game_entities(self.obj_entity)

    def action(self):
        pos = self.position()
        if pos is None:
            self.msg.append('no position')
            return
        if pos != self.last_pos:
            self.last_pos = pos
            self.last_moved = time.time()

        if self.walker is None:
            if self.args.object:
                if not self.choose_object(pos):
                    self.stop()
                return
            to = self.args.to
            target = (to[0], to[1], to[2] if len(to) > 2 else pos[2])
            self.walker = Walker(self, [target], cm=self.cm,
                                 doors=not self.args.no_doors)

        result = self.walker.step()
        if result is False:
            self.stop()
        elif result and self.obj is not None:
            self.use_object(pos)
        elif result:
            self.stop()


def main():
    app = WalkTest()
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
