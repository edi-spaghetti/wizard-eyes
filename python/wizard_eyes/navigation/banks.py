"""Named banks (``config/banks.json``) and walking to one of them.

    route = BankRoute(app, 'varrock_west')      # or a list: picked at random
    ...  # every frame, after updating gauges, game_screen, mouse_options
    route.update()
    result = route.step()   # None: walking, True: there, False: gave up
    route.entity            # the booth / chest, once there

Which banks to use, and when to change, is up to the bot.
"""
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from ..file_path_utils import get_root
from ..game_entities.entity import GameEntity
from .collision_map import CollisionMap, Tile
from .object_map import WALL_TYPES, MapObject, ObjectMap
from .walker import Walker, chebyshev

CONFIG_PATH = Path('config') / 'banks.json'

BOOTH_RADIUS = 12
"""Booths further than this from a bank's tile belong to something else."""

OPEN_OPTIONS = re.compile('^(Bank|Use)$')
BANK_NAMES = re.compile('^(Bank booth|Bank chest|Grand Exchange booth)$')


def load_banks(path: Union[str, Path, None] = None) -> Dict[str, Tile]:
    path = Path(path) if path else Path(get_root()) / CONFIG_PATH
    raw = json.loads(path.read_text(encoding='utf8'))
    return {k: tuple(v) for k, v in raw.items() if not k.startswith('_')}


def bank_objects(om: ObjectMap, tile: Tile) -> List[MapObject]:
    """The booths / chests of the bank at a tile, nearest first. Booths
    with a deposit box option are left out (they change form, so their left
    click isn't certain)."""
    return [o for o in om.find(BANK_NAMES, OPEN_OPTIONS, near=tile,
                               radius=BOOTH_RADIUS)
            if 'Deposit-box' not in o.actions]


def bank_at(pos: Tile, banks: Dict[str, Tile],
            radius: int = BOOTH_RADIUS) -> Optional[str]:
    """The named bank a position is at, if any."""
    near = [(chebyshev(pos, t), name) for name, t in banks.items()
            if t[2] == pos[2] and chebyshev(pos, t) <= radius]
    return min(near)[1] if near else None


class BankRoute:
    """Walk to a bank. Gives up (never retries) if none of the banks can be
    walked to or the walker gives up."""

    def __init__(self, app, names: Union[str, Sequence[str]],
                 om: Optional[ObjectMap] = None,
                 cm: Optional[CollisionMap] = None,
                 banks: Optional[Dict[str, Tile]] = None,
                 rng: Optional[random.Random] = None):
        self.app = app
        self.client = app.client
        self.banks = load_banks() if banks is None else banks
        self.names = [names] if isinstance(names, str) else list(names)
        unknown = [n for n in self.names if n not in self.banks]
        if unknown or not self.names:
            raise ValueError(f'Unknown banks {unknown}; '
                             f'known: {sorted(self.banks)}')
        self.om = om or ObjectMap()
        self.cm = cm or CollisionMap()
        self.rng = rng or random.Random()
        self.name: Optional[str] = None
        self.obj: Optional[MapObject] = None
        self.walker: Optional[Walker] = None
        self.entity: Optional[GameEntity] = None
        self.failed: Optional[str] = None

    def choose(self, pos: Tile) -> bool:
        """Pick a bank (at random among those that can be walked to), one
        of its booths and the tile to use it from."""
        names = self.names[:]
        self.rng.shuffle(names)
        for name in names:
            objs = bank_objects(self.om, self.banks[name])
            self.rng.shuffle(objs)
            for obj in objs:
                path = self.om.path_to(obj, pos, self.cm)
                if path is None:
                    continue
                self.name, self.obj = name, obj
                self.walker = Walker(self.app, [path[-1]], cm=self.cm)
                self.client.logger.info(
                    f'Bank route: {name}, {obj}, {len(path) - 1} steps')
                return True
        self.failed = f'none of {self.names} can be walked to from {pos}'
        self.client.logger.error(f'Bank route: {self.failed}')
        return False

    def update(self):
        if self.walker is not None:
            self.walker.update()

    def step(self) -> Optional[bool]:
        if self.failed is not None:
            return False
        if self.entity is not None:
            return True
        if self.walker is None:
            x, y, z = self.client.gauges.minimap.gps.get_coordinates()
            if x == -1:
                self.app.msg.append('no position')
            else:
                self.choose((int(x), int(y), int(z)))
            return False if self.failed is not None else None
        result = self.walker.step()
        if result is False:
            self.failed = self.walker.failed
        elif result:
            self.entity = self._create_entity()
        return result

    @property
    def wall(self) -> Optional[int]:
        """The side of its tile the booth is on (0 west, 1 north, 2 east,
        3 south), if it's a wall / counter."""
        return self.obj.orientation \
            if self.obj is not None and self.obj.type in WALL_TYPES else None

    def _create_entity(self) -> GameEntity:
        o = self.obj
        width, height = o.size
        entity = self.client.game_screen.create_game_entity(
            'bank', 'bank', (0, 0), self.client, self.client,
            tile_width=width, tile_height=height,
        )
        # entities are anchored at the north-west tile
        entity.set_global_coordinates(o.x, o.y + height - 1, o.plane)
        self.app._update_game_entities(entity)
        return entity
