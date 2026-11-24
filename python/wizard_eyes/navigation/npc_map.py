"""Where NPCs are (``config/npcs.json``): the game cache's map data only has
objects. For now just enough for bankers, which never move."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

from ..file_path_utils import get_root
from .collision_map import Tile

CONFIG_PATH = Path('config') / 'npcs.json'


@dataclass(frozen=True)
class MapNPC:
    label: str
    name: str
    tile: Tile
    wander: int = 0
    """Tiles it can stray from ``tile``; 0 never moves."""
    size: Tuple[int, int] = (1, 1)

    def distance(self, tile: Tile) -> int:
        """Tiles (Chebyshev) to the nearest tile of its home position."""
        dx = max(self.tile[0] - tile[0], 0,
                 tile[0] - (self.tile[0] + self.size[0] - 1))
        dy = max(self.tile[1] - tile[1], 0,
                 tile[1] - (self.tile[1] + self.size[1] - 1))
        return max(dx, dy)


class NPCMap:

    def __init__(self, path: Union[str, Path, None] = None):
        path = Path(path) if path else Path(get_root()) / CONFIG_PATH
        raw = json.loads(path.read_text(encoding='utf8'))['npcs']
        self.npcs = [MapNPC(n['label'], n['name'], tuple(n['tile']),
                            n['wander'], tuple(n.get('size', (1, 1))))
                     for n in raw]

    def find(self, label: Optional[str] = None, near: Optional[Tile] = None,
             radius: Optional[int] = None) -> List[MapNPC]:
        """NPCs by label; with ``near``, on its plane and nearest first."""
        found = [n for n in self.npcs if label is None or n.label == label]
        if near is not None:
            found = [n for n in found if n.tile[2] == near[2] and
                     (radius is None or n.distance(near) <= radius)]
            found.sort(key=lambda n: n.distance(near))
        return found
