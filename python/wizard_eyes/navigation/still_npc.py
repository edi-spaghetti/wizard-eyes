"""An NPC that never moves (``config/npcs.json``), for clicking on while we
stand still with the camera top-down.

Placed from the player's highlighted tile, not the tile grid: the Grand
Exchange's paving looks like a grid of two tiles, which throws that off. A
person only covers the middle of their tile, so that's all that's aimed at."""
from typing import Optional, Tuple

from ..camera import find_player_tile
from ..game_entities.entity import GameEntity
from .npc_map import MapNPC

MIDDLE = 0.5
"""Share of the tile (across and down) that's aimed at."""

Box = Tuple[int, int, int, int]


def middle_of_tile(tile: Box, dx: int, dy: int, share: float = MIDDLE) -> Box:
    """Middle of the tile ``dx`` east and ``dy`` north of one at ``tile``
    (x, y, width, height)."""
    x, y, w, h = tile
    cx = x + w * (dx + .5)
    cy = y + h * (.5 - dy)
    return (round(cx - w * share / 2), round(cy - h * share / 2),
            round(cx + w * share / 2), round(cy + h * share / 2))


class StillNPC(GameEntity):

    def __init__(self, app, npc: MapNPC):
        super().__init__('npc', (0, 0), app.client, app.client)
        self.npc = npc
        self.set_global_coordinates(*npc.tile)
        self._box: Optional[Box] = None
        app._update_game_entities(self)

    def get_bbox(self):
        img = self.client.original_img
        tile = find_player_tile(img) if img is not None else None
        px, py, _ = self.client.gauges.minimap.gps.get_coordinates()
        if tile is not None and px != -1:
            # the outline is drawn on the tile's edges
            box = middle_of_tile((tile.x, tile.y, tile.w, tile.h),
                                 self.npc.tile[0] - int(px),
                                 self.npc.tile[1] - int(py))
            self._box = tuple(self.client.globalise(*box))
        # hidden (e.g. behind an interface): where it was
        return self._box or super().get_bbox()
