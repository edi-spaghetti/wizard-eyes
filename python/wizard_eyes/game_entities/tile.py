import pickle
from os.path import isfile

import cv2

from ..game_objects.game_objects import GameObject
from ..file_path_utils import get_root
from ..constants import WHITE
from ..script_utils import wait_lock


class TileMarker(GameObject):
    """Class to estimate game screen position of tiles."""

    PATH_TEMPLATE = '{root}/data/game_screen/tile_grid_{zoom}.pickle'
    DEFAULT_GRID = {(0, 0): (0, 0)}
    GRID_POINT_RADIUS = 2

    def __init__(self, zoom, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zoom = zoom
        self.grid = self.load_grid()

    @property
    def grid_path(self):
        return self.resolve_path(root=get_root(), zoom=self.zoom)

    def load_grid(self):
        path = self.grid_path

        if isfile(path):
            with open(path, 'rb') as f:
                grid = pickle.load(f)
            return grid
        else:
            return self.DEFAULT_GRID

    def save_grid(self):
        with open(self.grid_path, 'wb') as f:
            pickle.dump(self.grid, f)

    @wait_lock
    def draw(self):
        super().draw()

        if 'grid' in self.client.args.show:
            for (x, y), (rx, ry) in self.grid.items():

                px, _, _, py = self.parent.player.tile_bbox()
                gx = px + rx
                gy = py + ry
                gx, gy, _, _ = self.client.localise(gx, gy, gx, gy)

                cv2.circle(
                    self.client.original_img, (gx, gy),
                    self.GRID_POINT_RADIUS, WHITE, 1)

                cv2.putText(
                    self.client.original_img, f'{x},{y}', (gx - 4, gy + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, WHITE, thickness=1
                )
