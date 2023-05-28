from random import randint, uniform

import cv2
import numpy

from ..game_objects.game_objects import GameObject
from ..file_path_utils import get_root
from ..constants import WHITE
from ..script_utils import wait_lock


x_angle = 180
rotation_x = numpy.matrix([
    [1, 0, 0, 0],
    [0, numpy.cos(numpy.radians(x_angle)), -numpy.sin(numpy.radians(x_angle)), 0],
    [0, numpy.sin(numpy.radians(x_angle)), numpy.cos(numpy.radians(x_angle)), 0],
    [0, 0, 0, 1]
], dtype=float)

y_angle = 90
rotation_y = numpy.matrix([
    [numpy.cos(numpy.radians(y_angle)), 0, numpy.sin(numpy.radians(y_angle)), 0],
    [0, 1, 0, 0],
    [-numpy.sin(numpy.radians(y_angle)), 0, numpy.cos(numpy.radians(y_angle)), 0],
    [0, 0, 0, 1]
], dtype=float)

z_angle = 180
rotation_z = numpy.matrix([
    [numpy.cos(numpy.radians(z_angle)), -numpy.sin(numpy.radians(z_angle)), 0, 0],
    [numpy.sin(numpy.radians(z_angle)), numpy.cos(numpy.radians(z_angle)), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], dtype=float)


class TileMarker(GameObject):
    """Class to estimate game screen position of tiles."""

    PATH_TEMPLATE = '{root}/data/game_screen/projection_matrix_{zoom}.npy'
    GRID_DRAW_RADIUS = 15
    GRID_POINT_RADIUS = 2

    def __init__(self, zoom, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zoom = zoom
        self.projection_matrix: numpy.matrix = self.load_matrix()

    @property
    def projection_matrix_path(self):
        return self.resolve_path(root=get_root(), zoom=self.zoom)

    def load_matrix(self) -> numpy.matrix:
        path = self.projection_matrix_path
        array = numpy.load(path)
        matrix = numpy.matrix(array)

        return matrix

    def save_matrix(self):
        numpy.save(self.projection_matrix_path, self.projection_matrix)

    def create_projection_matrix(self):
        """Run monte carlo simulation to get estimated projection matrix
        based on position of player tile marker."""

        x1, y1, x2, y2 = self.client.localise(
            *self.client.game_screen.player.tile_bbox())

        i0 = (x1, y1)
        i1 = (x2, y1)
        i2 = (x1, y2)
        i3 = (x2, y2)

        r0 = numpy.matrix([[0, 0, 0, 1]])
        r1 = numpy.matrix([[0, 0, 1, 1]])
        r2 = numpy.matrix([[1, 0, 0, 1]])
        r3 = numpy.matrix([[1, 0, 1, 1]])

        estimate = numpy.matrix([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
        ], dtype=float)

        def norm2(a, b):
            dx = b[0] - a[0]
            dy = b[1] - a[1]
            return dx ** 2 + dy ** 2

        def evaluate(mat):
            c0 = self.project(r0, mat)
            c1 = self.project(r1, mat)
            c2 = self.project(r2, mat)
            c3 = self.project(r3, mat)

            return (norm2(i0, c0) +
                    norm2(i1, c1) +
                    norm2(i2, c2) +
                    norm2(i3, c3))

        def peturb(mat, amount):
            new_matrix = mat.copy()
            new_matrix[randint(0, 3), randint(0, 3)] += uniform(-amount, amount)
            return new_matrix

        def approximate(mat, amount, n=1000):
            est = evaluate(mat)
            for _ in range(n):
                mat2 = peturb(mat, amount)
                est2 = evaluate(mat2)
                if est2 < est:
                    mat = mat2
                    est = est2
            return mat, est

        estimate_val = 0
        for _ in range(100):
            estimate, estimate_val = approximate(estimate, 1)
            estimate, estimate_val = approximate(estimate, .1)

        return estimate, estimate_val

    def project(self, point: numpy.matrix, matrix=None):
        """
        Convert 3D world vector into 2D screen vector.

        :param numpy.matrix point: 3D world vector that we want to project
            onto the screen. Must have shape (4, 4), in the format x, y, z, w.
            Note that in world vectors y is vertical height, x is north-south
            and z in east-west. w MUST be 1.
        :param numpy.matrix matrix: Projection matrix to apply transformation.
            Defaults to currently loaded projection matrix if None.

        :rtype: tuple[int, int]

        """

        if matrix is None:
            matrix = self.projection_matrix

        # TODO: remove the need for rotation!
        point = point * rotation_y
        point = point * rotation_z
        point = point * matrix

        x = point[0, 0]
        y = point[0, 1]
        # IMPORTANT: in 2D space z is depth (which is y in 3D space)
        # TODO: implement depth
        z = point[0, 2]
        w = point[0, 3]

        ox, oy = self.client.game_screen.player.bbox_offset()
        sx = int(self.client.width * (x / w + 1) / 2.) + ox
        sy = int(self.client.height -  # inverted
                 self.client.height * (y / w + 1) / 2.) + oy

        return sx, sy

    @wait_lock
    def draw(self):
        super().draw()

        if 'grid' in self.client.args.show:
            # NOTE: z because these are 3D coordinates
            # TODO: support y - pull from map values
            for x in range(-self.GRID_DRAW_RADIUS, self.GRID_DRAW_RADIUS):
                for z in range(-self.GRID_DRAW_RADIUS, self.GRID_DRAW_RADIUS):
                    vector = numpy.matrix([[x, 0, z, 1]])
                    sx, sy = self.project(vector)  # screen x and y

                    cv2.circle(
                        self.client.original_img, (sx, sy),
                        self.GRID_POINT_RADIUS, WHITE, 1)

                    cv2.putText(
                        self.client.original_img, f'{x},{z}', (sx - 4, sy + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, WHITE, thickness=1
                    )
