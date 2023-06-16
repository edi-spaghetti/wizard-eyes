from os.path import exists

from ..game_objects.game_objects import GameObject
from ..file_path_utils import get_root

import numpy


class InvalidImageError(Exception):
    """Colour correction cannot be performed on this image."""


class ColourCorrector(GameObject):
    """Applies colour correction to templates."""

    PATH_TEMPLATE = (
        '{root}/data/game_screen/colour_correction_matrix_{brightness}.npy'
    )

    def __init__(self, brightness, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.brightness = brightness
        self.cc_matrix: numpy.matrix = self.load_matrix()

    @property
    def cc_matrix_path(self):
        """Path to the colour correction matrix."""
        return self.resolve_path(root=get_root(), brightness=self.brightness)

    def load_matrix(self) -> numpy.matrix:
        """Load the colour correction matrix."""
        path = self.cc_matrix_path
        if exists(path):
            array = numpy.load(path)
            matrix = numpy.matrix(array)
        else:
            matrix = numpy.matrix([
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.],
            ], dtype=float)

        return matrix

    def save_matrix(self):
        """Save the colour correction matrix."""
        numpy.save(self.cc_matrix_path, self.cc_matrix)

    def correct(self, img: numpy.array, matrix: numpy.matrix=None):
        """Apply colour correction to an image."""
        if matrix is None:
            matrix = self.cc_matrix

        try:
            h, w, d = img.shape
            assert d == 3
        except (ValueError, AssertionError):
            msg = f'Image must be BGR, got {img.shape}.'
            self.logger.error(msg)
            raise InvalidImageError(msg)

        linear = img.reshape(-1, 3)
        corrected_linear = linear * matrix
        corrected = numpy.asarray(
            corrected_linear, dtype=numpy.uint8).reshape(h, w, 3)  # noqa

        return corrected
