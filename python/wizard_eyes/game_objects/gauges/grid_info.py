from typing import Tuple

from ..readable import OCRReadable
from ...dynamic_menus.locatable import Locatable


class Coordinate(OCRReadable):
    """Coordinate widget for the grid info gauge."""

    WHITE_LIST = '0123456789,'
    BLACK_LIST = (
        '!?@#$%&*()<>_-+=/.:;\'"'
        'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    )
    NUMERIC_MODE = '1'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_colours('white')

    def coordinates(self) -> Tuple[int, int, int]:
        """Return the x, y, z coordinates of the grid info widget."""
        try:
            x, y, z = self.state.split(',')
            return int(x), int(y), int(z)
        except ValueError:
            return -1, -1, -1


class Region(OCRReadable):
    """Region coordinate widget for the grid info gauge.

    Assumes the GridInfo plugin has been set up with
    Grid Info Type > Local Coordinates.

    """

    WHITE_LIST = '0123456789,'
    BLACK_LIST = (
        '!?@#$%&*()<>_-+=/.:;\'"'
        'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    )
    NUMERIC_MODE = '1'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_colours('white')

    def region(self) -> Tuple[int, int]:
        """Return the player's x, y region coordinates."""
        try:
            x, y, _, _ = self.state.split(',')
            return int(x), int(y)
        except ValueError:
            return -1, -1

    def tile(self) -> Tuple[int, int]:
        """Return the x, y tile coordinates of the player within the region."""
        try:
            _, _, x, y = self.state.split(',')
            return int(x), int(y)
        except ValueError:
            return -1, -1


class GridInfo(Locatable):
    """Grid info gauge game object class."""

    PATH_TEMPLATE = '{root}/data/gauges/grid_info/{name}.npy'

    ALPHA_MAPPING = {
        (0, 0, 255, 255): 'tile',
        (255, 0, 0, 255): 'region',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.locatable_init()

        self.tile: Coordinate = Coordinate(self.client, self)
        """Coordinate: The tile coordinate widget."""
        self.chunk: Coordinate = Coordinate(self.client, self)
        """Coordinate: The chunk coordinate widget."""
        self.region: Coordinate = Coordinate(self.client, self)
        """Coordinate: The region coordinate widget."""
        self.elements = []

    def update(self):
        """In addition to the usual update, locate and update child elements.

        The grid info gauge is a container for tile, chunk and region widgets.
        These will be located (if configured to do so) and updated when the
        grid info gauge is updated.

        """
        prev_located = self.located
        super().update()

        # now that we've located the grid info widget, we can update the
        # elements of the UI - setting bounding boxes for them.
        if self.located:
            if not prev_located:
                for colour, bbox in self.iterate_alpha():
                    element = getattr(self, self.ALPHA_MAPPING[colour])
                    element.set_aoi(*bbox)
                    element.colour = colour
                    self.elements.append(element)

            for element in self.elements:
                element.update()
