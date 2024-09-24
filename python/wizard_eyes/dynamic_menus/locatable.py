from typing import Generator, Tuple
from os.path import exists

from ..file_path_utils import get_root
from ..game_objects.game_objects import GameObject

import cv2
import numpy


class Locatable(GameObject):
    """A game object that can be dynamically located somewhere on game screen.

    On update if this object has not yet been located, it will attempt to
    locate itself. If the object is located, the located attribute will be
    set to True.

    The class must have a frame template and mask in order to be located. Once
    located it can also define an alpha image, which can be iterated over to
    locate sub-widgets in a fixed position.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.auto_locate: bool = True
        self.located: bool = False
        self.alpha: numpy.ndarray = self.get_alpha()
        self.frame: numpy.ndarray = self.load_templates(
            ['frame'], cache=False).get('frame')
        self.frame_mask: numpy.ndarray = self.load_masks(
            ['frame'], cache=False).get('frame')

    def locatable_init(self):
        """Initialise instance attributes specific to locatable.

        Some classes may use multiple inheritance and so do not run this
        class's __init__ method. In this case, the locatable_init method should
        be called after the object has been initialised.

        """
        self.auto_locate: bool = True
        self.located: bool = False
        self.alpha = self.get_alpha()
        self.frame = self.load_templates(['frame'], cache=False).get('frame')
        self.frame_mask = self.load_masks(['frame'], cache=False).get('frame')

    def get_alpha(self) -> numpy.ndarray:
        """Get the alpha image for the locatable object.

        :returns: The alpha image as a numpy array.

        """
        path = self.resolve_path(
            name='alpha', root=get_root()).replace('.npy', '.png')
        if exists(path):
            return cv2.imread(path)
        return numpy.uint8([[[0, 0, 0]]])

    def sort_unique_alpha(self, unique: numpy.ndarray) -> numpy.ndarray:
        """Optionally override the sort order for unique alpha colours.

        :param unique: The unique alpha colours to be sorted.
        :returns: The sorted unique alpha colours.

        """
        return unique

    def iterate_alpha(self) -> Generator[
        Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]], None, None
    ]:
        """Iterate over unique alpha colours.

        Colours in the alpha image are converted to a tuple of ints
        (b, g, r, a). If the alpha image is empty, a warning is logged and
        the generator returns immediately.

        :yields: Tuple of colour and global bounding box values for each
            unique alpha colour.

        """

        if not self.located:
            self.logger.warning(f'{self} must be located for iterating alpha!')
            return

        if not self.alpha.any():
            self.logger.warning(f'No alpha data found for {self}!')
            return

        unique = numpy.unique(
            self.alpha.reshape(-1, self.alpha.shape[2]), axis=0)
        unique = self.sort_unique_alpha(unique)

        for bgr in unique:
            if not bgr.any():
                continue

            # convert colour to tuple
            bgra = cv2.cvtColor(bgr.reshape(1, 1, 3), cv2.COLOR_BGR2BGRA)
            colour = tuple(int(c) for c in bgra.reshape(4))

            # get bounding box
            ay, ax = numpy.where(
                numpy.all(
                    self.alpha == colour[:3], axis=-1
                )
            )
            x1, y1, x2, y2 = min(ax), min(ay), max(ax), max(ay)
            bbox = self.globalise(x1, y1, x2, y2)

            yield colour, bbox

    def locate(self) -> bool:
        """Locate the interface itself. This method can also be used to check
        if the interface is currently open.

        :returns: True if the interface is located, False otherwise.

        """

        if self.frame is None:
            return False

        mask = self.frame_mask
        matches = cv2.matchTemplate(
            # must be client img, because we don't know where
            # the widget is yet
            self.client.img,
            self.frame,
            cv2.TM_CCOEFF_NORMED,
            mask=mask
        )

        (my, mx) = numpy.where(matches >= self.match_threshold)
        if len(mx) > 1:
            self.logger.warning(
                f'Found {len(mx)} matches for {self}, '
                'assuming the first one is correct.'
            )

        # assume interfaces are unique and we only get one match
        for y, x in zip(my, mx):
            h, w = self.frame.shape
            x1, y1, x2, y2 = self.client.globalise(x, y, x + w - 1, y + h - 1)
            self.set_aoi(x1, y1, x2, y2)

            # some area of game screen are 'black' and cause false positives
            img = self.img.copy()
            # apply the mask to the image
            img = cv2.bitwise_and(img, mask)
            img[img == 3] = 0
            if img.any():
                return True

        return False

    def update(self):
        """Update the locatable menu, attempting to locate it if it is not
        already located."""
        super().update()
        if self.covered_by_right_click_menu():
            return

        if not self.located:
            if self.locate():
                self.located = True
