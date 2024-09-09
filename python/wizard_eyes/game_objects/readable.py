from typing import List

from .game_objects import GameObject
from ..constants import ColourHSV, Colour

from PIL import Image
import numpy
import cv2


class OCRReadable(GameObject):
    """A GameObject that can be read with OCR."""

    WHITE_LIST = ''
    """str: The characters that can be read in the image."""
    BLACK_LIST = ''
    """str: The characters that cannot be read in the image."""
    NUMERIC_MODE = '0'
    """str: The mode for reading numbers in the image."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._state = None
        """str: The state of the object."""
        self.state_changed_at = 0
        """float: The time at which the state was last changed."""

        self.colours: List[Colour] = []
        """list[Colour]: The colours to be masked in the image. The mouse text
        can be any of these colours, either combined or separately."""
        self.process_combined = False
        """bool: If true, image will be processed as a combination of all 
        the colours masked into one image. If false, each colour will be
        processed separately."""

    @property
    def state(self):
        """str: The latest text read by OCR."""
        return str(self._state)

    def process_img(self, img, combined=False):
        """Process the image into a BW binary image for OCR or template match.

        :param img: The image to be processed.
        :param combined: If true, the image will be processed as a combination
            of all the colours masked into one image. If false, each colour
            will be processed separately.

        :yields: The processed image.
        """

        if img.shape[:2] != self.img.shape[:2]:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        else:
            hsv_img = self.client.get_img_at(
                self.get_bbox(), mode=self.client.HSV
            )

        img = numpy.zeros(hsv_img.shape[:2], dtype=numpy.uint8)
        for colour in self.colours:
            mask = cv2.inRange(hsv_img, colour.lower, colour.upper)
            if not self.process_combined:
                yield mask
            else:
                img = cv2.bitwise_or(img, mask)

        if self.process_combined:
            yield img

    def add_colours(self, *colours: str):
        """Add colours to the list of colours to be masked in the image.

        :param str colours: The names of the colours to be added.
        """
        for colour in colours:
            colour = Colour(
                upper=getattr(ColourHSV, colour).upper,
                lower=getattr(ColourHSV, colour).lower)
            self.colours.append(colour)

    def ocr_read(self, img: numpy.ndarray) -> str:
        """Use OCR to read the text in the image."""

        if self.WHITE_LIST:
            self.client.ocr.SetVariable(
                'tessedit_char_whitelist', self.WHITE_LIST
            )
        if self.BLACK_LIST:
            self.client.ocr.SetVariable(
                'tessedit_char_blacklist', self.BLACK_LIST
            )
        if self.NUMERIC_MODE:
            self.client.ocr.SetVariable(
                'classify_bln_numeric_mode', self.NUMERIC_MODE
            )

        img = Image.fromarray(img)
        self.client.ocr.SetImage(img)
        state = str(self.client.ocr.GetUTF8Text())
        state = state.strip().replace('\n', '').replace('\r', '')

        return state

    def set_state(self, state: str):
        """Set the state of the object and record time changed if necessary."""

        if state != self._state:
            self.logger.debug(
                f'mouse state changed from {self._state} to {state} '
                f'at {self.client.time:.3f}')
            self._state = state
            self.state_changed_at = self.client.time

    def update(self):
        """Update the state of the object."""

        states = []
        for mask in self.process_img(self.img):
            state = self.ocr_read(mask)
            states.append(state)
        state = ' | '.join(states)
        self.set_state(state)
