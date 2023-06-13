from dataclasses import dataclass
from typing import List

from numpy import ndarray
import cv2

from ..constants import REDA


@dataclass
class TemplateGroup:
    """Represents a group of game object images that have a common purpose,
    e.g. coin_1, coin_2, coin_3, coin_4, coin_5, coin_25, coin_100, coin_250,
    coin_1000 and coin_10000 might all be grouped under the name 'coin'."""
    name: str
    templates: List['Template']
    colour: tuple = REDA
    quantity: int = 1


@dataclass
class Template:
    """Represents a single game object image."""

    name: str
    """Name of template, used to load image and mask from file."""
    alias: str = None
    """Mask alias, used to load mask with a different name from template."""
    image: ndarray = None
    mask: ndarray = None
    threshold: float = .99
    method: int = cv2.TM_CCOEFF_NORMED
    invert: bool = False
