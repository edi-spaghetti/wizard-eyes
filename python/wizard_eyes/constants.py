"""
Provides named variables for commonly used values, usually for OpenCV
"""
from dataclasses import dataclass
from typing import Tuple


WHITE = (255, 255, 255)
WHITEA = (255, 255, 255, 255)
BLACK = (0, 0, 0)
BLACKA = (0, 0, 0, 255)
REDA = (0, 0, 255, 255)
DARK_REDA = (0, 0, 200, 255)
FILL = -1

RED = (92, 92, 205, 255)  # indianred
BLUE = (235, 206, 135, 255)  # skyblue
YELLOW = (0, 215, 255, 255)  # gold
GREEN = (0, 100, 0, 255)  # darkgreen

DEFAULT_ZOOM = 512
DEFAULT_BRIGHTNESS = 0

@dataclass
class Colour:
    lower: Tuple[int, int, int]
    upper: Tuple[int, int, int]

@dataclass
class ColourHSV:
    black = Colour(lower=(0, 0, 0), upper=(180, 255, 30))
    white = Colour(lower=(0, 0, 231), upper=(180, 18, 255))
    red1 = Colour(lower=(159, 50, 70), upper=(180, 255, 255))
    red2 = Colour(lower=(0, 50, 70), upper=(9, 255, 255))
    green = Colour(lower=(36, 50, 70), upper=(89, 255, 255))
    blue = Colour(lower=(90, 50, 70), upper=(128, 255, 255))
    yellow = Colour(lower=(25, 50, 70), upper=(35, 255, 255))
    purple = Colour(lower=(129, 50, 70), upper=(158, 255, 255))
    orange = Colour(lower=(10, 50, 70), upper=(24, 255, 255))
    gray = Colour(lower=(0, 0, 40), upper=(180, 18, 230))

    @classmethod
    def colours(cls):
        return [
            k for k in cls.__dict__.keys()
            if isinstance(getattr(cls, k), Colour)
        ]
