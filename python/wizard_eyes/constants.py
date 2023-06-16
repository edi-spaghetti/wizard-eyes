"""
Provides named variables for commonly used values, usually for OpenCV
"""

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

COLOUR_DICT_HSV = {
    # colour: ((upper), (lower))
    'black': ((180, 255, 30), (0, 0, 0)),
    'white': ((180, 18, 255), (0, 0, 231)),
    'red1': ((180, 255, 255), (159, 50, 70)),
    'red2': ((9, 255, 255), (0, 50, 70)),
    'green': ((89, 255, 255), (36, 50, 70)),
    'blue': ((128, 255, 255), (90, 50, 70)),
    'yellow': ((35, 255, 255), (25, 50, 70)),
    'purple': ((158, 255, 255), (129, 50, 70)),
    'orange': ((24, 255, 255), (10, 50, 70)),
    'gray': ((180, 18, 230), (0, 0, 40))
}
