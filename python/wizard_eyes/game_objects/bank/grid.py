"""Bank geometry: the item slot grid and the tag tab column.

With the Bank Tags / Bank Tag Layouts plugins every tag tab is laid out on
an even grid of item slots. The bank's width is fixed, but its height
follows the client window, so the number of slot rows (and of visible tag
tabs) is worked out from where the bottom button row (the "All" withdraw
button, padlock etc.) is: no slot or tag tab is ever below it.

Everything here is relative to the bank's top left corner (where the bank
frame is located), never to fixed screen positions.
"""

from typing import Tuple

BBox = Tuple[int, int, int, int]

SLOT_ORIGIN = (57, 77)
"""Top left of slot 0's item sprite, from the bank's top left (where the
herb templates captured with the old hand-painted alpha line up)."""
SLOT_PITCH = (48, 36)
"""Distance between slots across and down."""
SLOT_SIZE = (36, 32)
"""Width and height of a slot (item sprite area)."""
COLUMNS = 8

SLOT_BUTTON_GAP = 2
"""Minimum gap (px) between the last slot row and the bottom buttons."""

TAG_ORIGIN = (7, 98)
"""Top left of the first tag tab, from the bank's top left."""
TAG_PITCH = 40
TAG_SIZE = (38, 38)
TAG_BUTTON_GAP = 52
"""The tag column's down arrow sits between the last tag tab and the
bottom buttons."""

BUTTON_SIZE = 36
BUTTONS = {
    # name: (x offset from the "All" button's left edge, width)
    'rearrange': (-226, 36),
    'note': (-189, 36),
    'quantity_1': (-148, 36),
    'quantity_5': (-111, 36),
    'quantity_10': (-74, 36),
    'quantity_x': (-37, 36),
    'quantity_all': (0, 36),
    'search': (43, 46),
    'placeholders': (92, 36),
    'empty_containers': (137, 36),
    'deposit_inventory': (174, 36),
    'deposit_worn': (211, 36),
}
"""Bottom row buttons, positioned from the "All" withdraw button (found by
template), all in one row of the same height."""

QUANTITY_BUTTONS = ('quantity_1', 'quantity_5', 'quantity_10', 'quantity_x',
                    'quantity_all')
"""One of these is always selected."""
TOGGLE_BUTTONS = ('rearrange', 'note', 'search', 'placeholders')
"""Clicking toggles them on (selected) and off."""
ACTION_BUTTONS = ('empty_containers', 'deposit_inventory', 'deposit_worn')
"""Clicking does something (deposits); never selected."""


def button_bbox(name: str, all_x: int, all_y: int) -> BBox:
    """Button box from the "All" button's top left (same frame of
    reference as the result)."""
    dx, w = BUTTONS[name]
    x1 = all_x + dx
    return x1, all_y, x1 + w - 1, all_y + BUTTON_SIZE - 1


TAG_SELECTED_BGR = ((40, 75, 90), (50, 85, 100))
"""Background colour range of the selected tag tab (others are about
(50, 61, 69))."""


def slot_rows(buttons_top: int) -> int:
    """Visible slot rows above the bottom buttons (``buttons_top`` relative
    to the bank's top)."""
    bottom = buttons_top - SLOT_BUTTON_GAP
    usable = bottom - SLOT_ORIGIN[1] - SLOT_SIZE[1]
    return max(0, usable // SLOT_PITCH[1] + 1)


def slot_bbox(index: int) -> BBox:
    """Slot box relative to the bank's top left."""
    row, col = divmod(index, COLUMNS)
    x1 = SLOT_ORIGIN[0] + col * SLOT_PITCH[0]
    y1 = SLOT_ORIGIN[1] + row * SLOT_PITCH[1]
    return x1, y1, x1 + SLOT_SIZE[0] - 1, y1 + SLOT_SIZE[1] - 1


def slot_colour(index: int) -> Tuple[int, int, int, int]:
    """A distinct BGRA colour per slot (for drawing, like the old
    hand-painted alpha): blue by column, green by row, red by block of 24 rows."""
    row, col = divmod(index, COLUMNS)
    return 40 + col * 30, 40 + (row % 24) * 9, 200 - (row // 24) * 40, 255


def tag_count(buttons_top: int) -> int:
    bottom = buttons_top - TAG_BUTTON_GAP
    usable = bottom - TAG_ORIGIN[1] - TAG_SIZE[1]
    return max(0, usable // TAG_PITCH + 1)


def tag_bbox(index: int) -> BBox:
    x1 = TAG_ORIGIN[0]
    y1 = TAG_ORIGIN[1] + index * TAG_PITCH
    return x1, y1, x1 + TAG_SIZE[0] - 1, y1 + TAG_SIZE[1] - 1
