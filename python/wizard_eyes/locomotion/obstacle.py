from dataclasses import dataclass
from typing import Union, Tuple, List

from ..game_entities.entity import GameEntity


@dataclass
class Obstacle:
    """Dataclass to represent a single entity that needs to be clicked as
    part of a route. It could be an agility obstacle, or it could be a ladder
    to another floor of a building."""

    map_name: str
    label: Union[str, Tuple[int, int, int]]
    mouse_text: str
    timeout: float
    success_label: Union[
        str,
        Tuple[int, int, int],
        List[Union[Tuple[int, int, int], str]]
    ]
    """Union[str, tuple, list]: Label/node to check for success of a
    particular obstacle. It can be a named obstacle (ideal if unique), or a
    tuple of coordinates. It can also be a list of multiple of these options,
    allowing for multiple success locations."""

    fail_idx: int = None
    fail_label: Union[str, Tuple[int, int, int]] = None
    entity: GameEntity = None

    offsets: Tuple[int, int, int, int] = None
    """Tuple: (x1, y1, x2, y2) offsets to adjust the bounding box of the
    entity."""

    routes: tuple = None
    additional_delay: float = 0
    mouse_thresh_lower: int = None
    allow_partial: bool = False
    multi: int = 1
    min_confidence: float = 10.0
    """Minimum confidence required to swap map to the next obstacle."""
    custom_click_box: bool = False
    fallback_confidence_after: float = -float('inf')
    fallback_confidence_before: float = -float('inf')
    range: int = 1
