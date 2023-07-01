from dataclasses import dataclass

from ..game_entities.entity import GameEntity



@dataclass
class Obstacle:
    """Dataclass to represent a single entity that needs to be clicked as
    part of a route. It could be an agility obstacle, or it could be a ladder
    to another floor of a building."""

    map_name: str
    label: str
    mouse_text: str
    timeout: float
    success_label: str
    fail_idx: int = None
    fail_label: str = None
    entity: GameEntity = None
    offsets: tuple = None
    routes: tuple = None
    additional_delay: float = None
    mouse_thresh_lower: int = None
    allow_partial: bool = True
    multi: int = 1
    min_confidence: float = 0.0
    """Minimum confidence required to swap map to the next obstacle."""
    custom_click_box: bool = False
    fallback_confidence_after: float = -float('inf')
    fallback_confidence_before: float = -float('inf')
    range: int = 1
