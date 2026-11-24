"""Teleport methods and routes, from ``config/teleports.json``.

A route to a destination is a list of steps (methods), tried in the
configured order of preference, e.g. to the Grand Exchange:

1. Construction cape "Tele to POH" -> house jewellery box, hotkey L
2. Varrock teleport tablet "Grand Exchange"
3. Ring of wealth "Rub" -> "Grand Exchange"

Which routes are possible depends on what the account has where (worn,
inventory, bank slot); a bot passes that in (see :func:`available_routes`).
After each step its landing is checked (:meth:`Method.landed`).
"""

import difflib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from .file_path_utils import get_root

CONFIG_PATH = os.path.join('config', 'teleports.json')

Tile = Tuple[int, int, int]


def squash(text: str) -> str:
    return re.sub(r'\W', '', text or '').lower()


def name_matches(text: str, names: Iterable[str]) -> bool:
    """Whether mouse text (or part of it) names one of ``names``. Mouse
    text reads "<option> [n more options] | <Name>"; the name is after the
    bar when there is one. OCR drops spaces and slips a letter or two."""
    # pieces are per text colour (white option, orange item name, ...),
    # some can be empty
    parts = [squash(p) for p in (text or '').split('|')]
    for got in (p for p in parts if p):
        for name in names:
            want = squash(name)
            if want and (want in got or difflib.SequenceMatcher(
                    None, got, want).ratio() >= 0.8):
                return True
    return False


@dataclass
class Method:
    key: str
    option: str
    lands: dict
    """{'tiles': [[x, y, plane, r], ...]} or {'instance': name}."""
    cost: str = 'free'
    item: List[str] = field(default_factory=list)
    where: List[str] = field(default_factory=list)
    """Where the item can be used from: worn and / or inventory."""
    obj: Optional[str] = None
    """An object to use instead of an item (e.g. the jewellery box)."""
    needs: Optional[str] = None
    """Where the player must be first (e.g. 'house')."""
    choice: Optional[str] = None
    """Option to pick in the menu / dialogue that follows."""
    worn_option: Optional[str] = None
    """Option to use when worn, if different (e.g. a ring's destinations
    are right-click options when worn)."""
    hotkey: Optional[str] = None
    """Key that picks ``choice`` in the menu (e.g. the jewellery box's
    "l" for the Grand Exchange)."""
    minigame: Optional[str] = None
    """Emblem template in the Minigame Teleport menu."""
    cooldown: float = 0.0
    """Seconds between uses (shared by all minigame teleports)."""
    note: str = ''

    def landed(self, pos: Optional[Tile], in_instance: bool = False
               ) -> Optional[bool]:
        """Whether a position is where this method lands: True / False,
        or None when that can't be told from a position (instances)."""
        if 'instance' in self.lands:
            return in_instance if in_instance else None
        if pos is None:
            return False
        return self.landing_point(pos) is not None

    def landing_point(self, pos: Tile) -> Optional[Tile]:
        """The landing point (from the wiki) a position is within range
        of, or None."""
        for x, y, z, r in self.lands.get('tiles', []):
            if pos[2] == z and max(abs(pos[0] - x), abs(pos[1] - y)) <= r:
                return x, y, z
        return None


class Teleports:

    def __init__(self, path: Optional[str] = None):
        path = path or os.path.join(get_root(), CONFIG_PATH)
        with open(path, encoding='utf-8') as f:
            raw = json.load(f)
        self.methods: Dict[str, Method] = {}
        for key, m in raw['methods'].items():
            self.methods[key] = Method(
                key=key, option=m['option'], lands=m['lands'],
                cost=m.get('cost', 'free'), item=m.get('item', []),
                where=m.get('from', []), obj=m.get('object'),
                needs=m.get('needs'), choice=m.get('choice'),
                worn_option=m.get('worn_option'), hotkey=m.get('hotkey'),
                minigame=m.get('minigame'),
                cooldown=m.get('cooldown', 0.0),
                note=m.get('note', ''))
        self.menus: Dict[str, dict] = raw.get('menus', {})
        """Teleport menus of objects (e.g. the ornate jewellery box): the
        object, the option that opens it, a title template to tell it's
        open, and every hotkey -> destination."""
        self.routes: Dict[str, List[List[str]]] = raw['routes']
        for dest, routes in self.routes.items():
            for route in routes:
                for key in route:
                    if key not in self.methods:
                        raise ValueError(f'route to {dest}: unknown '
                                         f'method {key!r}')

    def usable(self) -> List[str]:
        """Methods that get a bot somewhere by themselves with what's
        implemented: inventory items with a single option
        (``item_teleport``) and minigames (``minigame_teleport``), landing
        on the map."""
        return sorted(k for k, m in self.methods.items()
                      if m.lands.get('tiles') and not m.choice
                      and not m.needs and (m.minigame or (
                          m.item and 'inventory' in m.where)))

    def menu_for(self, method: Method) -> Optional[dict]:
        """The menu a method's object opens, if it's in ``menus``."""
        for menu in self.menus.values():
            if method.obj and menu.get('object', '').lower() ==                     method.obj.lower():
                return menu
        return None

    def route_methods(self, destination: str) -> List[List[Method]]:
        return [[self.methods[k] for k in route]
                for route in self.routes.get(destination, [])]

    def available_routes(self, destination: str,
                         have: Iterable[str]) -> List[List[Method]]:
        """Routes whose items are all available, in order of preference.
        ``have`` lists item names the account has somewhere it can get
        them from (worn, inventory or the bank tag in use)."""
        have = list(have)
        out = []
        for route in self.route_methods(destination):
            if all(not m.item or any(name_matches(h, m.item) for h in have)
                   for m in route):
                out.append(route)
        return out
