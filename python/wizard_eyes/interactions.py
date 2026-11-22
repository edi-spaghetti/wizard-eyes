"""Using an option on something in the game: left-click when it's the
default option, otherwise through the right-click menu.

    use = UseOption(app, entity, 'Collect', ['Bank booth'])
    ...  # every frame:
    result = use.step()   # None: working on it, True: clicked, False: failed

The option is matched in the mouse-over text (left click) or in the
right-click menu, read with OCR (needs ``client.ocr``). OCR drops spaces and
misreads the odd letter ("collectbbankbookth"), so a menu line must start
with the option (not followed by "-", so "Climb" doesn't pick "Climb-up")
and the rest must look like one of the names.

Every step carries an expectation: the menu must appear within a couple of
seconds and contain the option; otherwise the mouse is moved off the menu
and it's tried once more before giving up.
"""

import difflib
import re
from typing import List, Optional

from .game_objects.game_objects import GameObject

MENU_SECONDS = 2.0
MAX_TRIES = 2
MAX_HOVERS = 20


def _squash(text: str) -> str:
    return re.sub(r'\s', '', text or '').lower()


MENU_MIN_SCORE = 0.65
MENU_MARGIN = 0.05


def menu_line_score(value: str, option: str, names: List[str]) -> float:
    """How well an OCR'd menu line reads as ``option`` + one of ``names``
    (0 to 1). Menu OCR often gets the first letter wrong ("kub" for
    "Rub", "lele" for "Tele"), so both parts are compared fuzzily: the
    start of the line against the option (60%), the rest against the
    names (40%). A line where the option is followed by "-" doesn't count
    for an option without one ("Climb" is not "Climb-up")."""
    value = _squash(value)
    option = _squash(option)
    if '-' not in option and value[len(option):len(option) + 1] == '-':
        return 0.0
    best = 0.0
    for k in range(max(1, len(option) - 2), len(option) + 3):
        ratio = difflib.SequenceMatcher(None, value[:k], option).ratio()
        if ratio < 0.6:
            continue
        rest = value[k:]
        name = max(difflib.SequenceMatcher(None, rest, _squash(n)).ratio()
                   for n in names) if names else 1.0
        if name < 0.5:
            continue  # right option, something else
        best = max(best, 0.6 * ratio + 0.4 * name)
    return best


def best_menu_line(values: List[str], option: str,
                   names: List[str]) -> Optional[int]:
    """Index of the menu line that is the option, or None if no line
    scores well enough or two are too close to tell apart (better to stop
    than to click the wrong one, e.g. "Drop")."""
    scores = sorted(((menu_line_score(v, option, names), i)
                     for i, v in enumerate(values)), reverse=True)
    if not scores or scores[0][0] < MENU_MIN_SCORE:
        return None
    if len(scores) > 1 and scores[0][0] - scores[1][0] < MENU_MARGIN:
        return None
    return scores[0][1]


def menu_line_matches(value: str, option: str, names: List[str]) -> bool:
    """Whether an OCR'd menu line reads as ``option`` + one of ``names``
    (on its own; prefer :func:`best_menu_line` over a whole menu)."""
    return menu_line_score(value, option, names) >= MENU_MIN_SCORE


def mouse_text_pattern(option: str, names: Optional[List[str]] = None) -> str:
    """Regex for the option in the mouse-over text. It isn't always first
    (the reader can put "/ N more options" before it); not followed by "-"
    or a letter. With ``names``, one of them must follow it (spaces
    optional, the reader drops them): "Collect" alone also matches the
    collection box's "Collect to bank" button."""
    pattern = r'\s*'.join(re.escape(c) for c in option.replace(' ', ''))
    pattern += '(?![-A-Za-z])'
    if names:
        # letters and digits only, anything (or nothing) between them: the
        # reader drops spaces and some punctuation ("Construct. cape(t)"
        # reads "Constructcape(t)")
        spaced = [r'\W*'.join(re.escape(c) for c in re.sub(r'\W', '', n))
                  for n in names]
        pattern += f'.*(?:{"|".join(spaced)})'
    return f'(?i){pattern}'


class UseOption:

    def __init__(self, app, target: GameObject, option: str,
                 names: List[str]):
        self.app = app
        self.client = app.client
        self.target = target
        self.option = option
        self.names = names
        self.menu_at = 0.0
        self.tries = 0
        self.hovers = 0
        self.done = False

    @property
    def log(self):
        return self.client.logger

    def step(self) -> Optional[bool]:
        if self.done:
            return True
        if self.target.context_menu is not None or self.menu_at:
            return self._menu()
        if self.app._click_entity(self.target, 1.0, 1.5,
                                  mouse_text_pattern(self.option,
                                                     self.names)):
            self.log.info(f'Used {self.option!r} on {self.names[0]!r} '
                          f'(left click, mouse text '
                          f'{self.client.mouse_options.state!r})')
            self.done = True
            return True
        text = self.client.mouse_options.state or ''
        hovering = self.target.is_inside(*self.client.screen.mouse_xy)
        named = any(_squash(n) in _squash(text) for n in self.names)
        if hovering and named and self.client.ocr is not None:
            # right target, but the option isn't the left-click one
            if self.tries >= MAX_TRIES:
                self.log.warning(f'No {self.option!r} in the menu of '
                                 f'{self.names[0]!r} after {self.tries} '
                                 f'tries')
                return False
            if self.app._right_click(self.target, set_ocr=True):
                self.menu_at = self.client.time
                self.tries += 1
            return None
        self.hovers += 1
        if self.hovers > MAX_HOVERS:
            self.log.warning(f'{self.names[0]!r} not found under the mouse '
                             f'after {self.hovers} hovers (mouse text '
                             f'{text!r})')
            return False
        return None

    def _menu(self) -> Optional[bool]:
        menu = self.target.context_menu
        items = menu.items if menu is not None else []
        index = best_menu_line([i.value for i in items], self.option,
                               self.names) if items else None
        if index is not None:
            item = items[index]
            if item.value:
                item.click(tmin=1.0, tmax=1.5, pause_before_click=True)
                self.log.info(f'Used {self.option!r} on {self.names[0]!r} '
                              f'(menu line {item.value!r})')
                self.menu_at = 0.0
                self.done = True
                return True
        if self.client.time - self.menu_at < MENU_SECONDS:
            return None
        self.log.warning(f'Expected {self.option!r} in the right-click menu '
                         f'of {self.names[0]!r}, menu: '
                         f'{[i.value for i in items]}')
        if menu is not None:
            self.client.screen.mouse_away_object(menu)
        self.menu_at = 0.0
        if self.tries >= MAX_TRIES:
            return False
        return None
