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


def menu_line_matches(value: str, option: str, names: List[str]) -> bool:
    """Whether an OCR'd menu line is ``option`` + one of ``names``."""
    value = _squash(value)
    option = _squash(option)
    if not value.startswith(option) or value[len(option):].startswith('-'):
        return False
    rest = value[len(option):]
    return any(difflib.SequenceMatcher(None, rest, _squash(n)).ratio() >= 0.6
               for n in names)


def mouse_text_pattern(option: str, names: Optional[List[str]] = None) -> str:
    """Regex for the option in the mouse-over text. It isn't always first
    (the reader can put "/ N more options" before it); not followed by "-"
    or a letter. With ``names``, one of them must follow it (spaces
    optional, the reader drops them): "Collect" alone also matches the
    collection box's "Collect to bank" button."""
    pattern = f'{re.escape(option)}(?![-A-Za-z])'
    if names:
        spaced = ['\\s*'.join(re.escape(c) for c in _squash(n))
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
        for item in items:
            if menu_line_matches(item.value, self.option, self.names):
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
