"""Capture templates of the bank's bottom row buttons in each state.

Run with the bank open::

    venv\\Scripts\\python utility\\sample_bank_buttons.py --out <dir>

For every button (see ``wizard_eyes.game_objects.bank.grid.BUTTONS``) it
saves, through the engine's own screen grab:

- ``<name>``: normal, mouse elsewhere
- ``<name>_hover``: mouse over it
- ``<name>_selected`` / ``<name>_selected_hover``: switched on (red)

Toggles (note, rearrange, search, placeholders) are clicked on and off
again; for the withdraw quantity buttons the one currently selected is
re-selected at the end. "X" is never clicked (it can open an "Enter amount"
prompt) and the deposit buttons are never clicked (they deposit things):
they only get normal and hover. Every click is checked (the button must
change state within a couple of seconds) and the original settings (and the
open tag tab) are restored and checked; on any mismatch it stops and says
so.

The hover text of each button is logged, to name them. Review the images
in ``--out``, then copy them to ``<master data>/bank/buttons`` and publish.
"""
import argparse
import os
import sys
import time
from typing import Dict, Optional

import cv2
import numpy

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))

from wizard_eyes.application import Application  # noqa: E402
from wizard_eyes.game_objects.bank import grid  # noqa: E402

SETTLE = 0.6
"""Seconds to let the game redraw after moving the mouse / clicking."""
CLICK_TIMEOUT = 2.5


def differs(a: numpy.ndarray, b: numpy.ndarray) -> bool:
    return a.shape != b.shape or         numpy.abs(a.astype(int) - b.astype(int)).mean() > 6


class SampleBankButtons(Application):

    def create_parser(self):
        parser = super().create_parser()
        parser.add_argument('--out', required=True,
                            help='Directory to save templates to.')
        parser.add_argument('--only', nargs='*',
                            help='Only these buttons.')
        return parser

    def setup(self):
        self.args = getattr(self, 'args', None) or self.parse_args()
        os.makedirs(self.args.out, exist_ok=True)
        self.client.mouse_options.add_colours('white', 'cyan', 'orange')
        self.done = False
        self.original_tag = None

    def update(self):
        self.client.bank.update()

    # helpers

    @property
    def log(self):
        return self.client.logger

    def refresh(self, wait: float = SETTLE):
        time.sleep(wait)
        self.client.update()
        self.client.bank.update()
        self.client.mouse_options.update()

    def boxes(self) -> Dict[str, tuple]:
        all_ = self.client.bank.withdraw_all
        x1, y1, _, _ = all_.get_bbox()
        return {n: grid.button_bbox(n, x1, y1) for n in grid.BUTTONS}

    def grab(self, bbox) -> numpy.ndarray:
        return self.client.get_img_at(bbox).copy()

    def is_selected(self, bbox) -> bool:
        """Red button background: compare red and green in the border
        band, away from the icon."""
        img = self.client.get_img_at(bbox, mode=self.client.BGRA)
        band = numpy.concatenate([img[2:6].reshape(-1, 4),
                                  img[-6:-2].reshape(-1, 4)])
        b, g, r = numpy.median(band[:, :3], axis=0)
        return r > g + 40

    def mouse_away(self):
        """Over the bank's title bar: no hover effects there."""
        x1, y1, x2, _ = self.client.bank.get_bbox()
        self.client.screen.mouse_to(x1 + 120, y1 + 10)

    def hover(self, bbox):
        x1, y1, x2, y2 = bbox
        self.client.screen.mouse_to((x1 + x2) // 2, (y1 + y2) // 2)

    def click(self, bbox):
        x1, y1, x2, y2 = bbox
        self.client.screen.click_aoi(x1 + 8, y1 + 8, x2 - 8, y2 - 8,
                                     pause_before_click=True)

    def wait_changed(self, name: str, before: numpy.ndarray) -> bool:
        """Expectation: within a couple of seconds of the click the button
        (mouse away from it) no longer looks like it did before. Works for
        buttons that turn red and for ones that change icon (rearrange)."""
        self.mouse_away()
        start = time.time()
        while time.time() - start < CLICK_TIMEOUT:
            self.refresh(0.2)
            if differs(self.grab(self.boxes()[name]), before):
                return True
        return False

    def normal_look(self, name: str) -> numpy.ndarray:
        self.mouse_away()
        self.refresh()
        return self.grab(self.boxes()[name])

    def save(self, name: str, img: numpy.ndarray):
        path = os.path.join(self.args.out, f'{name}.npy')
        numpy.save(path, img)
        cv2.imwrite(os.path.join(self.args.out, f'{name}.png'), img)
        self.log.info(f'Saved {name} {img.shape}')

    def capture_states(self, name: str, selected: bool):
        """Normal (mouse away) and hover captures in the current state."""
        suffix = '_selected' if selected else ''
        box = self.boxes()[name]
        self.mouse_away()
        self.refresh()
        self.save(f'{name}{suffix}', self.grab(box))
        self.hover(box)
        self.refresh()
        text = self.client.mouse_options.state
        self.save(f'{name}{suffix}_hover', self.grab(box))
        self.log.info(f'{name}{suffix} hover text: {text!r}')
        self.mouse_away()

    def toggle(self, name: str, want: bool):
        before = self.normal_look(name)
        self.click(self.boxes()[name])
        if not self.wait_changed(name, before):
            if name in grid.TOGGLE_BUTTONS:
                # it may have switched a setting without looking selected:
                # click once more to undo, then stop
                self.click(self.boxes()[name])
                self.refresh()
            raise RuntimeError(f'{name}: clicked, but it did not become '
                               f'{"selected" if want else "unselected"}'
                               f'{" (clicked again to undo)" if name in grid.TOGGLE_BUTTONS else ""}')

    # main

    def run_capture(self):
        bank = self.client.bank
        if not (bank.located and bank.open):
            raise RuntimeError('Open the bank first')
        names = self.args.only or list(grid.BUTTONS)
        boxes = self.boxes()
        original = {n: self.is_selected(b) for n, b in boxes.items()}
        self.mouse_away()
        self.refresh()
        self.original_look = {n: self.grab(b) for n, b in boxes.items()}
        # toggling search, for one, leaves the open tag tab
        tag = bank.selected_tag
        self.original_tag = tag.index if tag else None
        self.log.info(f'Tag tab open now: {tag}')
        self.log.info(f'Selected now: '
                      f'{[n for n, s in original.items() if s]}')
        quantity_now: Optional[str] = next(
            (n for n in grid.QUANTITY_BUTTONS if original[n]), None)
        if quantity_now is None:
            raise RuntimeError('No withdraw quantity button looks selected')

        try:
            for name in names:
                self.capture_states(name, original[name])
                if name in grid.ACTION_BUTTONS or name == 'quantity_x':
                    continue
                if name in grid.TOGGLE_BUTTONS:
                    self.toggle(name, not original[name])
                    self.capture_states(name, not original[name])
                    self.toggle(name, original[name])
                elif name != quantity_now:
                    self.toggle(name, True)
                    self.capture_states(name, True)
                    # capture the previously selected one deselected too
                    if not os.path.exists(os.path.join(
                            self.args.out, f'{quantity_now}.npy')):
                        self.capture_states(quantity_now, False)
                    self.toggle(quantity_now, True)
        finally:
            self.restore(original)

    def wrong_now(self):
        self.mouse_away()
        self.refresh()
        return [n for n, b in self.boxes().items()
                if differs(self.grab(b), self.original_look[n])]

    def restore(self, original: Dict[str, bool]):
        """Put the settings back as they were (every button must look as
        it did at the start), and check."""
        for name in self.wrong_now():
            if name in grid.TOGGLE_BUTTONS or original[name]:
                self.log.warning(f'Restoring {name}')
                before = self.normal_look(name)
                self.click(self.boxes()[name])
                self.wait_changed(name, before)
        still = self.wrong_now()
        bank = self.client.bank
        if self.original_tag is not None:
            tag = bank.tags[self.original_tag]
            if not tag.selected:
                self.log.warning(f'Restoring tag tab {self.original_tag}')
                tag.click(tmin=0.6, tmax=1.0, pause_before_click=True)
                self.mouse_away()
                self.refresh(1.0)
                if not bank.tags[self.original_tag].selected:
                    still.append(f'tag tab {self.original_tag}')
        if still:
            self.log.error(f'Could not restore: {still}. Check the bank '
                           f'settings by hand!')
        else:
            self.log.info('Bank settings restored')

    def action(self):
        if self.done:
            return
        self.done = True
        try:
            self.run_capture()
        except RuntimeError as err:
            self.log.error(f'Stopped: {err}')
        self.stop()


def main():
    app = SampleBankButtons()
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
