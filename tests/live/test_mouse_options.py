import argparse
import keyboard

from wizard_eyes.application import Application
from wizard_eyes.constants import ColourHSV, Colour


class MouseOptionsApp(Application):
    """Demo of how to use the mouse options class to find read the left/right
    click context of the mouse at current position."""

    def create_parser(self) -> argparse.ArgumentParser:
        parser = super().create_parser()

        parser.add_argument(
            '--method', type=str, default='tesseract',
            choices=['tesseract', 'template'],
            help='The method to use to find letters in the mouse options area '
                 'on top left of game screen.',
        )

        parser.add_argument(
            '--colours', nargs='+', type=str,
            choices=ColourHSV.colours(),
            default=['white'],
            help='Colours to detect in the mouse options area.'
        ),

        return parser


    def setup(self):
        """Set the mouse options class up to detect per input params.
        Add hotkeys to toggle between using tesseract and not, and to save
        the threshed image(s)."""

        self.client.args.show = {'mouse', '*bbox', '*state'}

        if self.args.method == 'tesseract':
            self.client.mouse_options.use_tesseract = True
        else:
            self.client.mouse_options.use_tesseract = False

        # add all colours, since we can toggle between using them or not
        self.client.mouse_options.add_colours(*self.args.colours)

        # set of every letter, upper and lower case
        names = set(''.join([chr(i) for i in range(65, 91)]))
        names.update([name.lower() for name in names])
        self.client.mouse_options.load_templates(names)
        self.client.mouse_options.load_masks(names)

        keyboard.add_hotkey('asterisk', self.save_threshed)

        keyboard.add_hotkey('c', self.toggle_combined)
        keyboard.add_hotkey('t', self.toggle_tesseract)

    def toggle_combined(self):
        """Toggle whether to process the mouse options area as a combined
        mask of all colours, or as individual masks for each colour."""
        self.client.mouse_options.process_combined = (
            not self.client.mouse_options.process_combined
        )

    def toggle_tesseract(self):
        """Toggle between tesseract and template matching for detection."""
        self.client.mouse_options.use_ocr = (
            not self.client.mouse_options.use_ocr
        )

    def save_threshed(self):
        """Save the threshed image(s) to disk. They are saved in the
        mouse options directory with the name test<colour index>.png."""
        mo = self.client.mouse_options
        path = mo.resolve_path().replace('.npy', '.png')
        for i, img in enumerate(mo.process_img(mo.img)):
            save_path = path.replace('test.png', f'test_{i}.png')
            self.client.screen.save_img(img, save_path)

    def update(self):
        """Update the mouse options class and print it's state on the
        console."""

        self.client.mouse_options.update()
        state = str(self.client.mouse_options.state)
        state = state.strip()
        self.msg.append(
            f'state: {state}, '
            f'conf: {self.client.mouse_options.confidence}')

    def action(self):
        """No action necessary, but required for abstract method."""


def main():
    app = MouseOptionsApp()
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
