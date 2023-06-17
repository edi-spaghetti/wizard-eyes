import argparse
import mouse

from wizard_eyes.application import Application
from wizard_eyes.constants import GREEN, BLACKA
from wizard_eyes.game_entities.screen import ClickChecker


class ClickCheckerApp(Application):
    """Run a demo on the click checker feature."""

    def create_parser(self) -> argparse.ArgumentParser:
        """Define what type of click to check, or check for both."""
        parser = super().create_parser()

        parser.add_argument(
            '--click-type',
            default='',
            choices=('red', 'yellow'),
            help='Type of click to check for',
        )

        return parser

    def __init__(self, *args, **kwargs):
        """Create an instance of the click checker assign to app attribute."""
        super().__init__(*args, **kwargs)
        self.click_checker = ClickChecker(self.client, self.client)

    def on_success(self):
        """Deactivate the click checker, but keep it alive for another 3
        seconds with a different colour, so you can see the result."""
        self.click_checker.active = False
        self.click_checker.colour = GREEN
        self.click_checker.timeout_at += 3

    def on_failure(self):
        """Deactivate the click checker, but keep it alive for another 3
        seconds with a different colour, so you can see the result."""
        self.click_checker.active = False
        self.click_checker.colour = BLACKA
        self.click_checker.timeout_at += 3

    def init_click_checker(self):
        """Initialise the click checker. This is called when the user clicks
        the mouse. It will only be called if the click checker is not already
        active, if the mouse is inside the client window and the application
        has not been paused."""
        if self.client.right_click_menu.located:
            return
        # currently click checker only supports one click at a time
        if self.click_checker.active:
            return
        if self.sleeping:
            return

        x, y = self.client.screen.mouse_xy
        if not self.client.is_inside(x, y):
            return

        c_type = self.args.click_type
        if c_type:
            c_type = c_type == 'red'

        self.click_checker.start(
            x, y,
            red=c_type,
            on_success=self.on_success,
            on_failure=self.on_failure,
        )

    def setup(self):
        """Set up a callback on all mouse left clicks to check for a red
        or yellow click."""
        self.client.args.show = {'mouse', '*bbox', '*state'}
        self.click_checker.logger.setLevel(10)
        self.click_checker.save_images = True
        mouse.on_click(self.init_click_checker)

    def update(self):
        """Update the click checker."""
        self.click_checker.update()

    def action(self):
        """Log if the click checker is currently active."""
        self.msg.append(str(self.click_checker.active))


def main():
    app = ClickCheckerApp()
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
