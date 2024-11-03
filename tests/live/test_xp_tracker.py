from argparse import ArgumentParser

from wizard_eyes.application import Application


class TestXPTracker(Application):
    """This test app checks the XP Tracker is correctly able to find xp drops.

    XP can be added to the command line arguments using the --xp flag. This
    will then run a test application with everything set up to scan the game
    screen for the xp tracker widget, and individual xp drops will be
    highlighted with a bounding box.

    """

    def __init__(self, *args, **kwargs):
        """Basic init, just make the message a bit longer."""
        super().__init__(*args, **kwargs)
        self.msg_length = 250

    def create_parser(self) -> ArgumentParser:
        """Add an extra argument to the parser for the xp"""

        parser = super().create_parser()
        parser.add_argument('--xp', nargs='+', default=[])
        return parser

    def setup(self):
        """Set up the xp tracker."""

        xp = self.client.gauges.xp_tracker
        xp.track_skills(*self.args.xp)

        self.client.args.show.add('xp')
        self.client.args.show.add('*bbox')

    def update(self):
        """We only need to update the gauges to locate and update the xp
        tracker. Since the xp tracker is actually only conceptually connected
        to the gauges (in that it measures something) strictly speaking we
        could do a more streamlined update of just locating the xp tracker, but
        since this module acts like a demo for this feature, this is the
        easiest way to update xp tracker if you're not concerned with the
        extra compute."""

        self.client.gauges.update()

    def action(self):
        """Log xp drop quantity to console."""

        xp = self.client.gauges.xp_tracker
        self.msg.append(
            f'XP Drops: {xp.find_xp_drops(*self.args.xp)} ({xp.located})'
        )


def main():
    app = TestXPTracker()
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
