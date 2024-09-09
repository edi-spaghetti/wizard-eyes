from wizard_eyes.application import Application


class TestGridInfo(Application):
    """App to test the grid info gauge."""

    def __init__(self, *args, **kwargs):
        """Cache grid info to class attribute."""
        super().__init__(*args, **kwargs)
        self.grid_info = self.client.gauges.grid_info
        self.mouse_option = self.client.mouse_options

    def setup(self):
        """Set mouse info up with additional colours.

        This is to check the black/white lists for mouse options are working
        in conjunction with the grid info's black/white lists. Mouse options
        will see (almost) any text, whereas grid info will only see the
        number coordinates.

        """
        self.mouse_option.add_colours('white', 'yellow', 'cyan')

    def update(self):
        """Update the grid info and mouse options."""
        self.grid_info.update()
        self.mouse_option.update()

    def action(self):
        """Log the grid info coordinates and mouse options text."""

        mo = self.mouse_option.state
        value = self.grid_info.tile.coordinates()
        self.msg.append(f'tile: {value}, mo: {mo}')


def main():
    app = TestGridInfo()
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
