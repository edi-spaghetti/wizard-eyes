from wizard_eyes.application import Application


class ExampleApp(Application):
    """Example application class.
    Copy and modify this class definition to make a bot or utility."""

    def __init__(self, *args, **kwargs):
        """Init function is run once when the application is created. This is
        where you should add any extra attributes you might need, or load any
        resources that need to be loaded before the setup function."""
        super().__init__(*args, **kwargs)

    def setup(self):
        """Setup function is run once before the main loop. This is where you
        should load any resources you need, and setup any game objects."""

    def update(self):
        """Update function is run once per frame. This is where you should
        update any game objects."""

    def action(self):
        """Action function is run once per frame. This is where you should
        perform any actions you want to perform, such as clicking, typing,
        etc."""


def main():
    app = ExampleApp()
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
