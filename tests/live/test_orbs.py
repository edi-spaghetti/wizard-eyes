from wizard_eyes.application import Application


class TestOrbsApp(Application):
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

        self.client.gauges.orb.hitpoints.update()
        self.client.gauges.orb.prayer.update()
        self.client.gauges.orb.run_energy.update()
        self.client.gauges.orb.special_attack.update()

    def action(self):
        """Action function is run once per frame. This is where you should
        perform any actions you want to perform, such as clicking, typing,
        etc."""

        orb = self.client.gauges.orb

        self.msg.append(
            f'hp: {orb.hitpoints.value}, '
            f'prayer: {orb.prayer.value}, '
            f'run: {orb.run_energy.value}, '
            f'spec: {orb.special_attack.value}'
        )


def main():
    app = TestOrbsApp()
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
