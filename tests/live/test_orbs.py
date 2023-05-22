from wizard_eyes.application import Application


class TestOrbsApp(Application):

    def setup(self):
        """"""

    def update(self):
        """"""

        self.client.update()
        self.client.minimap.orb.prayer.update()
        self.client.minimap.orb.hitpoints.update()

    def action(self):
        """"""

        prayer = self.client.minimap.orb.prayer
        hp = self.client.minimap.orb.hitpoints

        self.msg.append(f'prayer: {prayer.state}, hitpoints: {hp.state}')


def main():
    app = TestOrbsApp()
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
