from client import Application


class Slayer(Application):
    """
    Completes slayer tasks.
    No getting assignments (yet).
    No gearing up (yet).
    No banking (yet).
    Prayer flick enemy hits and piety for own attacks.
    """

    TAVERLY_HELLHOUNDS = {(44, 153, 20), (44, 154, 20)}

    MELEE_ON = 'protect_melee_active'
    MELEE_OFF = 'protect_melee'
    PRAYERS = [MELEE_ON, MELEE_OFF]

    def setup(self):
        """"""
        print('setting up')

        # set up the minimap
        mm = self.client.minimap.minimap
        # TODO: make this configurable
        mm.create_map(self.TAVERLY_HELLHOUNDS)
        mm.set_coordinates(37, 12, 44, 153, 20)
        mm.load_templates(['npc_tag'])
        mm.load_masks(['npc_tag'])

        # set up the prayer tab (it may not start active, so we'll locate the
        # icons during the event loop.
        # TODO: remove assumption that prayer tab is visible on startup
        i = self.client.tabs.prayer.interface
        i.load_templates(self.PRAYERS)
        i.load_masks(self.PRAYERS)

    def update(self):
        """"""

        self.client.game_screen.player.update()
        self.client.minimap.minimap.update()
        self.client.tabs.update()

        # TODO: design method to add tile base on construction
        for icon in self.client.minimap.minimap._icons.values():
            icon.tile_base = 2

    def prayer_icons_loaded(self):
        """
        Ensure we have the prayers icons loaded before we do other actions.
        :returns: True if prayer icons have been loaded
        """

        if self.client.tabs.prayer.interface.icons.get('melee') is None:
            if self.client.tabs.active_tab is self.client.tabs.prayer:
                self.client.tabs.prayer.interface.locate_icons({
                    'melee': {'templates': self.PRAYERS}
                })
                self.msg.append(
                    f'Loaded {len(self.client.tabs.prayer.interface.icons)} '
                    f'prayers')
            elif self.client.tabs.prayer.clicked:
                self.msg.append('Waiting prayer tab')
            else:
                self.client.tabs.prayer.click(tmin=0.1, tmax=0.2)
                self.msg.append('Clicked prayer tab')
            return False
        return True

    def action(self):
        """"""
        self.msg.append('action')

        # guard statement to check we have prayer icons loaded
        if not self.prayer_icons_loaded():
            return

        melee = self.client.tabs.prayer.interface.melee
        if self.client.tabs.active_tab is self.client.tabs.prayer:
            self.msg.append(
                f'Melee: {melee.state} {melee.confidence:.3f}')
        elif self.client.tabs.prayer.clicked:
            self.msg.append('Waiting prayer tab')
        else:
            self.client.tabs.prayer.click(tmin=0.1, tmax=0.2)
            self.msg.append('Clicked prayer tab')


def main():

    app = Slayer()
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
