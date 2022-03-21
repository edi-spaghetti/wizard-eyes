from client import Application
from game_screen import NPC


class Hellhound(NPC):

    def __repr__(self):
        return self.as_string

    def __str__(self):
        return self.as_string

    def __init__(self, *args, **kwargs):
        super(Hellhound, self).__init__(*args, **kwargs)
        self.tile_base = 2
        self.as_string = f'Hellhound<{self.id[:8]}>'


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

    RED = (0, 0, 255, 255)

    def __init__(self, client='RuneLite', msg_length=100):
        super(Slayer, self).__init__(client=client, msg_length=msg_length)

        # create a variable for current target
        self.target = None

    def setup(self):
        """"""
        print('setting up')

        # set up the minimap
        mm = self.client.minimap.minimap
        # TODO: make this configurable
        mm.create_map(self.TAVERLY_HELLHOUNDS)
        mm.set_coordinates(37, 12, 44, 153, 20)
        mm.load_templates(['npc', 'npc_tag'])
        mm.load_masks(['npc', 'npc_tag'])

        # set up the prayer tab (it may not start active, so we'll locate the
        # icons during the event loop.
        # TODO: remove assumption that prayer tab is visible on startup
        i = self.client.tabs.prayer.interface
        i.load_templates(self.PRAYERS)
        i.load_masks(self.PRAYERS)

        # setup our custom NPC
        self.client.game_screen.default_npc = Hellhound

    def update(self):
        """"""

        self.client.game_screen.player.update()
        self.client.minimap.minimap.update()
        self.client.tabs.update()

        # TODO: improve target resetting
        if self.target not in self.client.minimap.minimap._icons.values():
            if self.target is not None:
                self.target.colour = NPC.DEFAULT_COLOUR
                self.msg.append(f'Target lost (1): {self.target}')
            self.target = None

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

    def click_target(self):

        mm = self.client.minimap.minimap
        interface = self.client.tabs.prayer.interface
        dist = int(self.target.distance_from_player - 1)
        tmin = dist * self.client.TICK
        tmax = tmin + 1.2

        try:
            x, y = self.target.get_hitbox()
            if (self.client.is_inside(x, y)
                    and not mm.is_inside(x, y)
                    # TODO: dialog switchboard (buttons on bottom left)
                    and not self.client.tabs.is_inside(x, y)
                    and not interface.is_inside(x, y)):

                x, y = self.target.click(
                    tmin=tmin, tmax=tmax, bbox=(x, y, x, y),
                    pause_before_click=True)
                self.msg.append(f'Clicked {x, y}: {self.target}')
            else:
                raise TypeError
        except TypeError:
            x, y = self.target.click(
                tmin=tmin, tmax=tmax, bbox=self.target.mm_bbox(),
            )
            self.msg.append(f'Clicked (MM) {x, y}: {self.target}')

    def action(self):
        """"""

        # guard statement to check we have prayer icons loaded
        if not self.prayer_icons_loaded():
            return

        player = self.client.game_screen.player
        mm = self.client.minimap.minimap
        melee = self.client.tabs.prayer.interface.melee

        if player.combat_status == player.NOT_IN_COMBAT or self.target is None:

            # if we're not in combat, make sure we turn off prayer before we do
            # something else or we'll waste it
            if melee.state == self.MELEE_ON:
                x, y = melee.click(tmin=0.1, tmax=0.2)
                self.msg.append(f'Clicked melee: {x, y}')
                return

            if self.target is None:
                # TODO: filtering and sorting methods for entities
                npcs = sorted(
                    [i for i in mm._icons.values()
                     if i.name == 'npc_tag'],
                    key=lambda i: i.distance_from_player)
                target = npcs[0]

                # TODO: checks and balances to confirm target
                #       e.g. we may be under aggression timer and get PJed
                #       e.g. another player may get there first
                #       e.g. the click may miss
                self.target = target
                self.target.colour = self.RED

            # TODO: determine if target on screen

            if self.target.clicked:
                self.msg.append(
                    f'Clicked (1): {self.target} {self.target.time_left}')
            else:
                self.click_target()

        elif player.combat_status == player.LOCAL_ATTACK:

            if (self.target.combat_status == Hellhound.NOT_IN_COMBAT
                    and not self.target.clicked):

                if self.target.clicked:
                    self.msg.append(
                        f'Clicked (2): {self.target} {self.target.time_left}')
                else:
                    self.click_target()
                return

            time_since = self.client.time - player.combat_status_updated_at
            prayer_on_threshold = self.target.attack_time - 0.4
            next_flick_at = (player.combat_status_updated_at
                             + prayer_on_threshold)
            # prayer_off_threshold = self.target.attack_time + 0.2
            # TODO: offsets for ranged/mage attacks

            if time_since < prayer_on_threshold:
                # check if prayer is on, and turn off if necessary
                if melee.state == self.MELEE_ON:
                    if melee.time_left:
                        self.msg.append(
                            f'Melee ON: {melee.time_left:.3f}')
                    else:
                        x, y = melee.click(tmin=0.1, tmax=0.2)
                        self.msg.append(f'Clicked melee: {x, y}')
                else:
                    self.msg.append(
                        f'Waiting next flick: '
                        f'{next_flick_at - self.client.time:.3f}')
            elif prayer_on_threshold < time_since:
                if melee.state == self.MELEE_OFF:
                    if melee.time_left:
                        self.msg.append(
                            f'Waiting melee: {melee.time_left:.3f}')
                    else:
                        # slightly longer timeout so we can cross the tick
                        # threshold, and successfully prayer flick
                        x, y = melee.click(tmin=0.3, tmax=0.4)
                        self.msg.append(f'Clicked melee: {x, y}')
                else:
                    self.msg.append(f'Melee ON: {melee.time_left:.3f}')

        else:
            self.msg.append(f'Combat: {player.combat_status}')


def main():

    app = Slayer()
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
