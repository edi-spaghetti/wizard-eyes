from wizard_eyes.application import Application


class MinimapTrackingApp(Application):
    """Demo application for tracking of minimap "dots"."""

    def __init__(self, *args, **kwargs):
        """Init an application attribute to hold our entities."""
        super().__init__(*args, **kwargs)
        self.objects = None

    def setup(self):
        """Set up the minimap to search for players, npcs, npcs tagged by the
        slayer plugin and npcs tagged manually. These templates are all loaded
        rom data, so ensure they have been saved. They *must* be colour
        templates, not black and white."""

        self.client.args.show = {'mouse', '*bbox', '*id', '*name'}

        mm = self.client.gauges.minimap

        mm.setup_thresolds('player', 'npc', 'npc-slayer', 'npc-tag')

    def update(self):
        """Update objects in the minimap, which includes updating existing
        objects, adding new and removing old ones. For reference, new objects
        are forced to show blue instead of default colour."""

        self.client.gauges.update()
        self.client.game_screen.update()

        mm = self.client.gauges.minimap
        self.objects = mm.track(self.objects)

        for o in self.objects:
            if self.client.time - o.state_changed_at < 3:
                o.colour = (255, 0, 0, 255)
            else:
                o.colour = o.DEFAULT_COLOUR

    def action(self):
        """Log the number of objects found live and in the recycler."""

        self.msg.append(f'Live Objects: {len(self.objects)}')

        buffered_msg = 'Buffered: '
        for key, data in self.client.game_screen.buffer.items():
            buffered_msg += f'{key}: {len(data)}, '
        self.msg.append(buffered_msg[:-2])


def main():
    app = MinimapTrackingApp()
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
