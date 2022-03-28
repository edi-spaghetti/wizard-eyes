import argparse

from wizard_eyes.application import Application


class Lumberjack(Application):

    # template names
    TINDERBOX = 'tinderbox'
    NEST_RING = 'nest_ring'
    NEST_SEED = 'nest_seed'
    INVENTORY_TEMPLATES = [
        TINDERBOX, f'{TINDERBOX}_selected',
        NEST_RING,
        NEST_SEED,
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.msg_length = 200
        self.args = None
        self.inventory_templates = None
        self.logs = None
        self.trees = None
        self.items = None

    def parse_args(self):

        parser = argparse.ArgumentParser()
        parser.add_argument('--map-name', help='name of the map to load')

        # TODO: allow start xy by label
        parser.add_argument(
            '--start-xy', nargs=2, type=int,
            default=(133, 86),  # by the willow trees
            help='Specify starting coordinates'
        )

        parser.add_argument(
            '--tree-type', help='specify name of tree for templates etc.'
        )

        args, _ = parser.parse_known_args()
        return args

    def setup(self):
        """"""

        self.args = self.parse_args()

        mm = self.client.minimap.minimap
        gps = mm.gps
        inv = self.client.tabs.inventory

        # set up gps & map
        gps.load_map(self.args.map_name)
        start = tuple(self.args.start_xy)
        gps.set_coordinates(*start)

        # set up minimap templates & entities
        self.trees = dict()
        self.items = dict()
        mm.load_templates([self.args.tree_type, 'item'])
        # mm.load_masks([self.args.tree_type, 'item'])
        nodes = gps.current_map.find(
            label=f'{self.args.tree_type}[0-9]+', edges=False)
        for x, y in nodes:

            key = (int((x - self.args.start_xy[0]) * mm.tile_size),
                   int((y - self.args.start_xy[1]) * mm.tile_size))

            tree = self.client.game_screen.create_game_entity(
                self.args.tree_type, self.args.tree_type,
                key, self.client, self.client
            )
            self.trees[(x, y)] = tree

        # set up inventory templates
        self.logs = [f'{self.args.tree_type}_log',
                f'{self.args.tree_type}_log_selected']
        self.inventory_templates = self.INVENTORY_TEMPLATES + self.logs
        inv.interface.load_templates(self.inventory_templates)
        inv.interface.load_masks(self.inventory_templates)

    def update(self):
        """"""

        player = self.client.game_screen.player
        mm = self.client.minimap.minimap

        self.client.tabs.update()
        player.update()

        (x, y), matches = mm.update(threshold=0.95)

        # first update trees, which are static
        for name, (rx, ry) in matches:
            # convert to tile coordinate
            tx, ty = int(rx / mm.tile_size), int(ry / mm.tile_size)
            # convert relative xy to global
            gx, gy = x + tx, y + ty
            # self.msg.append(f'tree: {gx, gy}')

            if name == self.args.tree_type:
                tree = self.trees.get((gx, gy))
                if tree:
                    tree.key = rx, ry
                    tree.update()
                    # self.msg.append(str(tree))

        # next update items, which can be dropped / despawn
        items = [(name, (int(x * mm.tile_size), int(y * mm.tile_size)))
                 for name, (x, y) in matches if name in {'item'}]
        mm.generate_entities(
            items, entity_templates=[self.NEST_SEED, self.NEST_RING])

    def inventory_icons_loaded(self):
        """
        Ensure we have inventory icons loaded before we do other actions.
        :return: True if inventory icons have been loaded
        """

        inv = self.client.tabs.inventory

        if len(inv.interface.icons) < self.client.INVENTORY_SIZE:
            if self.client.tabs.active_tab is inv:
                # this will overwrite icons, so any click timeouts will be
                # lost, but that should be OK while we don't have a full
                # inventory because we won't need to click anything yet anyway.
                inv.interface.locate_icons({
                    'item': {
                        'templates': self.inventory_templates,
                        'quantity': self.client.INVENTORY_SIZE},
                })
                self.msg.append(f'Loaded {len(inv.interface.icons)} items')
            elif inv.clicked:
                self.msg.append('Waiting inventory tab')
                return False
            else:
                inv.click(tmin=0.1, tmax=0.2)
                self.msg.append('Clicked prayer tab')
                return False
        return True

    def action(self):
        """"""

        if not self.inventory_icons_loaded():
            return

        inv = self.client.tabs.inventory

        self.msg.append(
            f'Location: {self.client.minimap.minimap.gps.get_coordinates()} ')

        for state, count in inv.interface.state_count.items():
            self.msg.append(f'{state}={count}')

        # self.msg.append(f'Trees: {self.trees.keys()}')
        # self.msg.append(
        #     f'Items: {self.client.minimap.minimap._icons.values()}')


def main():
    app = Lumberjack()
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
