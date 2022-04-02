import argparse

from wizard_eyes.application import Application
from wizard_eyes.constants import REDA, DARK_REDA


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

    # states
    WOODCUTTING = 1
    FIRE_MAKING = 2
    BANKING = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.msg_length = 200
        self.args = None
        self.inventory_templates = None
        self.logs = None
        self.log = None
        self.log_selected = None
        self.trees = None
        self.items = None
        self.state = None
        self.target_tree = None
        self.target_item = None
        self.new_log_this_frame = None
        self.new_nest_this_frame = None
        self.newest_log_at = None
        self.newest_nest_at = None
        self.num_icons_loaded = 0
        # longest I've seen a log be received is about 14 seconds,
        # TODO: tweak for other logs, player level etc.
        self.log_timeout = 14

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
            tree.set_global_coordinates(x, y)
            self.trees[(x, y)] = tree

        # set up inventory templates
        self.log = f'{self.args.tree_type}_log'
        self.log_selected = f'{self.args.tree_type}_log_selected'
        self.logs = [self.log, self.log_selected]
        self.inventory_templates = self.INVENTORY_TEMPLATES + self.logs
        inv.interface.load_templates(self.inventory_templates)
        inv.interface.load_masks(self.inventory_templates)

    def update(self):
        """"""

        player = self.client.game_screen.player
        mm = self.client.minimap.minimap
        gps = mm.gps
        inv = self.client.tabs.inventory

        self.client.tabs.update()
        player.update()

        # get the old coordinates, we may have to set them back if the gps is
        # on the fritz
        ox, oy = gps.get_coordinates()
        (x, y), matches = mm.update(threshold=0.95)

        # first update trees, which are static
        for tree in self.trees.values():
            tree.refresh()

        for name, (rx, ry) in matches:
            # convert to tile coordinate
            tx, ty = int(rx / mm.tile_size), int(ry / mm.tile_size)
            # convert relative xy to global
            gx, gy = x + tx, y + ty
            # self.msg.append(f'tree: {gx, gy}')

            if name == self.args.tree_type:
                tree = self.trees.get((gx, gy))
                if tree:
                    tree.update(key=(rx, ry))
                    # self.msg.append(str(tree))
                else:
                    # the gps is wrong, because trees can't move. set it back!
                    gps.set_coordinates(ox, oy)

        # update any trees that we couldn't find by relative position without
        # setting their key
        for tree in self.trees.values():
            if not tree.checked:
                tree.update()

        # next update items, which can be dropped / despawn
        items = [(name, (int(x * mm.tile_size), int(y * mm.tile_size)))
                 for name, (x, y) in matches if name in {'item'}]
        mm.generate_entities(
            items, entity_templates=[self.NEST_SEED, self.NEST_RING])

        # update state
        fm_condition = (
            # inventory is full
            (inv.interface.icon_count == self.client.INVENTORY_SIZE
             # we're not full on something else (like nests)
             and inv.interface.sum_states(self.log, self.log_selected) > 0
             # we don't have any empty spaces
             and inv.interface.sum_states(None) == 0
             # and we're still in woodcutting mode
             and self.state == self.WOODCUTTING)
            # or there's still logs left and we're in firemaking mode
            or (inv.interface.sum_states(self.log, self.log_selected) > 0
                and self.state == self.FIRE_MAKING)
            # TODO: make sure we don't abandon the last log before lighting it
        )
        b_condition = (
            # TODO: make the threshold for banking dynamic
            inv.interface.sum_states(self.NEST_SEED, self.NEST_RING) > 12
            # if no trees are on screen, we need to get closer
            or not any([t.is_on_screen for t in self.trees.values()])
        )

        if fm_condition:
            self.state = self.FIRE_MAKING
        elif b_condition:
            self.state = self.BANKING
        else:
            self.state = self.WOODCUTTING

        # run state-specific updates - they should have guard statements
        self.woodcutting_update()

    def woodcutting_update(self):
        """Update routines specific to when we're in woodcutting mode."""

        if self.state != self.WOODCUTTING:
            return

        inv = self.client.tabs.inventory
        mm = self.client.minimap.minimap
        gps = self.client.minimap.minimap.gps

        # check if we have logs or nests coming in recently
        prev_new_log = self.newest_log_at or -float('inf')
        prev_new_nest = self.newest_nest_at or -float('inf')
        self.newest_log_at = -float('inf')
        self.newest_nest_at = -float('inf')
        self.new_log_this_frame = False
        self.new_nest_this_frame = False
        for item in inv.interface.icons.values():
            is_log = item.state == self.log
            newer_log = item.state_changed_at > self.newest_log_at
            is_nest = item.state in {self.NEST_SEED, self.NEST_RING}
            newer_nest = item.state_changed_at > self.newest_nest_at
            if is_log and newer_log:
                self.newest_log_at = item.state_changed_at
                if item.state_changed_at > prev_new_log:
                    self.new_log_this_frame = True
            if is_nest and newer_nest:
                self.newest_nest_at = item.state_changed_at
                if item.state_changed_at > prev_new_nest:
                    self.new_nest_this_frame = True

        # self.msg.append(f'diff: {self.client.time - self.newest_log_at:.3f}')

        # set a target tree
        target_condition = (
            # if we don't already have one
            self.target_tree is None
            # or it has been cut down
            or self.target_tree.state is not None
        )
        if target_condition:

            if self.target_tree is not None:
                self.target_tree.colour = self.target_tree.DEFAULT_COLOUR

            # find the nearest choppable tree
            nearest_tree = None
            cur_distance = float('inf')
            for xy, tree in self.trees.items():
                distance = mm.distance_between(gps.get_coordinates(), xy)
                if distance < cur_distance and tree.state is None:
                    nearest_tree = tree
                    cur_distance = distance

            self.target_tree = nearest_tree
            self.target_tree.colour = REDA

        # set a target nest (if there is one)
        nests = [item for item in mm._icons.values()
                 if item.state in {self.NEST_RING, self.NEST_SEED}]
        # check if we already have a target
        if self.target_item is None:
            if nests:
                self.target_item = nests[0]
                x, y = gps.get_coordinates()
                tx, ty = self.target_item.key
                tx, ty = int(tx / mm.tile_size) + x, int(ty / mm.tile_size) + y
                self.target_item.set_global_coordinates(tx, ty)
                self.target_item.colour = DARK_REDA
        else:
            if not self.target_item.checked:
                self.target_item.update()
            # check if that item has expired (either despawned, or we
            # picked it up)
            if self.target_item.state is None:
                self.target_item = None

    def inventory_icons_loaded(self):
        """
        Ensure we have inventory icons loaded before we do other actions.
        :return: True if inventory icons have been loaded
        """

        inv = self.client.tabs.inventory

        if len(inv.interface.icons) < self.client.INVENTORY_SIZE:
            if self.client.tabs.active_tab is inv:
                inv.interface.locate_icons({
                    'item': {
                        'templates': self.inventory_templates,
                        'quantity': self.client.INVENTORY_SIZE},
                })
                # self.msg.append(f'Loaded {len(inv.interface.icons)} items')
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

        actions = {
            self.WOODCUTTING: self.do_woodcutting,
            self.FIRE_MAKING: self.do_firemaking,
            self.BANKING: self.do_banking,
        }

        action = actions.get(self.state)
        if action:
            action()

    def do_woodcutting(self):
        """Cut some wood."""

        mm = self.client.minimap.minimap
        gps = self.client.minimap.minimap.gps

        # all trees have been chopped down, just wait for them to regrow.
        if self.target_tree is None:
            self.msg.append('Waiting for valid target tree')
            return

        # if there's a nest to go pick up, go get that nest!
        if self.target_item is not None:
            pxy = gps.get_coordinates()
            txy = self.target_item.get_global_coordinates()
            # TODO: calculate route with tile path
            est_time_to_item = mm.distance_between(pxy, txy) * 1.5
            if self.target_item.clicked:
                if self.new_nest_this_frame:
                    self.msg.append(f'Picked up {self.target_item}')
                    self.target_item = None
                else:
                    self.msg.append(f'Waiting to arrive at {self.target_item}')
            else:
                self.target_item.click(tmin=est_time_to_item,
                                       tmax=est_time_to_item + 3,
                                       pause_before_click=True)
                self.msg.append(f'Clicked {self.target_item}')
            return

        # otherwise it's tree chopping time!
        if self.target_tree.in_base_contact(*gps.get_coordinates()):
            if self.target_tree.clicked:
                if self.new_log_this_frame:
                    self.target_tree.add_timeout(self.log_timeout)
                    self.msg.append(f'{self.target_tree.clicked}')
                    self.msg.append(
                        f'Got logs ... {self.target_tree.time_left:.3f}')
                else:
                    self.msg.append(
                        f'Chopping ... {self.target_tree.time_left:.3f}')
            else:
                self.msg.append(f'{self.target_tree.clicked}')
                self.target_tree.click(
                    tmin=self.log_timeout, tmax=self.log_timeout * 1.5,
                    pause_before_click=True)
                self.msg.append(f'Clicked {self.target_tree}')
        else:
            if self.target_tree.clicked:
                self.msg.append(f'Waiting to arrive at {self.target_tree}')
            else:
                # assume it's a straight shot to the tree, and give ourselves
                # a 50% buffer
                # TODO: calculate route with tile path
                pxy = gps.get_coordinates()
                txy = self.target_tree.get_global_coordinates()
                est_time_to_tree = mm.distance_between(pxy, txy) * 1.5
                self.target_tree.click(tmin=est_time_to_tree,
                                       tmax=est_time_to_tree + 3,
                                       pause_before_click=True)

    def do_firemaking(self):
        """Burn it all."""
        self.msg.append('Fire making mode.')

    def do_banking(self):
        """Bank that loot."""
        self.msg.append('Banking mode.')


def main():
    app = Lumberjack()
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
