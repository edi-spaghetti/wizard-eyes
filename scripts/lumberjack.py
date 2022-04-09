import argparse
import random
import math

import pyautogui
import numpy

from wizard_eyes.application import Application
from wizard_eyes.constants import REDA, DARK_REDA


class Lumberjack(Application):

    # template names
    TINDERBOX = 'tinderbox'
    TINDERBOX_S = f'{TINDERBOX}_selected'
    NEST_RING = 'nest_ring'
    NEST_SEED = 'nest_seed'
    NEST_EGG_BLUE = 'nest_egg_blue'
    NEST_EGG_RED = 'nest_egg_red'
    NESTS = (NEST_RING, NEST_SEED, NEST_EGG_BLUE, NEST_EGG_RED)
    INVENTORY_TEMPLATES = [
        TINDERBOX, TINDERBOX_S,
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
        self.fire_lanes = None
        self.target_fire_lane = None
        self.target_fire = None
        self.tinderbox = None
        self.target_log = None
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
        nodes = gps.current_map.find(label=self.args.tree_type)
        for x, y in nodes:

            key = (int((x - self.args.start_xy[0]) * mm.tile_size),
                   int((y - self.args.start_xy[1]) * mm.tile_size))

            tree = self.client.game_screen.create_game_entity(
                self.args.tree_type, self.args.tree_type,
                key, self.client, self.client
            )
            tree.set_global_coordinates(x, y)
            self.trees[(x, y)] = tree

        # set up fire lanes
        self.fire_lanes = dict()
        lane_starts = gps.current_map.find(label='fire_lane')
        fire_colour = gps.current_map.label_colour('fire_lane')
        for x, y in lane_starts:

            fires = list()
            vectors = set()

            i = 0
            while (
                # keep going west until we hit a blocked tile
                'impassable' not in gps.current_map.node_to_label((x - i, y))
                # safety catch in case we missed a blocker label
                and i < 28
                # TODO: check we're not going off the edge of the map
            ):

                # determine relative position to player
                key = (int((x - self.args.start_xy[0]) * mm.tile_size),
                       int((y - self.args.start_xy[1]) * mm.tile_size))

                # generate a new entity
                fire = self.client.game_screen.create_game_entity(
                    'fire', 'fire', key, self.client, self.client,
                    # entity_templates=['fire'],
                )
                fire.set_global_coordinates(x - i, y)
                fire.colour = fire_colour
                # fire.state_threshold = 0.7

                # add them to book keeping variables
                fires.append(fire)
                vectors.add((x - i, y))

                # iterate counter so we move one tile west
                i += 1

            self.fire_lanes[(x, y)] = dict(entities=fires, vectors=vectors)

        # set up inventory templates
        self.log = f'{self.args.tree_type}_log'
        self.log_selected = f'{self.args.tree_type}_log_selected'
        self.logs = [self.log, self.log_selected]
        self.inventory_templates = self.INVENTORY_TEMPLATES + self.logs
        self.inventory_templates.extend(self.NESTS)
        inv.interface.load_templates(self.inventory_templates)
        inv.interface.load_masks(self.inventory_templates)

        # set up icon tracker on inventory interface for items we're expecting
        # to be added/removed from the inventory
        inv.interface.add_icon_tracker_grouping((None, ))
        inv.interface.add_icon_tracker_grouping((self.log, ))
        inv.interface.add_icon_tracker_grouping(self.NESTS)

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
        (x, y), matches = mm.update(threshold=0.95, auto_gps=False)

        # first update trees, which are static
        for tree in self.trees.values():
            tree.refresh()

        accept_gps = False
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
                    accept_gps = True
                    # self.msg.append(str(tree))
                else:
                    # the gps is wrong, because trees can't move. set it back!
                    accept_gps = False

        # based off static entities, we can determine if we should update gps
        if accept_gps:
            gps.set_coordinates(x, y)
        else:
            gps.set_coordinates(ox, oy)

        # update any trees that we couldn't find by relative position without
        # setting their key
        for tree in self.trees.values():
            if not tree.checked:
                tree.update()

        # now update fire spots, which are also static
        for _, data in self.fire_lanes.items():
            fires = data['entities']
            for fire in fires:
                fx, fy = fire.get_global_coordinates()
                px, py = gps.get_coordinates()
                # update the entity key, which is the relative position to
                # the player in pixels
                fire.key = (fx - px) * mm.tile_size, (fy - py) * mm.tile_size
                # fire.update_state()
                fire.update()
                # assume the fire has gone out on timer
                if not fire.time_left:
                    fire.state = None
                if fire != self.target_fire:
                    fire_colour = gps.current_map.label_colour('fire_lane')
                    fire.colour = fire_colour

        # next update items, which can be dropped / despawn
        items = [(name, (int(x * mm.tile_size), int(y * mm.tile_size)))
                 for name, (x, y) in matches if name in {'item'}]
        mm.generate_entities(
            items, entity_templates=list(self.NESTS))

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
            # make sure we don't abandon the last log before lighting it
            or (self.target_log is not None and self.state == self.FIRE_MAKING)
        )
        b_condition = (
            # TODO: make the threshold for banking dynamic
            inv.interface.sum_states(*self.NESTS) > 12
            # if no trees are on screen, we need to get closer
            or not any([t.is_on_screen for t in self.trees.values()])
        )

        if fm_condition:
            self.state = self.FIRE_MAKING
        elif b_condition:
            self.state = self.BANKING
        else:
            self.state = self.WOODCUTTING
            self.target_fire = None
            self.target_fire_lane = None
            self.target_log = None

        # run state-specific updates - they should have guard statements
        self.woodcutting_update()
        self.firemaking_update()

    def firemaking_update(self):
        """Updates specific to when we're making fires."""

        if self.state != self.FIRE_MAKING:
            return

        gps = self.client.minimap.minimap.gps
        mm = self.client.minimap.minimap

        if self.target_fire is None:
            # find a new fire lane and start from the most westerly one
            candidates = list()
            for xy in self.fire_lanes:
                entities = self.fire_lanes[xy]['entities']
                on_fire = any([e.state == 'fire' for e in entities])
                if not on_fire:
                    candidates.append(xy)

            # choose a random start position from the viable candidates,
            # weighted by distance to the player
            pxy = gps.get_coordinates()
            distances = list()
            for candidate in candidates:
                # get distance, but ensure > 0 or we get zero division error
                distance = mm.distance_between(candidate, pxy) or 0.01
                distances.append(distance)

            txy = weighted_random(candidates, distances)
            self.target_fire = self.fire_lanes[txy]['entities'][0]
            self.target_fire.colour = REDA
            self.target_fire_lane = txy

        else:
            if self.target_fire.state == 'fire':
                # we've already lit a fire in the current spot, see where the
                # next one should be
                fire_colour = gps.current_map.label_colour('fire_lane')
                x, y = self.target_fire.get_global_coordinates()
                if 'impassable' in gps.current_map.node_to_label((x - 1, y)):
                    # we've reached the end of the line
                    self.target_fire.colour = fire_colour
                    self.target_fire = None
                    self.firemaking_update()
                else:
                    fires = self.fire_lanes[self.target_fire_lane]['entities']
                    index = fires.index(self.target_fire)
                    # swap the old target colour back
                    self.target_fire = fire_colour
                    self.target_fire = fires[index + 1]
                    # mark the new target with different colour
                    self.target_fire.colour = REDA
            # if target is not in fire state, it means we're either walking
            # to it, or in the process of going to it.

    def woodcutting_update(self):
        """Update routines specific to when we're in woodcutting mode."""

        if self.state != self.WOODCUTTING:
            return

        # inv = self.client.tabs.inventory
        mm = self.client.minimap.minimap
        gps = self.client.minimap.minimap.gps

        # new_log = inv.interface.icon_tracker.get_grouping(
        #     (self.log,)).get('new_this_frame')
        # new_nest = inv.interface.icon_tracker.get_grouping(
        #     self.NESTS).get('new_this_frame')
        # new_empty = inv.interface.icon_tracker.get_grouping(
        #     (None, )).get('new_this_frame')
        #
        # self.msg.append(f'New:: '
        #                 f'log: {new_log}, '
        #                 f'nest: {new_nest}, '
        #                 f'empty: {new_empty}')

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
            # sometimes it isn't able to find a new nearest tree,
            # so check first
            if self.target_tree is not None:
                self.target_tree.colour = REDA

        # set a target nest (if there is one)
        nests = [item for item in mm._icons.values()
                 if item.state in self.NESTS]
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

                # if we found a tinderbox, keep track of for later
                icons = inv.interface.icons_by_state(
                    'tinderbox', 'tinderbox_selected')
                if icons:
                    self.tinderbox = icons[0]

                if (len(inv.interface.icons) == self.client.INVENTORY_SIZE
                        and self.state == self.WOODCUTTING):
                    self.msg.append('Fixing state')
                    return False

            elif inv.clicked:
                self.msg.append('Waiting inventory tab')
                return False
            else:
                inv.click(tmin=0.1, tmax=0.2)
                self.msg.append('Clicked inventory tab')
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

        inv = self.client.tabs.inventory
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
                new_nest = inv.interface.icon_tracker.get_grouping(self.NESTS)

                if new_nest.get('new_this_frame'):
                    self.msg.append(f'Picked up {self.target_item}')
                    self.target_item = None
                else:
                    self.msg.append(f'Waiting to arrive at {self.target_item}')
            else:
                if self.target_item.state in self.NESTS:
                    self.target_item.click(tmin=est_time_to_item,
                                           tmax=est_time_to_item + 3,
                                           pause_before_click=True)
                    self.msg.append(f'Clicked {self.target_item}')
                else:
                    self.msg.append(f'Target {self.target_item} lost')
                    self.target_item = None
            return

        # otherwise it's tree chopping time!
        if self.target_tree.in_base_contact(*gps.get_coordinates()):
            if self.target_tree.clicked:
                new_log = inv.interface.icon_tracker.get_grouping(
                    (self.log, ))

                if new_log.get('new_this_frame'):
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

        mm = self.client.minimap.minimap
        gps = self.client.minimap.minimap.gps
        inv = self.client.tabs.inventory

        pxy = gps.get_coordinates()
        txy = self.target_fire.get_global_coordinates()

        if txy != pxy and self.target_log is None:
            if self.target_fire.clicked:
                self.msg.append(
                    f'Waiting to arrive at {txy} {self.target_fire.time_left}')
            else:
                # TODO: ensure we don't accidentally click on an NPC on the
                #       tile by checking the mouse-over text
                est_time_to_tile = mm.distance_between(pxy, txy) * 1.5
                if self.target_fire.is_on_screen:
                    self.target_fire.click(
                        tmin=est_time_to_tile, tmax=est_time_to_tile + 3
                    )
                    self.msg.append(
                        f'Clicked tile at {txy}')
                else:
                    self.target_fire.click(
                        tmin=est_time_to_tile, tmax=est_time_to_tile + 3,
                        bbox=self.target_fire.mm_bbox()
                    )
        else:
            click_tinderbox = (self.tinderbox.state != self.TINDERBOX_S
                               and self.target_log is None)
            if click_tinderbox:
                if self.tinderbox.clicked:
                    self.msg.append(
                        f'Waiting tinderbox: {self.tinderbox.time_left}')
                else:
                    self.tinderbox.click(tmin=0.3, tmax=0.6)
                    self.msg.append('Clicked tinderbox')
            elif self.target_log is None:

                # pick a target log and click it
                candidates = inv.interface.icons_by_state(self.log)
                mouse = pyautogui.position()
                mx, my = mouse.x, mouse.y
                distances = list()
                for candidate in candidates:
                    cx, cy = self.client.screen.distribute_normally(
                        *candidate.get_bbox())
                    distance = math.sqrt((cx - mx)**2 + (cy - my)**2) or 0.01
                    distances.append(distance)

                self.target_log = weighted_random(candidates, distances)
                self.target_log.click(tmin=3, tmax=5)  # TODO: tweak these
                self.msg.append(f'Clicked {self.log} ({self.target_log.name})')
            else:

                log_timeout = (not self.target_log.clicked
                               and self.target_log.state is None)

                if log_timeout:
                    self.msg.append(
                        f'Timeout {self.target_log.name}, choose new log')
                    self.target_log = None

                elif self.target_log.state is None:

                    if txy != pxy:
                        # we should either be west or east of our fire, which
                        # means the player auto-walked afer making a fire
                        self.target_fire.state = 'fire'
                        # according to wiki fires can last up to two minutes
                        self.target_fire.add_timeout(2 * 60)

                        # reset the target log so a new one can be picked up
                        # on the next round
                        self.target_log.clear_timeout()
                        self.target_log = None
                        self.msg.append('FIRE!')
                    else:
                        self.msg.append('Making fire')
                else:
                    self.msg.append(
                        f'Waiting log at '
                        f'{self.target_log.name} {self.target_log.time_left}')

    def do_banking(self):
        """Bank that loot."""
        self.msg.append('Banking mode.')


def weighted_random(candidates, distances):
    """
    Pick a random item from a list of candidates, weighted by distance.
    Indexes of candidates and distances must exactly match.
    """
    inverse = [1 / d for d in distances]
    normalised = [i / sum(inverse) for i in inverse]
    cum_sum = numpy.cumsum(normalised)
    r = random.random()
    for i, val in enumerate(cum_sum):
        if val > r:
            return candidates[i]



def main():
    app = Lumberjack()
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
