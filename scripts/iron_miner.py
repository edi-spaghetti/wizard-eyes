import sys
import time
import re
import argparse
import random

import client
from game_objects import GameObject
from script_utils import safety_catch, logout


def bbox(value):
    try:
        values = value.split(' ')
        ret_values = list()
        for value in values:
            coords = value.split(',')
            ret_coords = list()
            for coord in coords:
                ret_coords.append(int(coord))
            ret_coords = tuple(ret_coords)
            assert len(ret_coords) == 4
            ret_values.append(ret_coords)

        return ret_values
    except (ValueError, AssertionError):
        raise argparse.ArgumentError('AOI must be 4 comma separated ints')


def first_unclicked(client_, item, regex=False):

    if regex:
        method = re.compile(item).match
    else:
        method = item.__eq__

    for slot in client_.inventory.slots:
        # slot contents could be None
        name_str = str(slot.contents)
        if method(name_str) and not slot.clicked:
            return slot


def select_chisel(c, names, msg):
    slots = c.inventory.filter_slots(names)
    if slots and not any([s.clicked for s in slots]):
        slot = first_unclicked(c, 'chisel')
        if slot:
            slot.click(tmin=0.6, tmax=0.9)
            msg.append(f'Click chisel in slot {slot.idx}')
        else:
            msg.append(f'Waiting chisel selection')
    else:
        msg.append('Waiting gem cut')


def main():

    # setup
    print('Setting Up')
    c = client.Client('RuneLite')

    # TODO: set up argparser
    parser = argparse.ArgumentParser()

    parser.add_argument('--rock-aoi', type=bbox)
    parser.add_argument('--time-limit', type=int, default=float('inf'),
                        help='time for script to run in hours')
    args = parser.parse_args()

    rocks = list()
    for i in range(3):
        if args.rock_aoi:
            rock_aoi = args.rock_aoi[i]
        else:
            rock_aoi = c.screen.gen_bbox()
        rock = GameObject(c, c)
        rock.set_aoi(*rock_aoi)
        rocks.append(rock)

    aoi_params = list()
    for rock in rocks:
        aoi_params.append(','.join([str(c) for c in rock.get_bbox()]))
    print_params = ' '.join(aoi_params)
    print(f'--rock-aoi "{print_params}"')

    # set up item names
    iron = 'iron_ore'
    gems = {'sapphire', 'emerald', 'ruby', 'diamond'}
    uncut_gems = {f'uncut_{g}' for g in gems}
    cut_gems = {f'cut_{g}' for g in gems}
    chisel = 'chisel'
    chisel_selected = 'chisel_selected'

    # setup inventory slots
    items = list(uncut_gems.union(cut_gems)) + [chisel, chisel_selected, iron]
    for i in range(28):
        c.inventory.set_slot(i, items)

    # tracking variables
    waiting_gems = False

    msg_length = 100
    t3 = time.time()
    duration = 60 * 60 * args.time_limit + (random.random() * 60 * 20)

    # main loop
    print('Entering Main Loop')
    c.activate()
    while True:

        # reset for logging
        t1 = time.time()
        sys.stdout.write('\b' * msg_length)
        msg = list()

        if safety_catch(c, msg_length):
            continue
        if time.time() - t3 > duration:
            logout(c)
            break

        # update
        img = c.screen.grab_screen(*c.get_bbox())
        inventory = c.inventory.identify(img)
        for i in range(len(inventory)):
            c.inventory.slots[i].update()
        for i in range(3):
            rocks[i].update()

        t1 = time.time() - t1
        msg.append(f'Update {t1:.2f}')

        t2 = time.time()
        cool_down = 0.05

        # action
        if iron in inventory:

            slot = first_unclicked(c, iron)
            if slot:
                slot.click(tmin=0.6, tmax=0.9, shift=True)
                msg.append(f'Drop Iron in position {slot.idx}')
            else:
                msg.append(f'Drop Iron (TODO: time left)')

        elif c.inventory.contains(uncut_gems):

            if c.inventory.contains([chisel_selected]):

                slot = first_unclicked(c, 'uncut_.*', regex=True)
                if slot:
                    slot.click(tmin=1.8, tmax=2.4)
                    msg.append(f'Cutting {slot.contents} in position {slot.idx}')
                else:
                    msg.append('Waiting gem cut')

            else:
                # select the chisel if a gem hasn't already been clicked
                select_chisel(c, uncut_gems, msg)

        elif c.inventory.contains(cut_gems):

            if c.inventory.contains([chisel_selected]):
                slot = first_unclicked(c, 'cut_.*', regex=True)
                if slot:
                    slot.click(tmin=5.4, tmax=6)
                    msg.append(f'Fletching {slot.contents} bolt tips '
                               f'in position {slot.idx}')
                else:
                    msg.append('Waiting gem cut')
            else:
                # select the chisel if a gem hasn't already been clicked
                select_chisel(c, cut_gems, msg)

        else:

            available_rocks = list(filter(lambda r: not r.clicked, rocks))
            if available_rocks:
                available_rocks[0].click(tmin=6.6, tmax=7.2, pause_before_click=True)
                msg.append(f'Clicked Iron')
                cool_down = 1.8
            else:
                msg.append(f'Waiting Iron')

        t2 = time.time() - t2
        msg.insert(1, f'Action {t2:.2f}')
        msg.insert(2, f'Loop {time.time() - t3:.2f}')
        msg = ' - '.join(msg)

        sys.stdout.write(f'{msg[:msg_length]:{msg_length}}')
        sys.stdout.flush()

        time.sleep(cool_down)


if __name__ == '__main__':
    main()
