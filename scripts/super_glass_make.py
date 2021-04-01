import argparse
import sys
import time
import random

import keyboard

import client


if __name__ == '__main__':

    # setup
    print('Setting Up')
    c = client.Client('RuneLite')

    parser = argparse.ArgumentParser()

    parser.add_argument('-gsi', '--seaweed-index', type=int, required=True)
    parser.add_argument('-si', '--sand-index', type=int, required=True)
    parser.add_argument('-ti', '--tab-index', type=int, required=True)
    parser.add_argument('-b', '--bank-aoi', type=lambda x: tuple([int(y) for y in x]))

    args = parser.parse_args()

    bank_aoi = args.bank_aoi or c.screen.gen_bbox()

    # TODO: test bank fillers set to exclude astral runes
    # TODO: test withdraw x is set to 18

    # set up item names
    # TODO: create better way to manage names & variations
    seaweed = 'giant_seaweed'
    # seaweed_selected = f'{seaweed}_selected'
    seaweed_placeholder = f'{seaweed}_placeholder'
    sand = 'bucket_of_sand'
    # sand_selected = f'{sand}_selected'
    sand_placeholder = f'{sand}_placeholder'
    glass = 'molten_glass'
    # glass_selected = f'{glass}_selected'
    rune = 'astral_rune'

    # set up bank slots
    # TODO: check tab is open in-game
    tab = c.bank.tabs.set_tab(args.tab_index, is_open=True)
    seaweed_bank_slot = tab.set_slot(args.seaweed_index, [seaweed_placeholder])
    sand_bank_slot = tab.set_slot(args.sand_index, [sand_placeholder])

    # set up slots
    # TODO: variable threshold / template matching for stackable items
    c.inventory.set_slot(0, [rune])
    for i in range(1, 28):
        c.inventory.set_slot(i, [seaweed, sand, glass])

    # set up timeouts
    deposit = c.bank.utilities.deposit_inventory

    # logging
    msg_length = 50

    # main loop
    print('Entering Main Loop')
    c.activate()
    while True:

        # reset for logging
        sys.stdout.write('\b' * msg_length)
        msg = ''

        # caps lock to pause the script
        # p to exit
        # TODO: convert these to utility functions
        if not c.screen.on_off_state():
            msg += f'Sleeping @ {time.time()}'
            sys.stdout.write(f'{msg:50}')
            sys.stdout.flush()
            time.sleep(0.1)
            continue
        elif keyboard.is_pressed('p'):
            exit(1)

        # update
        t2 = time.time()
        img = c.screen.grab_screen(*c.get_bbox())
        inventory = c.inventory.identify(img)
        # TODO: identify current tab (should be inventory via bank or magic
        bank_open = c.bank.utilities.deposit_inventory.identify(img)

        t2 = time.time() - t2
        msg += f'Update {round(t2, 2)}'

        cool_down = 0.05

        # do something
        if bank_open:

            if glass in inventory:

                # TODO: probably easier to click a random glass instead
                if not deposit.clicked:
                    deposit.click()

                    msg += ' - Deposit'
                else:
                    msg += f' - Deposit ({deposit.time_left})'

            elif inventory.count(seaweed) < 3:

                seaweed_required = 3 - inventory.count(seaweed)
                if len(seaweed_bank_slot.clicked) < seaweed_required:
                    seaweed_bank_slot.click()

                    # TODO: variable cooldown between repeat clicks?

                    msg += f' - Withdraw {len(seaweed_bank_slot.clicked)} ' \
                           f'({seaweed_bank_slot.time_left})'

                else:

                    msg += f' - Wait Withdraw {seaweed} ' \
                           f'({seaweed_bank_slot.time_left})'

            # TODO: check for over-withdrawing seaweed
            # elif inventory.count(seaweed) > 3:

            elif inventory.count(sand) != 18:

                msg += f' - Withdraw {sand}'

            else:

                msg += f' - Close'

        else:

            msg += ' - Bank is Closed'

        sys.stdout.write(f'{msg[:msg_length]:50}')
        sys.stdout.flush()

        time.sleep(cool_down)