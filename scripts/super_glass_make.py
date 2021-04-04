import argparse
import sys
import time
import random

import keyboard

import client
from game_objects import Magic, GameObject


if __name__ == '__main__':

    # setup
    print('Setting Up')
    c = client.Client('RuneLite')

    parser = argparse.ArgumentParser()

    parser.add_argument('-gsi', '--seaweed-index', type=int, required=True)
    parser.add_argument('-swc', '--seaweed-count', type=int, default=3)
    parser.add_argument('-si', '--sand-index', type=int, required=True)
    parser.add_argument('-scw', '--sand-context-width', type=int, required=True)
    parser.add_argument('-sci', '--sand-context-items', type=int, required=True)
    parser.add_argument('-sc', '--sand-count', type=int, default=18)
    parser.add_argument('-ti', '--tab-index', type=int, required=True)
    parser.add_argument('-b', '--bank-aoi', type=lambda x: tuple([int(y) for y in x]))

    args = parser.parse_args()

    bank_aoi = args.bank_aoi or c.screen.gen_bbox()
    bank = GameObject(c, c)
    bank.set_aoi(*bank_aoi)

    # TODO: test bank fillers set to exclude astral runes
    # TODO: test withdraw x is set to 18
    withdraw_x_index = 3

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
    for i in range(28):
        c.inventory.set_slot(i, [rune, seaweed, sand, glass])

    # set up spellbook
    magic = Magic(c, c, spellbook='lunar')
    spell = magic.set_slot(20, ['superglass_make'])

    # set up timeouts
    deposit = c.bank.utilities.deposit_inventory
    close_bank_timeout = 0
    close_bank_pressed = False

    # logging
    msg_length = 100
    t3 = time.time()

    # main loop
    print('Entering Main Loop')
    c.activate()
    while True:

        # reset for logging
        t1 = time.time()
        sys.stdout.write('\b' * msg_length)
        msg = list()

        # caps lock to pause the script
        # p to exit
        # TODO: convert these to utility functions
        if not c.screen.on_off_state():
            msg = f'Sleeping @ {time.time()}'
            sys.stdout.write(f'{msg:{msg_length}}')
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
        bank_open = deposit.identify(img)
        sand_bank_slot.update()
        seaweed_bank_slot.update()
        spell.update()
        for i in range(len(inventory)):
            c.inventory.slots[i].update()

        t1 = time.time() - t1
        msg.append(f'Update {round(t1, 2)}')

        t2 = time.time()
        cool_down = 0.05

        # do something
        if bank_open:

            seaweed_count = inventory.count(seaweed)
            seaweed_required = args.seaweed_count - seaweed_count
            seaweed_clicks = seaweed_bank_slot.clicked

            bank_contents = c.bank.tabs.active.identify(img, threshold=0.95)
            no_sand = sand_placeholder in bank_contents
            no_seaweed = seaweed_placeholder in bank_contents

            if no_sand or no_seaweed:

                sys.stdout.write('No materials left - Quiting\n')
                sys.stdout.flush()

                break

            # sys.stdout.write(
            #     f'seaweed: {seaweed_count}, '
            #     f'clicks: {len(seaweed_clicks)}, '
            #     f'timeouts: {[round(time.time() - c.offset, 2) for c in seaweed_clicks]}, '
            #     f'req: {seaweed_required}\n'
            # )
            # sys.stdout.flush()

            if glass in inventory:

                # TODO: probably easier to click a random glass instead
                if not deposit.clicked:
                    deposit.click()

                    msg.append('Deposit')
                else:
                    msg.append(f'Deposit ({deposit.time_left})')

            elif len(seaweed_clicks) < args.seaweed_count and seaweed_required > 0:

                if len(seaweed_clicks) < args.seaweed_count:
                    seaweed_bank_slot.click(tmin=2, speed=0.1)
                    cool_down = 0.01

                    # TODO: variable cooldown between repeat clicks?

                    msg.append(
                        f'Withdraw {len(seaweed_bank_slot.clicked)} '
                        f'({seaweed_bank_slot.time_left})'
                    )

                else:

                    msg.append(
                        f'Wait Withdraw {seaweed} '
                        f'({seaweed_bank_slot.time_left})'
                    )

            elif seaweed_required < 0:

                seaweed_slots = filter(
                    lambda s: s.contents == seaweed and not s.clicked,
                    c.inventory.slots
                )

                if seaweed_slots:
                    slot = random.choice(list(seaweed_slots))
                    slot.click()

                    msg.append(f'Deposit Extra {seaweed}')

                else:

                    msg.append(f'Waiting Deposit Extra {seaweed}')

            # TODO: check for over-withdrawal of sand
            elif inventory.count(sand) != args.sand_count:

                if sand_bank_slot.context_menu is None:

                    x, y = sand_bank_slot.right_click(speed=0.01)
                    sand_bank_slot.set_context_menu(
                        x, y,
                        args.sand_context_width,
                        args.sand_context_items,
                        sand_bank_slot.config['context']
                    )
                    cool_down = 0.01

                    msg.append(f'Open Sand Context Menu')

                else:

                    # TODO: template matching for context menu
                    item = sand_bank_slot.context_menu.items[withdraw_x_index]

                    if item.clicked:
                        msg.append(f'Waiting for Sand ({item.time_left})')

                    else:
                        item.click(speed=0.5)
                        msg.append(f'Withdraw {sand}')

            else:

                if not c.bank.close.clicked:
                    c.bank.close.click()
                    msg.append('Close')

                else:
                    msg.append(f'Close ({c.bank.close.time_left})')

        else:

            # TODO: confirm magic tab is open / open if not

            # TODO: use spell container identify
            x, y, _, _ = c.get_bbox()
            x1, y1, x2, y2 = spell.get_bbox()
            spell_img = img[y1 - y:y2 - y, x1 - x:x2 - x]
            if spell.identify(spell_img) == spell.name:

                if spell.clicked:

                    if bank.clicked:

                        msg.append(f'Waiting Bank Open')

                    else:

                        bank.click(tmin=0.6, tmax=1.2)
                        msg.append(f'Open Bank')

                else:
                    spell.click(tmin=3.6, tmax=5)
                    # bank can be opened after 1.8 seconds, so wait at least
                    # that long, and sometimes a bit longer seconds
                    cool_down = max(
                        [1.9, c.screen.map_between(start=1.2, stop=2.4)]
                    )

                    t3 = time.time()
                    msg.append(f'Casting {spell.name} ({round(cool_down, 2)})')

        t2 = time.time() - t2
        msg.insert(1, f'Action {round(t2, 2)}')
        msg.insert(2, f'Loop {round(time.time() - t3, 2)}')
        msg = ' - '.join(msg)

        sys.stdout.write(f'{msg[:msg_length]:{msg_length}}')
        sys.stdout.flush()

        time.sleep(cool_down)
