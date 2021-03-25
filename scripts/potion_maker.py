import sys
import time
import random

import keyboard

import client
import screen_tools


if __name__ == '__main__':

    # setup
    print('Setting Up')
    c = client.Client('RuneLite')
    s = screen_tools.Screen()

    # TODO: add these to argparse / script config
    # bank_aoi = (-1173, 578, -779, 787)
    bank_aoi = s.gen_bbox()
    print(bank_aoi)

    water_bank_index = int(sys.argv[1])  # 91
    clean_herb_bank_index = int(sys.argv[2])  # 90
    water = 'vial_of_water'
    water_selected = f'{water}_selected'
    clean = sys.argv[3]  # lantadyme
    clean_selected = f'{clean}_selected'
    unfinished_potion = f'{clean}_potion_unf'

    # set up slots
    for i in range(14):
        c.inventory.set_slot(i, [clean, unfinished_potion, clean_selected])
    for i in range(14, 28):
        c.inventory.set_slot(i, [water, water_selected])

    # set up dialog buttons
    c.dialog.add_make('lantadyme')

    # set up timeouts
    open_bank_clicked = False
    open_bank_timeout = time.time()
    close_bank_clicked = False
    close_bank_timeout = time.time()
    bank_herb_clicked = False
    bank_herb_timeout = time.time()
    bank_vial_clicked = False
    bank_vial_timeout = time.time()
    deposit_clicked = False
    deposit_timeout = time.time()
    inventory_clicked = [False] * len(c.inventory.slots)
    inventory_timeout = [time.time()] * len(c.inventory.slots)
    dialog_make_clicked = False
    dialog_make_timeout = time.time()

    # audit fields for when clicked
    close_bank_clicked_at = 0
    dialog_make_clicked_at = 0

    msg_length = 50

    # main loop
    print(f'Entering Main Loop')
    c.activate()
    t = time.time()
    while True:

        sys.stdout.write('\b' * msg_length)
        msg = ''

        # caps lock to pause the script
        # p to exit
        # TODO: convert these to utility functions
        if not s.on_off_state():
            msg += f'Sleeping @ {time.time()}'
            sys.stdout.write(f'{msg:50}')
            sys.stdout.flush()
            time.sleep(0.1)
            continue
        elif keyboard.is_pressed('p'):
            exit(1)

        # update
        t2 = time.time()
        img = s.grab_screen(*c.get_bbox())
        inventory = c.inventory.identify(img)
        bank_open = c.bank.utilities.deposit_inventory.identify(img)
        dialog_open = c.dialog.makes[0].identify(img)

        # TODO: update timeouts altogether on each iteration?
        for i, state in enumerate(inventory):
            if time.time() > inventory_timeout[i]:
                inventory_clicked[i] = False

        t2 = time.time() - t2
        msg += f'Update {round(t2, 2)}'

        cool_down = 0.05

        # do something
        if bank_open:
            if unfinished_potion in inventory[:14]:

                if time.time() > deposit_timeout:
                    deposit_clicked = False

                if not deposit_clicked:
                    # put the unfinished potions back in the bank
                    s.click_aoi(*c.bank.utilities.deposit_inventory.get_bbox())
                    deposit_timeout = time.time() + 1 + random.random() * 3
                    deposit_clicked = True

                    msg += ' - Deposit'
                else:
                    time_left = round(deposit_timeout - time.time(), 2)
                    msg += f' - Waiting Deposit ({time_left})'

            elif all([i is None for i in inventory]):

                if time.time() > bank_herb_timeout:
                    bank_herb_clicked = False

                if not bank_herb_clicked:
                    # take out 14 herbs
                    s.click_aoi(*c.bank.get_slot_bbox(clean_herb_bank_index))
                    bank_herb_timeout = time.time() + 1 + random.random() * 3
                    bank_herb_clicked = True

                    msg += ' - Withdraw Herbs'

                else:
                    time_left = round(bank_herb_timeout - time.time(), 2)
                    msg += f' - Waiting Withdraw Herbs ({time_left})'

            elif clean in inventory and water not in inventory:

                if time.time() > bank_vial_timeout:
                    bank_vial_clicked = False

                if not bank_vial_clicked:
                    # take out 14 vials of water
                    s.click_aoi(*c.bank.get_slot_bbox(water_bank_index))
                    bank_vial_timeout = time.time() + 1 + random.random() * 3
                    bank_vial_clicked = True

                    msg += ' - Withdraw Water'

                else:
                    time_left = round(bank_vial_timeout - time.time(), 2)
                    msg += f' - Waiting Withdraw Water ({time_left})'

            else:

                if time.time() > close_bank_timeout:
                    close_bank_clicked = False

                if not close_bank_clicked:
                    # close the bank window
                    keyboard.press('esc')
                    close_bank_pressed = True
                    close_bank_clicked_at = time.time()
                    close_bank_timeout = time.time() + 1 + random.random() * 3

                    msg += ' - Close'

                else:
                    time_left = round(close_bank_timeout - time.time(), 2)
                    msg += f' - Waiting Close ({time_left})'

        elif dialog_open:

            if time.time() > dialog_make_timeout:
                dialog_make_clicked = False

            if not dialog_make_clicked:
                # select make
                keyboard.press('space')
                dialog_make_clicked = True
                dialog_make_clicked_at = time.time()
                dialog_make_timeout = time.time() + 1 + random.random() * 3

                msg += ' - Make'

            else:
                time_left = round(dialog_make_timeout - time.time(), 2)
                msg += f' - Waiting Make ({time_left})'

        else:

            if None in inventory:

                if water not in inventory and clean not in inventory:

                    if time.time() > open_bank_timeout:
                        open_bank_clicked = False

                    if not open_bank_clicked:
                        # open the bank
                        s.click_aoi(*bank_aoi, pause_before_click=True)
                        open_bank_timeout = time.time() + 1 + random.random() * 3
                        open_bank_clicked = True

                        msg += ' - Open'

                    else:
                        time_left = round(open_bank_timeout - time.time(), 2)
                        msg += f' - Waiting Bank Open ({time_left})'

                else:

                    # wait for potions - TODO: check for timeout
                    msg += f' - Waiting for Potions'

            else:

                if water_selected in inventory or clean_selected in inventory:

                    # click the other one
                    if water_selected in inventory:
                        other_item = clean
                    else:
                        other_item = water

                    # choose one of those items to use it on
                    # TODO: nearest weighted random
                    other_slots = list(filter(
                        lambda s: s.contents == other_item,
                        c.inventory.slots
                    ))
                    recently_clicked = list(filter(
                        lambda s: inventory_clicked[s.idx]
                                  and s.contents == other_item,
                        other_slots
                    ))

                    if not recently_clicked:

                        # randomly choose one of the other items a click it
                        other_slot = random.choice(other_slots)
                        i = other_slot.idx

                        s.click_aoi(*c.inventory.slots[i].get_bbox())
                        inventory_clicked[i] = True
                        inventory_timeout[i] = time.time() + 1 + random.random() * 3

                        msg += f' - Clicked {other_item} at index {i}'

                    else:

                        # this condition shouldn't trigger, because if we're
                        # still registering a selected item after clicking
                        # another item likely the game has frozen mid-click
                        names = [s.contents for s in recently_clicked]
                        msg += f' - Waiting for Click on {names}'

                else:

                    bank_clicked_last = close_bank_clicked_at > dialog_make_clicked_at
                    neither_clicked = close_bank_clicked_at == 0 and dialog_make_clicked_at == 0

                    if bank_clicked_last or neither_clicked:

                        # check if any inventory items have been clicked
                        # recently
                        recently_clicked = list(filter(
                            lambda s: inventory_clicked[s.idx],
                            c.inventory.slots
                        ))

                        if not recently_clicked:

                            # click any item
                            i = random.randint(0, 27)

                            s.click_aoi(*c.inventory.slots[i].get_bbox())
                            inventory_clicked[i] = True
                            inventory_timeout[i] = time.time() + 1 + random.random() * 3

                            slot = c.inventory.slots[i]
                            msg += f' - Clicked {slot.contents} at index {i}'

                            # slight race condition here where we've clicked the
                            # item, but it hasn't shown up as selected yet, so
                            # we'll end up clicking again
                            cool_down = 0.1

                        else:

                            msg += ' - Waiting for dialog'

                    else:

                        # TODO: if interRupted we stuck here

                        msg += ' - Waiting for Potions'

        sys.stdout.write(f'{msg:50}')
        sys.stdout.flush()

        time.sleep(cool_down)
