import random
import time
import sys

import keyboard

from wizard_eyes import client
from wizard_eyes.game_objects.game_objects import GameObject


def main():

    # setup
    c = client.Client('RuneLite')

    print('Bank AOI')
    bank_aoi = c.screen.gen_bbox()
    bank = GameObject(c, c)
    bank.set_aoi(*bank_aoi)

    print('Range AOI')
    range_aoi = c.screen.gen_bbox()
    range_ = GameObject(c, c)
    range_.set_aoi(*range_aoi)

    # set up item names
    item = 'karambwan'
    raw = f'raw_{item}'
    raw_selected = f'{raw}_selected'
    raw_placeholder = f'{raw}_placeholder'
    cooked = f'cooked_{item}'
    burnt = f'burnt_{item}'

    # set up bank
    tab = c.bank.tabs.set_tab(4, is_open=True)
    raw_slot = tab.set_slot(13, [raw_placeholder])
    deposit = c.bank.utilities.deposit_inventory

    # setup inventory slots
    items = [raw, cooked, burnt, raw_selected]
    for i in range(28):
        c.inventory.set_slot(i, items)

    # setup sleepers
    sleeper = GameObject(c, c)

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

        # reset for logging
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
        img = c.screen.grab_screen(*c.get_bbox())
        inventory = c.inventory.identify(img)
        bank_open = deposit.identify(img)
        raw_slot.update()
        for i in range(len(inventory)):
            c.inventory.slots[i].update()
        sleeper.update()

        # check if we want to have a little rest
        if not sleeper.time_left:
            if random.random() > 0.99:
                sleeper.add_timeout(random.random() * 3)
            if random.random() > 0.9995:
                sleeper.add_timeout(random.random() * 60)

        t1 = time.time() - t1
        msg.append(f'Update {t1:.2f}')

        t2 = time.time()
        cool_down = 0.05

        if sleeper.time_left:
            msg.append(f'Sleeping {sleeper.time_left}')

        elif bank_open:

            if keyboard.is_pressed('2'):
                keyboard.release('2')

            bank_contents = c.bank.tabs.active.identify(img, threshold=0.95)
            no_fish = raw_placeholder in bank_contents

            if no_fish:
                sys.stdout.write('No materials left - Qutting\n')
                sys.stdout.flush()
                break

            if inventory.count(raw) == 28:
                if c.bank.close.clicked:
                    msg.append(f'Wait Bank Close {c.bank.close.time_left}')
                else:
                    c.bank.close.click()
                    msg.append('Close Bank')
            elif inventory.count(None) == 28:
                if raw_slot.clicked:
                    msg.append(f'Wait Withdraw {raw} {raw_slot.time_left}')
                else:
                    raw_slot.click(tmin=2)
                    msg.append(f'Withdraw {raw}')
            else:
                if deposit.clicked:
                    msg.append(f'Wait Deposit {deposit.time_left}')
                else:
                    deposit.click(tmin=0.6, tmax=0.9)

        else:

            if not keyboard.is_pressed('2'):
                keyboard.press('2')

            if raw_selected in inventory:
                if not range_.clicked:
                    range_.click(tmin=0.6, tmax=0.9, pause_before_click=True)
                    msg.append(f'Click Range')
                else:
                    msg.append(f'Wait Click Range {range_.time_left:.2f}')
            elif raw in inventory:

                slot = c.inventory.first({raw}, order=-1)
                if slot.clicked:
                    msg.append(f'Wait Slot at index {slot.idx} '
                               f'{slot.time_left:.2f}')
                else:
                    slot.click(tmin=0.4, tmax=0.6)
                    msg.append(f'Clicked {slot.contents} at index {slot.idx}')
            else:

                if bank.clicked:
                    msg.append(f'Wait Bank Open {bank.time_left}')
                else:
                    bank.click(tmin=0.6, tmax=0.8, pause_before_click=True)
                    msg.append(f'Open Bank')

        t2 = time.time() - t2
        msg.insert(1, f'Action {t2:.2f}')
        msg.insert(2, f'Loop {time.time() - t3:.2f}')
        msg = ' - '.join(msg)

        sys.stdout.write(f'{msg[:msg_length]:{msg_length}}')
        sys.stdout.flush()

        time.sleep(cool_down)


if __name__ == '__main__':
    main()