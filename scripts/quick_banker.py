import random
import time
import sys

import keyboard

import client
from game_objects import GameObject


def main():

    # setup
    c = client.Client('RuneLite')

    # print('Stairs AOI')
    # stairs_aoi = c.screen.gen_bbox()
    # stairs = GameObject(c, c)
    # stairs.set_aoi(*stairs_aoi)

    # set up item names
    bowstring = 'bowstring'
    flax = 'flax'

    # set up bank
    tab = c.bank.tabs.set_tab(2, is_open=True)
    flax_slot = tab.set_slot(0, [])
    deposit = c.bank.utilities.deposit_inventory

    # setup inventory slots
    items = [flax, bowstring]
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

        for i in range(len(inventory)):
            c.inventory.slots[i].update()
        sleeper.update()

        # check if we want to have a little rest
        # if not sleeper.time_left:
        #     if random.random() > 0.99:
        #         sleeper.add_timeout(random.random() * 3)
        #     if random.random() > 0.9995:
        #         sleeper.add_timeout(random.random() * 60)

        t1 = time.time() - t1
        msg.append(f'Update {t1:.2f}')

        t2 = time.time()
        cool_down = 0.05

        if sleeper.time_left:
            msg.append(f'Sleeping {sleeper.time_left}')

        elif bank_open:

            if inventory.count(flax) == 28:
                if c.bank.close.clicked:
                    msg.append(f'Wait Bank Close {c.bank.close.time_left}')
                else:
                    c.bank.close.click()
                    msg.append('Close Bank')
            elif inventory.count(None) == 28:
                if flax_slot.clicked:
                    msg.append(f'Wait Withdraw {flax} {flax_slot.time_left}')
                else:
                    flax_slot.click(tmin=2)
                    msg.append(f'Withdraw {flax}')
            else:
                if deposit.clicked:
                    msg.append(f'Wait Deposit {deposit.time_left}')
                else:
                    deposit.click(tmin=0.6, tmax=0.9)
        else:
            msg.append(f'Waiting')

        #
        # else:
        #
        #     # if not keyboard.is_pressed('2'):
        #     #     keyboard.press('2')
        #
        #     if raw_selected in inventory:
        #         if not range_.clicked:
        #             range_.click(tmin=0.6, tmax=0.9, pause_before_click=True)
        #             msg.append(f'Click Range')
        #         else:
        #             msg.append(f'Wait Click Range {range_.time_left:.2f}')
        #     elif raw in inventory:
        #
        #         slot = c.inventory.first({raw}, order=-1)
        #         if slot.clicked:
        #             msg.append(f'Wait Slot at index {slot.idx} '
        #                        f'{slot.time_left:.2f}')
        #         else:
        #             slot.click(tmin=0.4, tmax=0.6)
        #             msg.append(f'Clicked {slot.contents} at index {slot.idx}')
        #     else:
        #
        #         if bank.clicked:
        #             msg.append(f'Wait Bank Open {bank.time_left}')
        #         else:
        #             bank.click(tmin=0.6, tmax=0.8, pause_before_click=True)
        #             msg.append(f'Open Bank')

        t2 = time.time() - t2
        msg.insert(1, f'Action {t2:.2f}')
        msg.insert(2, f'Loop {time.time() - t3:.2f}')
        msg = ' - '.join(msg)

        sys.stdout.write(f'{msg[:msg_length]:{msg_length}}')
        sys.stdout.flush()

        time.sleep(cool_down)


if __name__ == '__main__':
    main()