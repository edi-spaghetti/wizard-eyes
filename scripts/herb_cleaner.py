import sys
import time
import random
import argparse

import pyautogui
import keyboard

from wizard_eyes.client import Client
from wizard_eyes import screen_tools

pyautogui.PAUSE = 0


if __name__ == '__main__':

    # setup
    print('Setting Up')
    c = Client('RuneLite')
    s = screen_tools.Screen()

    parser = argparse.ArgumentParser()
    parser.add_argument('-ti', '--tab-index', type=int, required=True)
    parser.add_argument('-gi', '--grimy-index', type=int, required=True)
    parser.add_argument('-hn', '--herb-name', type=str, required=True)
    parser.add_argument('-b', '--bank-aoi', type=lambda x: tuple([int(y) for y in x]))
    parser.add_argument('-s', '--speed', type=float, default=1)

    args = parser.parse_args()

    bank_aoi = args.bank_aoi or s.gen_bbox()
    print(bank_aoi)

    tab_idx = args.tab_index
    grimy_herbs_bank_index = args.grimy_index
    grimy = f'grimy_{args.herb_name}'
    clean = args.herb_name
    placeholder = f'{grimy}_placeholder'

    # set up bank slots
    tab = c.bank.tabs.set_tab(tab_idx, is_open=True)
    grimy_bank_slot = tab.set_slot(grimy_herbs_bank_index, [placeholder])

    # set up slots with desired items
    for i in range(len(c.inventory.slots)):
        c.inventory.set_slot(i, [grimy, clean])

    # TODO: better of managing timeouts
    inventory_clicked = [False] * len(c.inventory.slots)
    inventory_timeout = [time.time()] * len(c.inventory.slots)
    deposit_btn_clicked = False
    deposit_timeout = time.time()
    bank_herb_clicked = False
    bank_herb_timeout = time.time()
    close_bank_pressed = False
    close_bank_timeout = time.time()
    open_bank_clicked = False
    open_bank_timeout = time.time()

    msg_length = 50

    print(f'Entering Main Loop')
    c.activate()
    t = time.time()
    do_update = True
    while True:

        sys.stdout.write('\b' * msg_length)
        msg = ''

        # press caps locks to pause the script
        if not s.on_off_state():
            msg += f'Sleeping @ {time.time()}'
            sys.stdout.write(f'{msg:50}')
            sys.stdout.flush()
            time.sleep(0.1)
            continue
        elif keyboard.is_pressed('p'):
            exit(1)

        # update
        msg += 'Refreshing Screen'

        img = s.grab_screen(*c.get_bbox())
        inventory = c.inventory.identify(img, threshold=0.98)
        bank_open = c.bank.utilities.deposit_inventory.identify(img)
        t = time.time()

        cool_down = 0.05

        # do something
        if bank_open:

            if clean in inventory:

                if time.time() > deposit_timeout:
                    deposit_btn_clicked = False

                if not deposit_btn_clicked:
                    # put the clean herbs back in the bank
                    s.click_aoi(*c.bank.utilities.deposit_inventory.get_bbox())
                    deposit_timeout = time.time() + 1 + random.random() * 3
                    deposit_btn_clicked = True

                    msg += ' - Deposit'
                    cool_down = 0.2
                else:
                    msg += f' - Waiting Deposit ({round(deposit_timeout - time.time(), 2)})'

            elif grimy in inventory:

                if time.time() > close_bank_timeout:
                    close_bank_pressed = False

                if not close_bank_pressed:
                    # close the bank window
                    msg += ' - Close'
                    keyboard.press('esc')
                    close_bank_pressed = True
                    close_bank_timeout = time.time() + 1 + random.random() * 3
                else:
                    msg += f' - Waiting Close ({round(close_bank_timeout - time.time(), 2)})'

            elif placeholder in c.bank.tabs.active.identify(img, threshold=0.98):
                sys.stdout.write('No herbs left - Quiting\n')
                sys.stdout.flush()

                break

            else:

                if time.time() > bank_herb_timeout:
                    bank_herb_clicked = False

                if not bank_herb_clicked:
                    # withdraw a new set of grimy herbs
                    s.click_aoi(*grimy_bank_slot.get_bbox())
                    bank_herb_timeout = time.time() + 1 + random.random() * 3
                    bank_herb_clicked = True

                    msg += ' - Withdraw'
                    clicked = [False] * len(c.inventory.slots)
                    cool_down = 0.2
                else:
                    msg += f' - Waiting Withdraw ({round(bank_herb_timeout - time.time(), 2)})'

        else:
            if grimy in inventory:

                # if no grimy herbs are found, we just wait this round
                cool_down = 0.1

                choices = list()
                for i, state in enumerate(inventory):

                    if time.time() > inventory_timeout[i]:
                        inventory_clicked[i] = False

                    if state == grimy and not inventory_clicked[i]:
                        t2 = time.time()
                        s.click_aoi(*c.inventory.slots[i].get_bbox(),
                                    speed=args.speed)
                        t2 = time.time() - t2

                        inventory_clicked[i] = True
                        inventory_timeout[i] = time.time() + 1 + random.random() * 3

                        msg += f' - Clean in {round(t2, 2)}'
                        cool_down = 0.01
                        break

            else:

                if time.time() > open_bank_timeout:
                    open_bank_clicked = False

                if not open_bank_clicked:
                    # open the bank
                    # TODO: make sure you don't have an item selected! This can
                    #       accidentally lead to using it on a player and then
                    #       everything's broken
                    s.click_aoi(*bank_aoi, pause_before_click=True)
                    open_bank_timeout = time.time() + 1 + random.random() * 3
                    open_bank_clicked = True

                    msg += ' - Open'
                else:
                    msg += f' - Waiting Bank Open ({round(open_bank_timeout - time.time(), 2)})'

        sys.stdout.write(f'{msg:50}')
        sys.stdout.flush()

        time.sleep(cool_down)
