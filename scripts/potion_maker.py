import sys
import time

import keyboard

import client
import screen_tools


if __name__ == '__main__':

    # setup
    print('Setting Up')
    c = client.Client('RuneLite')
    s = screen_tools.Screen()

    # TODO: add these to argparse / script config
    bank_aoi = s.gen_bbox()
    print(bank_aoi)

    water_bank_index = sys.argv[1]  # 91
    clean_herb_bank_index = sys.argv[2]  # 90
    water = 'vial_of_water'
    clean = sys.argv[3]  # lantadyme
    unfinished_potion = f'{clean}_potion_unf'

    # set up slots
    # TODO

    # set up timeouts
    # TODO

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
        msg += 'Refreshing Screen'
        img = s.grab_screen(*c.get_bbox())
        # TODO: identify what's on screen

        cool_down = 0.05

        # do something
        # TODO: sneaky bot stuff

        sys.stdout.write(f'{msg:50}')
        sys.stdout.flush()

        time.sleep(cool_down)
