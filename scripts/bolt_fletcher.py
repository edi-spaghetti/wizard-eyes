import sys
import time
import random

import client
from game_objects import GameObject
from script_utils import safety_catch


def main():

    # setup
    print('Setting Up')
    c = client.Client('RuneLite')

    # setup inventory slots
    for i in range(28):
        c.inventory.set_slot(i, [])

    msg_length = 100

    # main loop
    print('Entering Main Loop')
    c.activate()
    while True:

        # reset for logging
        sys.stdout.write('\b' * msg_length)
        msg = list()

        if safety_catch(c, msg_length):
            continue

        # update
        img = c.screen.grab_screen(*c.get_bbox())
        inventory = c.inventory.identify(img)
        for i in range(len(inventory)):
            c.inventory.slots[i].update()

        t2 = time.time()
        cool_down = 0.05

        # action
        if c.inventory.slots[4].clicked:
            c.inventory.slots[5].click(tmin=0.3, tmax=0.4)
        else:
            c.inventory.slots[4].click(tmin=0.3, tmax=0.4)

        if random.random() > 0.9:
            cool_down = random.random()

        msg = f'Fletching - {time.time() - t2}'

        sys.stdout.write(f'{msg[:msg_length]:{msg_length}}')
        sys.stdout.flush()

        time.sleep(cool_down)


if __name__ == '__main__':
    main()
