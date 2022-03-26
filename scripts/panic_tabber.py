import argparse
import time
import sys

import cv2
import numpy
import keyboard

from wizard_eyes.client import Client
from wizard_eyes.script_utils import safety_catch


def main():

    c = Client('RuneLite')

    # template = numpy.ones((2, 2))
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory-id', type=int)
    parser.add_argument('--hotkey', type=str, default='g')
    args = parser.parse_args()

    # TOOD: identify the teleport tab
    for i in range(28):
        c.inventory.set_slot(i, [])

    msg_length = 100
    cool_down = 0.05
    print('Entering Main Loop')
    while True:

        # reset for logging
        t1 = time.time()
        sys.stdout.write('\b' * msg_length)
        msg = list()

        if safety_catch(c, msg_length):
            continue
        if keyboard.is_pressed(args.hotkey):
            c.inventory.slots[args.inventory_id].click(speed=0.1)
            msg.append(f'Clicking inventory slot: {args.inventory_id}')

        t1 = time.time() - t1
        msg.insert(0, f'Update {t1:.2f}')
        msg = ' - '.join(msg)

        sys.stdout.write(f'{msg[:msg_length]:{msg_length}}')
        sys.stdout.flush()

        # TODO: auto-tab on player detection
        # img = c.screen.grab_screen(*c.get_bbox())
        # x, y, _, _ = c.get_bbox()
        # x1, y1, x2, y2 = c.minimap.minimap.get_bbox()
        # minimap_img = img[y1 - y:y2 - y, x1 - x:x2 - x]
        #
        # match = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

        time.sleep(cool_down)


if __name__ == '__main__':
    main()
