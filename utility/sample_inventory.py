import time
from os.path import dirname, join, exists
from os import makedirs

import cv2
import numpy

from wizard_eyes import client
from wizard_eyes.screen_tools import Screen


def sample(item_name, s, c):
    t = time.time()

    # grab whole screen and sample from it for each slot
    img = s.grab_screen(*c.get_bbox())
    # we need client bbox to zero the slot coordinates
    x, y, _, _ = c.get_bbox()

    for i in range(28):

        x1, y1, x2, y2 = c.inventory.slots[i].get_bbox()
        # numpy arrays are stored rows x columns, so flip x and y
        slot_img = img[y1-y:y2-y, x1-x:x2-x]

        # prepare path to save template into
        path = c.inventory.slots[i].PATH_TEMPLATE.format(
                root=join(dirname(__file__), '..'),
                index=i,
                name=item_name
            )
        if not exists(dirname(path)):
            makedirs(dirname(path))

        # first save a colour copy for reference
        cv2.imwrite(path.replace('.npy', '.png'), slot_img)
        # process and save the numpy array
        processed_img = c.inventory.slots[i].process_img(slot_img)
        numpy.save(path, processed_img)

    t = time.time() - t

    print(f'Got slots in {round(t, 2)} seconds')


def main():

    c = client.Client('RuneLite')
    s = c.screen

    for i in range(len(c.inventory.slots)):
        c.inventory.set_slot(i, [])

    print('Press enter on blank item to exit')
    while True:
        item_name = input('Item Name: ')
        if item_name:
            sample(item_name, s, c)
        else:
            break


if __name__ == '__main__':
    main()
