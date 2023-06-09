import argparse
import time
from os.path import dirname, exists, basename
from os import makedirs
import re
from shutil import rmtree, copy2

import cv2
import numpy

from wizard_eyes import client
from wizard_eyes.file_path_utils import get_root


def sample(item_name, s, c, idx, overwrite=False):
    t = time.time()

    # grab whole screen and sample from it for each slot
    img = s.grab_screen(*c.get_bbox())
    # we need client bbox to zero the slot coordinates
    x, y, _, _ = c.get_bbox()

    for i in range(28):

        x1, y1, x2, y2 = c.inventory.slots[i].get_bbox()
        # numpy arrays are stored rows x columns, so flip x and y
        slot_img = img[y1-y:y2-y, x1-x:x2-x]
        # create a rough mask - will need to be verified & refined
        mask = cv2.inRange(slot_img, (38, 50, 59, 255), (46, 56, 66, 255))
        mask = cv2.bitwise_not(mask)

        # prepare path to save template into
        path = c.inventory.slots[i].PATH_TEMPLATE.format(
                root=get_root(),
                index=i,
                name=item_name
            )
        if not exists(dirname(path)):
            makedirs(dirname(path))

        # first save a colour copy for reference
        cv2.imwrite(path.replace('.npy', '.png'), slot_img)
        cv2.imwrite(path.replace('.npy', '_mask.png'), mask)
        # process and save the numpy array
        processed_img = c.inventory.slots[i].process_img(slot_img)
        numpy.save(path, processed_img)
        numpy.save(path.replace('.npy', '_mask.npy'), mask)

    if idx is not None:
        for ext in ('.npy', '.png', '_mask.npy', '_mask.png'):
            source = c.inventory.slots[idx].PATH_TEMPLATE.format(
                root=get_root(),
                index=idx,
                name=item_name
            ).replace('.npy', ext)
            target = (f'{get_root()}/data/tabs/{basename(source)}'
                      .replace('.npy', ext))

            if exists(target) and not overwrite:
                print(f'{target} already exists, skipping')
                continue

            copy2(source, target)

    t = time.time() - t

    print(f'Got slots in {round(t, 2)} seconds')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--clear', action='store_true', default=False)
    parser.add_argument('--overwrite', action='store_true', default=False)
    args = parser.parse_args()

    c = client.Client('RuneLite')
    s = c.screen

    for i in range(len(c.inventory.slots)):
        c.inventory.set_slot(i)

        path = c.inventory.slots[i].PATH_TEMPLATE.format(
            root=get_root(),
            index=i,
            name='_'
        )
        path = dirname(path)

        if args.clear and exists(path):
            rmtree(path)

    print('Press enter on blank item to exit')
    print('Follow item name with index to save to tabs directory '
          '(will overwrite existing without warning!)')
    while True:
        item_name = input('Item Name: ')
        idx = None
        match = re.match('([^()]+)\s?\(?([0-9]+)?\)?', item_name)
        if match:
            item_name = match.group(1)
            idx = int(match.group(2))

        if item_name:
            sample(item_name, s, c, idx, overwrite=args.overwrite)
        else:
            break


if __name__ == '__main__':
    main()
