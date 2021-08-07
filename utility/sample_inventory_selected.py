import time
from os.path import dirname, join, exists
from os import makedirs

import numpy

import client
from screen_tools import Screen


def sample(item_name, s, c):
    t = time.time()

    for i in range(28):

        slot = c.inventory.slots[i]

        # select the slot and give a reasonable time for game to update
        s.click_aoi(*slot.get_bbox())
        time.sleep(0.1)

        # grab whole screen fresh for each slot
        img = s.grab_screen(*c.get_bbox())
        # we need client bbox to zero the slot coordinates
        x, y, _, _ = c.get_bbox()

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

        # process and save the numpy array
        processed_img = c.inventory.slots[i].process_img(slot_img)
        numpy.save(path, processed_img)

        # now click again to deselect the current slot
        s.click_aoi(*slot.get_bbox())
        time.sleep(0.1)

    t = time.time() - t

    print(f'Got slots in {round(t, 2)} seconds')


def main():

    c = client.Client('RuneLite')
    s = Screen()
    c.set_rect()
    c._client.activate()

    for i in range(len(c.inventory.slots)):
        c.inventory.set_slot(i, [])

    print('Press enter on blank item to exit')
    print('Item name will have _selected appended')
    while True:
        item_name = input('Item Name: ')
        if item_name:
            # slight pause in case we need to hold down shift
            # for the 'use' option
            # TODO: implement [SHIFT] syntax
            time.sleep(1)
            sample(f'{item_name}_selected', s, c)
        else:
            break


if __name__ == '__main__':
    main()
