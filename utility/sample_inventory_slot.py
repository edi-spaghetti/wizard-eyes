import sys
import time
from os.path import dirname, join, exists
from os import makedirs

import numpy

from wizard_eyes import client
from wizard_eyes.screen_tools import Screen


if __name__ == '__main__':
    item_name = sys.argv[1]
    inv_slot_idx = int(sys.argv[2])

    c = client.Client('RuneLite')
    s = Screen()
    c.set_rect()
    c._client.activate()

    c.inventory.set_slot(inv_slot_idx, [])

    t = time.time()

    # grab whole screen and sample from it for each slot
    img = s.grab_screen(*c.get_bbox())
    # we need client bbox to zero the slot coordinates
    x, y, _, _ = c.get_bbox()

    x1, y1, x2, y2 = c.inventory.slots[inv_slot_idx].get_bbox()
    # numpy arrays are stored rows x columns, so flip x and y
    slot_img = img[y1-y:y2-y, x1-x:x2-x]

    # prepare path to save template into
    path = c.inventory.slots[inv_slot_idx].PATH_TEMPLATE.format(
            root=join(dirname(__file__), '..'),
            index=inv_slot_idx,
            name=item_name
        )
    if not exists(dirname(path)):
        makedirs(dirname(path))

    # process and save the numpy array
    processed_img = c.inventory.slots[inv_slot_idx].process_img(slot_img)

    numpy.save(path, processed_img)

    t = time.time() - t

    print(f'Got slots in {round(t, 2)} seconds')
