import sys
import time
from os.path import dirname, join, exists
from os import makedirs

import client
from screen_tools import Screen


if __name__ == '__main__':
    item_name = sys.argv[1]

    c = client.Client('RuneLite')
    s = Screen()
    c.set_rect()
    c._client.activate()

    t = time.time()

    # grab whole screen and sample from it for each slot
    img = s.grab_screen(*c.get_bbox())
    # we need client bbox to zero the slot coordinates
    x, y, _, _ = c.get_bbox()

    for i in range(28):

        x1, y1, x2, y2 = c.inventory.slots[i].get_bbox()
        # numpy arrays are stored rows x columns, so flip x and y
        slot_img = img[y1-y:y2-y, x1-x:x2-x]

        path = join(
            dirname(__file__), '..', 'data',
            'inventory', str(i), item_name+'.png')
        if not exists(dirname(path)):
            makedirs(dirname(path))

        s.save_img(slot_img, path)

    t = time.time() - t

    print(f'Got slots in {round(t, 2)} seconds')
