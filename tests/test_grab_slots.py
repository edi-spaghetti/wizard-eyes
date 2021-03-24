import time
from os.path import dirname, join

import client
from screen_tools import Screen


if __name__ == '__main__':
    c = client.Client('RuneLite')
    s = Screen()
    c.set_rect()
    c._client.activate()

    t = time.time()
    c.inventory.set_slot(0, [])
    x0, y0 = c.inventory.slots[0].get_bbox()[:2]
    c.inventory.set_slot(27, [])
    x27, y27 = c.inventory.slots[27].get_bbox()[2:]

    img = s.grab_screen(x0, y0, x27, y27)
    path = join(dirname(__file__), '..', 'data', 'test.png')
    s.save_img(img, path)
    t = time.time() - t

    print(f'Got screen in {round(t, 2)} seconds')
