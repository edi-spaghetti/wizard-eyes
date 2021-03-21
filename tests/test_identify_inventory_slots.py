import time

import client
from screen_tools import Screen


if __name__ == '__main__':

    c = client.Client('RuneLite')
    s = Screen()
    c.set_rect()
    c._client.activate()

    t = time.time()

    # grab whole screen and sample from it for each slot
    img = s.grab_screen(*c.get_bbox())

    # run identification
    names = c.inventory.identify(img)

    # print with indexes to check
    print([item for item in enumerate(names)])

    t = time.time() - t

    print(f'Identified slots in {round(t, 2)} seconds')
