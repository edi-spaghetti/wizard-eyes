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
    matched = c.bank.utilities.deposit_inventory.identify(img)

    # print with indexes to check
    print(matched)

    t = time.time() - t

    print(f'Identified deposit inventory button in {round(t, 2)} seconds')
