import sys
import time

import client
from screen_tools import Screen


if __name__ == '__main__':

    name = sys.argv[1]

    c = client.Client('RuneLite')
    s = Screen()
    c.set_rect()
    c._client.activate()

    # set up dialog with one make button
    button = c.dialog.add_make(name)

    t = time.time()

    # grab whole screen and sample from it for each slot
    img = s.grab_screen(*c.get_bbox())

    # run identification
    matched = button.identify(img)

    # print with indexes to check
    print(matched)

    t = time.time() - t

    print(f'Identified dialog make {name} in {round(t, 2)} seconds')
