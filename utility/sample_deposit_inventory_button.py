import sys
import time
from os.path import dirname, join, exists
from os import makedirs

import numpy

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
    # we need client bbox to zero the slot coordinates
    x, y, _, _ = c.get_bbox()

    button = c.bank.utilities.deposit_inventory

    x1, y1, x2, y2 = button.get_bbox()
    # numpy arrays are stored rows x columns, so flip x and y
    trg_img = img[y1 - y:y2 - y, x1 - x:x2 - x]

    # prepare path to save template into
    path = button.PATH_TEMPLATE.format(
        root=join(dirname(__file__), '..'),
    )
    if not exists(dirname(path)):
        makedirs(dirname(path))

    # process and save the numpy array
    processed_img = button.process_img(trg_img)

    numpy.save(path, processed_img)

    t = time.time() - t

    print(f'Got slots in {round(t, 2)} seconds')
