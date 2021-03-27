import sys
from os.path import dirname, join, exists
from os import makedirs

import numpy

import client
from screen_tools import Screen


if __name__ == '__main__':

    # parse command line params
    tab_idx = int(sys.argv[1])
    slot_idx = int(sys.argv[2])
    slot_name = sys.argv[3]

    # setup
    c = client.Client('RuneLite')
    s = Screen()
    c.set_rect()

    c._client.activate()

    tab = c.bank.tabs.set_tab(tab_idx, is_open=True)
    slot = tab.set_slot(slot_idx, [])

    # currently only sampling one slot at a time
    img = s.grab_screen(*slot.get_bbox())

    path = slot.PATH_TEMPLATE.format(
        root=join(dirname(__file__), '..'),
        tab=tab.idx,
        index=slot.idx,
        name=slot_name
    )
    if not exists(dirname(path)):
        makedirs(dirname(path))

    # save as pre-processed numpy array
    processed_img = slot.process_img(img)
    numpy.save(path, processed_img)
