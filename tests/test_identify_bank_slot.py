import sys
import time

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
    tab = c.bank.tabs.set_tab(tab_idx, is_open=True)
    slot = tab.set_slot(slot_idx, [slot_name])

    t = time.time()

    c._client.activate()

    # get section of screen we think bank slot should be in
    img = s.grab_screen(*slot.get_bbox())

    # run a template match
    matched = slot.identify(img)

    print(matched == slot_name)

    t = time.time() - t

    print(f'Identified bank slot {slot_name} in {round(t, 2)} seconds')
