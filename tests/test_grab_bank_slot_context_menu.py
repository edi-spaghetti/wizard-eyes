import sys
import time
from os.path import dirname, join

import client
from screen_tools import Screen


if __name__ == '__main__':

    # parse command line params
    tab_idx = int(sys.argv[1])
    slot_idx = int(sys.argv[2])
    context_menu_width = int(sys.argv[3])

    # setup
    c = client.Client('RuneLite')
    s = Screen()
    c.set_rect()
    tab = c.bank.tabs.set_tab(tab_idx, is_open=True)
    slot = tab.set_slot(slot_idx, [])

    c._client.activate()

    t = time.time()

    rx, ry = slot.right_click()
    menu = slot.set_context_menu(
        rx, ry, context_menu_width, 10, slot.config['context']
    )
    img = s.grab_screen(*menu.get_bbox())

    path = join(dirname(__file__), '..', 'data', 'test.png')
    s.save_img(img, path)
    t = time.time() - t

    print(
        f'Got right click context menu '
        f'tab {tab_idx} slot {slot_idx} in {round(t, 2)} seconds'
    )
