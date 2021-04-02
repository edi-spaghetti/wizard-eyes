import sys
import time
from os.path import dirname, join

import pyautogui

import client
from screen_tools import Screen


if __name__ == '__main__':

    # parse command line params
    tab_idx = int(sys.argv[1])
    slot_idx = int(sys.argv[2])
    context_menu_width = int(sys.argv[3])
    context_menu_item_index = int(sys.argv[4])

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
    bbox = menu.items[context_menu_item_index].get_bbox()
    img = s.grab_screen(*bbox)

    pyautogui.moveTo(*bbox[:2])
    # TODO: assert mouseover text matches item
    time.sleep(1)

    pyautogui.moveTo(*bbox[2:])
    # TODO: assert mouseover text matches item
    time.sleep(1)

    path = join(dirname(__file__), '..', 'data', 'test.png')
    s.save_img(img, path)
    t = time.time() - t

    print(
        f'Got right click context menu item {context_menu_item_index} '
        f'tab {tab_idx} slot {slot_idx} in {round(t, 2)} seconds'
    )
