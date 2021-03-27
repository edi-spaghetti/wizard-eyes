import sys
import time
from os.path import dirname, join

import pyautogui

import client
from screen_tools import Screen


if __name__ == '__main__':

    # parse command line params
    tab_idx = int(sys.argv[1])
    slot_idx = int(sys.argv[1])

    # setup
    c = client.Client('RuneLite')
    s = Screen()
    c.set_rect()
    tab = c.bank.tabs.set_tab(tab_idx, is_open=True)
    slot = tab.set_slot(slot_idx, [])

    c._client.activate()

    t = time.time()

    img = s.grab_screen(*slot.get_bbox())
    path = join(dirname(__file__), '..', 'data', 'test.png')
    s.save_img(img, path)
    t = time.time() - t

    x, y = slot.get_bbox()[:2]
    pyautogui.moveTo(x, y)
    # TODO: assert mouseover text says 'Deposit inventory / n more options'
    time.sleep(1)

    pyautogui.moveTo(*slot.get_bbox()[2:])
    # TODO: assert mouseover text says 'Deposit inventory / n more options'
    time.sleep(1)

    print(f'Got bank tab {tab_idx} slot {slot_idx} in {round(t, 2)} seconds')
