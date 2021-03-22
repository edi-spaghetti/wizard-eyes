import sys
import time
from os.path import dirname, join

import pyautogui

import client
from screen_tools import Screen


if __name__ == '__main__':
    c = client.Client('RuneLite')
    s = Screen()
    c.set_rect()
    c._client.activate()

    t = time.time()

    idx = int(sys.argv[1])

    img = s.grab_screen(*c.bank.get_slot_bbox(idx))
    path = join(dirname(__file__), '..', 'data', 'test.png')
    s.save_img(img, path)
    t = time.time() - t

    x, y = c.bank.get_slot_bbox(idx)[:2]
    pyautogui.moveTo(x, y)
    # TODO: assert mouseover text says 'Deposit inventory / 1 more options'
    time.sleep(1)

    pyautogui.moveTo(*c.bank.get_slot_bbox(idx)[2:])
    # TODO: assert mouseover text says 'Deposit inventory / 1 more options'
    time.sleep(1)

    print(f'Got screen in {round(t, 2)} seconds')
