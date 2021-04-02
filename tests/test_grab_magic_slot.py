import sys
import time
from os.path import dirname, join

import pyautogui

import client
from screen_tools import Screen
from game_objects import Magic


if __name__ == '__main__':
    spellbook = sys.argv[1]
    spell_idx = int(sys.argv[2])

    c = client.Client('RuneLite')
    s = Screen()
    c.set_rect()
    magic = Magic(c, c, spellbook=spellbook)
    spell = magic.set_slot(spell_idx, str(spell_idx))

    c._client.activate()

    t = time.time()

    bbox = spell.get_bbox()
    img = s.grab_screen(*bbox)
    path = join(dirname(__file__), '..', 'data', 'test.png')
    s.save_img(img, path)
    t = time.time() - t

    pyautogui.moveTo(*bbox[:2])
    # TODO: assert mouseover text says 'Deposit inventory / n more options'
    time.sleep(1)

    pyautogui.moveTo(*bbox[2:])
    # TODO: assert mouseover text says 'Deposit inventory / n more options'
    time.sleep(1)

    print(f'Got screen in {round(t, 2)} seconds')
