import sys

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
    spell = magic.set_slot(spell_idx, [])

    bbox = spell.get_bbox()
    img = s.grab_screen(*bbox)

    print(spell.identify(img))
