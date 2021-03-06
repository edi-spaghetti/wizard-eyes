import sys
import time
from os.path import dirname, join, exists
from os import makedirs

import numpy

from wizard_eyes import client
from wizard_eyes.screen_tools import Screen
from wizard_eyes.game_objects.personal_menu import Magic


if __name__ == '__main__':
    spellbook = sys.argv[1]
    spell_idx = int(sys.argv[2])

    c = client.Client('RuneLite')
    s = Screen()
    c.set_rect()
    magic = Magic(c, c, spellbook=spellbook)
    spell = magic.set_slot(spell_idx, str(spell_idx))

    t = time.time()

    # grab whole screen and sample from it for each slot
    img = s.grab_screen(*c.get_bbox())
    # we need client bbox to zero the slot coordinates
    x, y, _, _ = c.get_bbox()

    x1, y1, x2, y2 = spell.get_bbox()
    # numpy arrays are stored rows x columns, so flip x and y
    slot_img = img[y1-y:y2-y, x1-x:x2-x]

    # prepare path to save template into
    path = spell.resolve_path(
            root=join(dirname(__file__), '..'),
    )
    if not exists(dirname(path)):
        makedirs(dirname(path))

    # process and save the numpy array
    processed_img = spell.process_img(slot_img)

    numpy.save(path, processed_img)

    t = time.time() - t

    print(f'Got spellbook {spellbook} {spell.name} in {round(t, 2)} seconds')

