"""
An experiment trying to combine npc detection on the minimap with GPS
functionality. This will allow us to determine NPCs map position,
which we can use to track their movement across frames, so we can track state
e.g. combat status
"""
import sys

import cv2

from client import Client
from game_objects import GameObject

c = Client('RuneLite')
mm = c.minimap.minimap
msg_length = 100
folder = r'C:\Users\Edi\Desktop\programming\wizard-eyes\data\npc_minimap'

mm.create_map({(44, 153, 20), (44, 154, 20)})
mm.set_coordinates(37, 12, 44, 153, 20)

save = False
images = list()

while True:

    sys.stdout.write('\b' * msg_length)
    msg = list()

    img = c.screen.grab_screen(*c.minimap.minimap.get_bbox())
    img_grey = mm.process_img(img)

    results = mm.identify(img_grey, threshold=0.99)
    msg.append(f'NPCs: {len(results)}')

    coords = mm.run_gps()
    msg.append(f'Coords: {coords}')

    for name, x, y in results:
        colour = {'npc': (255, 0, 0, 255), 'npc_tag': (0, 255, 0, 255)}.get(name)
        px = int(mm.config['width'] / 2 + x * mm.tile_size)
        py = int(mm.config['height'] / 2 + y * mm.tile_size)
        img = cv2.rectangle(img, (px, py), (px + 4, py + 4), colour, 1)

    cv2.imshow('npc coords', img)
    k = cv2.waitKey(5)
    if k == 27 and save:
        print(f'Saving {len(images)} images')
        for i_, image in enumerate(images):
            path = f'{folder}/img{i_}.png'
            c.screen.save_img(image, path)
        print('destroying windows')
        cv2.destroyAllWindows()
        break

    msg = ' - '.join(msg)
    sys.stdout.write(f'{msg[:msg_length]:{msg_length}}')
    sys.stdout.flush()
