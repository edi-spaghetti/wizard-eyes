"""
An experiment trying to combine npc detection on the minimap with GPS
functionality. This will allow us to determine NPCs map position,
which we can use to track their movement across frames, so we can track state
e.g. combat status
"""
import sys

import cv2
import numpy

from client import Client
from game_objects import GameObject

c = Client('RuneLite')
mm = c.minimap.minimap
msg_length = 100
folder = r'C:\Users\Edi\Desktop\programming\wizard-eyes\data\npc_minimap'

mm.create_map({(44, 153, 20), (44, 154, 20)})
mm.set_coordinates(37, 12, 44, 153, 20)

# determine centre bound box as offset from middle
x1, y1, x2, y2 = c.get_bbox()
x_m, y_m = (x1 + x2) / 2, (y1 + y2) / 2
cx1, cy1, cx2, cy2 = int(x_m - 29), int(y_m - 17), int(x_m+29), int(y_m+41)

t_width = 59
t_height = 59

save = False
images = list()

while True:

    sys.stdout.write('\b' * msg_length)
    msg = list()

    # identify npcs on minimap
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

    # locate NPCs on main screen
    images = [img]
    for name, x, y in results:
        rx = (cx2 - t_width) + (t_width * x)
        ry = (cy2 - t_height) + (t_height * y)

        g_client = GameObject(c, c)
        g_client.set_aoi(*c.get_bbox())

        if g_client.is_inside(rx, ry):
            s_img = c.screen.grab_screen(rx, ry, int(rx + t_width), int(ry + t_height))
            images.append(s_img)

    # concatenate images for display
    max_width = max([m.shape[1] for m in images])
    total_height = sum(map(lambda m: m.shape[0], images))
    final_image = numpy.zeros((total_height, max_width, 4), dtype=numpy.uint8)
    current_y = 0
    for image in images:
        final_image[current_y:image.shape[0]+current_y,:image.shape[1],:] = image
        current_y += image.shape[0]

    # img = c.screen.grab_screen(cx1 + 11, cy1, cx2, cy2 - 11)
    # tx = cx2 - cx1 + 1
    # ty = cy2 - cy1 + 1
    # msg.append(f'Tile: {tx, ty}')

    cv2.imshow('npc coords', final_image)
    k = cv2.waitKey(5)
    if k == 27:
        if save:
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
