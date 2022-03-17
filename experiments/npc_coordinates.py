"""
An experiment trying to combine npc detection on the minimap with GPS
functionality. This will allow us to determine NPCs map position,
which we can use to track their movement across frames, so we can track state
e.g. combat status
"""
import sys
from os.path import join

import cv2
import numpy

from client import Client
from game_objects import GameObject

c = Client('RuneLite')
mm = c.minimap.minimap
msg_length = 100
folder = r'C:\Users\Edi\Desktop\programming\wizard-eyes\data\npc_minimap'
folder2 = r'C:\Users\Edi\Desktop\programming\wizard-eyes\data\main_window'

# load templates
player_marker = cv2.imread(join(folder2, 'player_marker.png'))
player_marker_mask = cv2.imread(join(folder2, 'player_marker_mask.png'))
player_marker_grey = cv2.cvtColor(player_marker, cv2.COLOR_BGRA2GRAY)
player_marker_mask = cv2.cvtColor(player_marker_mask, cv2.COLOR_BGRA2GRAY)

colour_mapping = {'npc': (255, 0, 0, 255), 'npc_tag': (0, 255, 0, 255)}

mm.create_map({(44, 153, 20), (44, 154, 20)})
mm.set_coordinates(37, 12, 44, 153, 20)

# determine centre bound box as offset from middle
x1, y1, x2, y2 = c.get_bbox()
x_m, y_m = (x1 + x2) / 2, (y1 + y2) / 2
cx1, cy1, cx2, cy2 = int(x_m - 29), int(y_m - 17), int(x_m+29), int(y_m+41)

t_width = 59 - 11
t_height = 59 - 11

save = False
images = list()

while True:

    sys.stdout.write('\b' * msg_length)
    msg = list()

    # get whole image
    img = c.screen.grab_screen(x1, y1, x2, y2)
    img_grey = mm.process_img(img)

    # first, find player tile
    p_img = img_grey[cy1-y1:cy2-y1+1, cx1-x1:cx2-x1+1]
    match = cv2.matchTemplate(
        p_img, player_marker_grey, cv2.TM_CCOEFF_NORMED,
        mask=player_marker_mask
    )
    _, max_match, _, (mx, my) = cv2.minMaxLoc(match)
    if max_match > 0.99:
        px = mx
        py = my
        msg.append(f'Match: {max_match:.2f}')
    else:
        msg.append(f'No match: {max_match:.2f}')
        px = py = None

    # identify npcs on minimap
    mm_x1, mm_y1, mm_x2, mm_y2 = mm.get_bbox()
    mm_img = img_grey[mm_y1-y1:mm_y2-y1+1, mm_x1-x1:mm_x2-x1+1]
    results = mm.identify(mm_img, threshold=0.99)
    msg.append(f'NPCs: {len(results)}')

    # get player coords on map
    coords = mm.run_gps()
    msg.append(f'Coords: {coords}')

    for name, x, y in results:

        # player marker may not have been found
        if px is None or py is None:
            continue

        # locate NPCs on mini map
        colour = colour_mapping.get(name, (255, 255, 255))
        nx = int(mm_x1-x1 + mm.config['width'] / 2 + x * mm.tile_size)
        ny = int(mm_y1-y1 + mm.config['height'] / 2 + y * mm.tile_size)
        img = cv2.rectangle(img, (nx, ny), (nx + 4, ny + 4), colour, 1)

        # locate NPCs on main screen
        _x1 = cx1 - x1
        _y1 = cy1 - y1
        px1 = (cx1 - x1) + px + (t_width * x)
        py1 = (cy1 - y1) + py + (t_height * y)
        px2 = px1 + (t_width * 2) - x
        py2 = py1 + (t_height * 2) - y

        g_client = GameObject(c, c)
        g_client.set_aoi(0, 0, c.width, c.height)

        if g_client.is_inside(px1, py1) and g_client.is_inside(px2, py2):
            colour = colour_mapping.get(name)
            img = cv2.rectangle(img, (px1, py1), (px2, py2), colour, 1)
            msg.append('1')
        else:
            msg.append('0')

    cv2.imshow('npc coords', img)
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
