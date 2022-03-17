"""
Trying to determine how to convert (mini)map coordinates to on-screen position.
For example we know an npc is (3, 4) relative to the player. We want to convert
that so we can find the bounding box of that map position on the main screen.
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
folder = r'C:\Users\Edi\Desktop\programming\wizard-eyes\data\main_window'

# load templates
player_marker = cv2.imread(join(folder, 'player_marker.png'))
player_marker_mask = cv2.imread(join(folder, 'player_marker_mask.png'))
player_marker_grey = cv2.cvtColor(player_marker, cv2.COLOR_BGRA2GRAY)
player_marker_mask = cv2.cvtColor(player_marker_mask, cv2.COLOR_BGRA2GRAY)

print(player_marker_grey.shape, player_marker_mask.shape)

mm.create_map({(44, 153, 20), (44, 154, 20)})
mm.set_coordinates(37, 12, 44, 153, 20)

# determine centre bound box as offset from middle
x1, y1, x2, y2 = c.get_bbox()
x_m, y_m = (x1 + x2) / 2, (y1 + y2) / 2
cx1, cy1, cx2, cy2 = int(x_m - 29), int(y_m - 17), int(x_m+29), int(y_m+41)

px = py = None
t_width = 59 - 11
t_height = 59 - 11

save = False
images = list()

# coordinates relative to player
# we will try to capture these on the main screen
rx = 0
ry = 0
# offsets to calculate how much it changes
ox = 0
oy = 0

while True:

    sys.stdout.write('\b' * msg_length)
    msg = list()

    # first, find player tile
    img = c.screen.grab_screen(cx1, cy1, cx2, cy2)
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    match = cv2.matchTemplate(
        img_grey, player_marker_grey, cv2.TM_CCOEFF_NORMED,
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

    if px is not None and py is not None:
        px1 = cx1 + px + (t_width * rx)
        py1 = cy1 + py + (t_height * ry)
        px2 = px1 + t_width - rx
        py2 = py1 + t_height - ry
        msg.append(f'New Img: {px1, py1, px2, py2}')
        img = c.screen.grab_screen(px1, py1, px2, py2)

    cv2.imshow('map to main', img)
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
    else:
        # WASD to move coordinates
        if k == 115:
            ry += 1
        elif k == 119:
            ry -= 1
        elif k == 97:
            rx -= 1
        elif k == 100:
            rx += 1

    msg.append(f'Img: {img.shape}')
    msg.append(f'Rel: {rx, ry}')

    msg = ' - '.join(msg)
    sys.stdout.write(f'{msg[:msg_length]:{msg_length}}')
    sys.stdout.flush()
