import sys
from os.path import join

import cv2
import numpy

from client import Client
from game_objects import GameObject

# define our game objects
c = Client('RuneLite')
ptile = GameObject(c, c)
folder = r'C:\Users\Edi\Desktop\programming\wizard-eyes\data\hit_splats'
msg_length = 100

# player character is always* in the middle of the game screen
x1, y1, x2, y2 = c.get_bbox()
x_m, y_m = (x1 + x2) / 2, (y1 + y2) / 2


# get the hit splat template images
blue_splat_colour = cv2.imread(join(folder, 'zero_splat.png'))
blue_splat_mask = cv2.imread(join(folder, 'zero_splat_mask.png'))
red_splat_colour = cv2.imread(join(folder, 'red_splat.png'))
red_splat_mask = cv2.imread(join(folder, 'red_splat_mask2.png'))

blue_splat_grey = cv2.cvtColor(blue_splat_colour, cv2.COLOR_BGRA2GRAY)
blue_splat_grey_mask = cv2.cvtColor(blue_splat_mask, cv2.COLOR_BGRA2GRAY)

red_splat_grey = cv2.cvtColor(red_splat_colour, cv2.COLOR_BGRA2GRAY)
red_splat_grey_mask = cv2.cvtColor(red_splat_mask, cv2.COLOR_BGRA2GRAY)

templates = ((blue_splat_grey, blue_splat_grey_mask, (255, 0, 0)),
             (red_splat_grey, red_splat_grey_mask, (0, 0, 255)))

# splat templates shoyuld be same size (24x24)
tx, ty = red_splat_grey.shape

# determine centre bound box as offset from middle
cx1, cy1, cx2, cy2 = int(x_m - 18), int(y_m - 17), int(x_m+29), int(y_m+30)
# ptile.set_aoi(int(cx1 - 11), cy1, cx2, int(cy2 + 11))

# determine larger bounding box
ptile.set_aoi(int(cx1 - 100), int(cy1 - 100), int(cx2 + 100), int(cy2 + 100))

i = 0

while True:
    try:

        sys.stdout.write('\b' * msg_length)
        msg = list()

        img = c.screen.grab_screen(*ptile.get_bbox())
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        # save frame (for testing / sampling)
        # c.screen.save_img(img, join(folder, f'img{i}.png'))

        # multi-matching
        for template, mask, colour in templates:
            matches = cv2.matchTemplate(
                img_grey, template,
                cv2.TM_CCOEFF_NORMED, mask=mask)
            (mx, my) = numpy.where(matches >= 0.99)
            for y, x in zip(mx, my):
                img = cv2.rectangle(
                    img, (x, y), (x + tx - 1, y + ty - 1), colour, 1)
        #
        #
        # # attempt to match for blue splat
        # match = cv2.matchTemplate(
        #     img_grey, blue_splat_grey, cv2.TM_CCOEFF_NORMED,
        #     mask=blue_splat_grey_mask)
        # _, max_match, _, (mx, my) = cv2.minMaxLoc(match)
        # if max_match > 0.99:
        #     # mx, my = match.shape
        #     img = cv2.rectangle(
        #         img,
        #         (mx, my),
        #         (mx + tx, my + ty),
        #         (0, 255, 0), 1)
        #     msg.append(f'Blue: {mx, my}, {max_match:.3f}')
        # else:
        #     msg.append(f'No Blue: {max_match:.3f}')
        #
        # match = cv2.matchTemplate(
        #     img_grey, red_splat_grey, cv2.TM_CCOEFF_NORMED,
        #     mask=red_splat_grey_mask)
        # _, max_match, _, (mx, my) = cv2.minMaxLoc(match)
        # if max_match > 0.99:
        #     # mx, my = match.shape
        #     img = cv2.rectangle(
        #         img,
        #         (mx, my),
        #         (mx + tx, my + ty),
        #         (0, 0, 255), 1)
        #     msg.append(f'Red: {mx, my}, {max_match:.3f}')
        # else:
        #     msg.append(f'No Red: {max_match:.3f}')

        msg = ' - '.join(msg)
        sys.stdout.write(f'{msg[:msg_length]:{msg_length}}')
        sys.stdout.flush()

        cv2.imshow('player_tile', img)
        cv2.waitKey(10)
        i += 1
    except KeyboardInterrupt:
        print('destroying windows')
        cv2.destroyAllWindows()
        break
