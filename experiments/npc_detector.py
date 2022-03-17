import sys
from os.path import join

import cv2
import numpy

from client import Client
from game_objects import GameObject


c = Client('RuneLite')
ptile = GameObject(c, c)
folder = r'C:\Users\Edi\Desktop\programming\wizard-eyes\data\npc_minimap'
msg_length = 100

# player character is always* in the middle of the game screen
x1, y1, x2, y2 = c.get_bbox()
x_m, y_m = (x1 + x2) / 2, (y1 + y2) / 2


# load npc templates
tx, ty = 4, 5
colour_mapping = {'npc': (0, 0, 255, 255), 'npc_tag': (0, 255, 0, 255), 'npc2': (255, 0, 0, 255)}
templates = list()
for template in (
        'npc',
        # 'npc2',
        'npc_tag',
):
    template_colour = cv2.imread(join(folder, f'{template}.png'))
    template_grey = cv2.cvtColor(template_colour, cv2.COLOR_BGRA2GRAY)
    tx, ty = template_grey.shape

    mask = cv2.imread(join(folder, f'{template}_mask.png'))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)

    # create npy templates
    numpy.save(join(folder, f'{template}.npy'), template_grey)
    numpy.save(join(folder, f'{template}_mask.npy'), mask)

    templates.append((template_grey, mask, colour_mapping[template]))


# determine centre bound box as offset from middle
cx1, cy1, cx2, cy2 = int(x_m - 18), int(y_m - 17), int(x_m+29), int(y_m+30)
# ptile.set_aoi(int(cx1 - 11), cy1, cx2, int(cy2 + 11))

# determine larger bounding box
ptile.set_aoi(int(cx1 - 300), int(cy1 - 200), int(cx2 + 300), int(cy2 + 200))

i = 0
images = list()
save = True


while True:
    sys.stdout.write('\b' * msg_length)
    msg = list()

    img = c.screen.grab_screen(*c.minimap.minimap.get_bbox())
    images.append(img)

    # img = c.screen.grab_screen(*ptile.get_bbox())
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    marked = set()

    # multi-matching
    for template, mask, colour in templates:
        matches = cv2.matchTemplate(
            img_grey, template,
            cv2.TM_CCOEFF_NORMED)  # , mask=mask)
        # _, max_match, _, (mx, my) = cv2.minMaxLoc(matches)
        # if max_match > .7:
        #     img = cv2.rectangle(img, (mx, my), (mx+tx, my+ty), (255, 255, 255), 1)
        (my, mx) = numpy.where(matches >= 0.99)
        for y, x in zip(my, mx):
            matched_img = img_grey[x:x + tx - 1,y:y + ty - 1]
            condition = (
                    # numpy.any(matched_img)
                    # and
                    (x, y) not in marked
            )
            if condition:
                marked.add((x, y))
                img = cv2.rectangle(
                    img, (x, y), (x + tx - 1, y + ty - 1), colour, 1)
                # cv2.imshow('minimap', img)
                # cv2.waitKey(0)

    msg = ' - '.join(msg)
    sys.stdout.write(f'{msg[:msg_length]:{msg_length}}')
    sys.stdout.flush()

    cv2.imshow('minimap', img)
    i += 1

    k = cv2.waitKey(5)
    if k == 27 and save:
        print(f'Saving {len(images)} images')
        for i_, image in enumerate(images):
            path = f'{folder}/img{i_}.png'
            c.screen.save_img(image, path)
        print('destroying windows')
        cv2.destroyAllWindows()
        break
