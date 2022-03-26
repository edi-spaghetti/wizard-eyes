import time
from os.path import join, dirname

import cv2
import numpy


from wizard_eyes.client import Client
c = Client('RuneLite')
c.update()
print(c.get_bbox())


for t in c.tabs.DEFAULT_TABS:
    x1, y1, x2, y2 = c.get_bbox()

    cx1, cy1, cx2, cy2 = c.tabs.get_bbox()
    img = c.img[cy1 - y1:cy2 - y1 + 1, cx1 - x1:cx2 - x1 + 1]
    template = c.tabs.templates.get(t)
    if template is None:
        print(f'skipping {t}')
        continue

    # c.screen.show_img(img)
    # continue

    match = cv2.matchTemplate(
        img, template, cv2.TM_CCOEFF_NORMED,
        mask=c.tabs.masks.get('tab_mask')
    )
    _, confidence, _, (mx, my) = cv2.minMaxLoc(match)
    h, w = template.shape
    tx1 = mx
    ty1 = my
    tx2 = tx1 + w
    ty2 = ty1 + h

    print(f'{t}: {confidence}')
    # c.screen.show_img(img[ty1:ty2, tx1:tx2], 'template')

    # convert back to screen space
    sx1 = tx1 + cx1
    sy1 = ty1 + cy1
    sx2 = tx2 + cx1 - 1
    sy2 = ty2 + cy1 - 1
    # print(sx1, sy1, sx2, sy2)

    # show it on a new image so we can check we got the right coordinates
    # img2 = c.screen.grab_screen(sx1, sy1, sx2, sy2)
    # c.screen.show_img(img2)

    # click each tab, and wait a second for it to register
    c.screen.click_aoi(sx1, sy1, sx2, sy2)
    time.sleep(1)

    # grab image with the tab selected
    img3 = c.screen.grab_screen(sx1, sy1, sx2, sy2)
    img_grey = c.tabs.process_img(img3)
    c.screen.show_img(img3)

    # save it out to disk
    path = join(dirname(dirname(__file__)), 'data', 'tabs', f'{t}_selected.npy')
    # print(path)
    # numpy.save(path, img_grey)
