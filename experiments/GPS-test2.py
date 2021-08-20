import sys
import itertools
import time

import cv2
import numpy

from client import Client

c = Client('RuneLite')
mm = c.minimap.minimap
msg_length = 50

# lumbridge castle
# sections = [['0_50_50']]
# greater lunbridge
sections = [['0_49_51', '0_50_51', '0_51_51'],
                ['0_49_50', '0_50_50', '0_51_50'],
                ['0_49_49', '0_50_49', '0_51_49']]

train_img = mm.load_map_sections(sections)
train_img_grey = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

apply_mask = True


while True:

    sys.stdout.write('\b' * msg_length)
    msg = list()

    query_img = c.screen.grab_screen(*mm.get_bbox())
    query_img_grey = cv2.cvtColor(query_img, cv2.COLOR_BGRA2GRAY)

    if apply_mask:
        mask = numpy.zeros_like(query_img_grey)
        mask = cv2.circle(mask, (mask.shape[0] // 2, mask.shape[1] // 2),
                                 (164 - 18) // 2, (255, 255, 255), -1)
        # masked_img = cv2.bitwise_and(query_img_grey, query_img_grey, mask=mask_circle)
    else:
        mask = None

    orb = cv2.ORB_create()

    # update
    t1 = time.time()
    kp1, des1 = orb.detectAndCompute(query_img_grey, mask)
    kp2, des2 = orb.detectAndCompute(train_img_grey, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    top_matches = matches[:10]
    coords = list()
    for m in top_matches:
        x1, y1 = kp1[m.queryIdx].pt
        x2, y2 = kp2[m.trainIdx].pt
        coords.append(((x1, y1), (x2, y2)))

    x_ratios = list()
    y_ratios = list()
    for (m0, m1) in itertools.combinations(coords, 2):
        (qx0, qy0), (tx0, ty0) = m0
        (qx1, qy1), (tx1, ty1) = m1
        try:
            x_ratio = abs(tx0 - tx1) / abs(qx0 - qx1)
            x_ratios.append(x_ratio)
        except ZeroDivisionError:
            pass
        try:
            y_ratio = abs(ty0 - ty1) / abs(qy0 - qy1)
            y_ratios.append(y_ratio)
        except ZeroDivisionError:
            pass

    try:
        median_x_ratio = x_ratios[len(x_ratios) // 2]
    except IndexError:
        median_x_ratio = 0.95
    try:
        median_y_ratio = y_ratios[len(y_ratios) // 2]
    except IndexError:
        median_y_ratio = 0.95

    # median_x_ratio = median_y_ratio = 0.95

    (bx0, by0), (bx1, by1) = coords[0]
    x = int((mm.config['width'] // 2 - bx0) * median_x_ratio + bx1)
    y = int((mm.config['height'] // 2 - by0) * median_y_ratio + by1)

    t1 = time.time() - t1
    msg.append(f'Update {t1:.2f}')

    img3 = cv2.drawMatches(query_img_grey, kp1, train_img_grey, kp2, top_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    map_display = train_img.copy()
    marked_map_display = cv2.circle(map_display, (x, y), 2, (255, 255, 255), -1)

    msg = ' - '.join(msg)
    sys.stdout.write(f'{msg[:msg_length]:{msg_length}}')
    sys.stdout.flush()

    cv2.imshow('Brute Force Matcher', img3)
    cv2.imshow('Map Coords', marked_map_display)
    cv2.waitKey(1)
