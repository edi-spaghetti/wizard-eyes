import sys
import itertools
import time

import cv2
import numpy

from client import Client


def find_best_match_subset(query_img, train_img, matches, kp1, kp2, client):

    # TODO: vectorise this for speed
    # iterate over entire map and attempt to find the minimap section with the
    # most matches

    mm = client.minimap.minimap

    max_match_rect = None
    min_avg_distance = float('inf')
    best_match_subset = None
    coords_subset = list()

    for slide_y in range(train_img.shape[0] - query_img.shape[0] * mm.scale):
        for slide_x in range(train_img.shape[1] - query_img.shape[1] * mm.scale):

            # calculate bounding box of minimap at current xy
            sx0 = slide_x
            sy0 = slide_y
            sx1 = sx0 + query_img.shape[1] * mm.scale
            sy1 = sy0 + query_img.shape[0] * mm.scale

            # check each coordinate to see if it's within bounds
            matches_subset = list()
            for m in matches:
                # get source coords of minimap
                cx0, cy0 = kp1[m.queryIdx].pt
                # get corresponding coords from main map
                cx1, cy1 = kp2[m.trainIdx].pt
            # for ((cx0, cy0), (cx1, cy1)) in coords:
                in_bounds = (
                    sx0 < cx1 < sx1 and
                    sy0 < cy1 < sy1
                )
                if in_bounds:
                    matches_subset.append(m)
            try:
                avg_distance = sum([
                    # was normalising here, but it wasn't doing much, so I've
                    # removed it for simplicity
                    m.distance
                    for m in matches_subset]) / (
                        # factor the number of matches, so we're weighted
                        # towards more matches. Otherwise we also ways end up
                        # with wherever we can get the best match on it's own
                        # len(matches_subset) * (len(matches_subset) / 2)
                        len(matches_subset) ** 2
                )
                if avg_distance < min_avg_distance:
                    max_match_rect = ((sx0, sy0), (sx1, sy1))
                    min_avg_distance = avg_distance
                    best_match_subset = matches_subset

                    coords_subset = list()
                    for m in matches:
                        # get source coords of minimap
                        cx0, cy0 = kp1[m.queryIdx].pt
                        # get corresponding coords from main map
                        cx1, cy1 = kp2[m.trainIdx].pt
                        coords_subset.append(((cx0, cy0), (cx1, cy1)))

            except ZeroDivisionError:
                pass

    return max_match_rect, min_avg_distance, best_match_subset, coords_subset


def calculate_median_ratios(coords_subset):
    x_ratios = list()
    y_ratios = list()
    for (m0, m1) in itertools.combinations(coords_subset, 2):
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

    return median_x_ratio, median_y_ratio


def main():

    c = Client('RuneLite')
    mm = c.minimap.minimap
    c.activate()
    msg_length = 200

    # lumbridge castle
    # sections = [['0_50_50']]
    # greater lunbridge
    sections = [['0_49_51', '0_50_51', '0_51_51'],
                    ['0_49_50', '0_50_50', '0_51_50'],
                    ['0_49_49', '0_50_49', '0_51_49']]

    train_img = mm.load_map_sections(sections)
    train_img_grey = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

    apply_mask = True

    time.sleep(1)

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

        # top_matches = matches
        top_matches = [m for m in matches if m.distance < 70]
        # top_matches = top_matches[:3]
        min_match = top_matches[0].distance
        max_match = top_matches[-1].distance

        # msg.append(', '.join([f'{int(m.distance)}' for m in matches]))

        coords = list()
        for m in top_matches:
            # get source coords of minimap
            x1, y1 = kp1[m.queryIdx].pt
            # get corresponding coords from main map
            x2, y2 = kp2[m.trainIdx].pt
            coords.append(((x1, y1), (x2, y2)))

        # msg.append(', '.join([str(c) for c in coords]))

        (
            max_match_rect,
            min_avg_distance,
            best_match_subset,
            coords_subset
        ) = find_best_match_subset(
            query_img, train_img, top_matches, kp1, kp2, c)

        msg.append(f'matches: {len(best_match_subset)} @ {max_match_rect}')

        median_x_ratio, median_y_ratio = calculate_median_ratios(coords_subset)

        (bx0, by0), (bx1, by1) = coords_subset[0]
        x = int((mm.config['width'] // 2 - bx0) * median_x_ratio + bx1)
        y = int((mm.config['height'] // 2 - by0) * median_y_ratio + by1)

        t1 = time.time() - t1
        msg.append(f'Update {t1:.2f}')

        map_copy = train_img_grey.copy()
        img3 = cv2.rectangle(map_copy, max_match_rect[0], max_match_rect[1], (255, 255, 255), 1)
        img3 = cv2.drawMatches(query_img_grey, kp1, img3, kp2, top_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        map_display = train_img.copy()
        marked_map_display = cv2.circle(map_display, (x, y), 2, (255, 255, 255), -1)

        msg = ' - '.join(msg)
        sys.stdout.write(f'{msg[:msg_length]:{msg_length}}')
        sys.stdout.flush()

        cv2.imshow('Brute Force Matcher', img3)
        cv2.imshow('Map Coords', marked_map_display)
        cv2.waitKey(1)



if __name__ == '__main__':
    main()
