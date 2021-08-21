import sys
import itertools
import time
from collections import defaultdict

import cv2
import numpy

from client import Client
from gps_tests import (
    BLACK,
    WHITE,
    GREATER_CHAOS_ALTAR,
    GREATER_LUMBRIDGE,
    LUMBRIDGE_ONLY,
    GHORROCK_TELEPORT,

    load_map_sections,
)


BY_DISTANCE = 0
MANUAL_FILTER = 1
TOP_X_RESULTS = 2
GROUPED_MATCHES = 3

MINI2MAP_RATIO = 0.95
GROUPED_MATCH_TOLERANCE = 2



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


def filter_matches(matches, train_img, query_img, kp1, kp2, code,
                   max_results=20, client=None):

    filtered_matches = matches
    client = client or Client('RuneLite')
    mm = client.minimap.minimap

    # filter matches by hand
    if code == MANUAL_FILTER:
        correct_matches = list()

        # make life easier on ourselves and only do the top 20 results
        # most of them seem to be false matches anyway
        top_matches = sorted(matches, key=lambda x: x.distance)[:max_results]

        for i, match in enumerate(top_matches):

            img = train_img.copy()
            img = cv2.drawMatches(query_img, kp1, img, kp2, [match],
                                   None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            win_name = f'Match {i} of {len(matches)}'
            cv2.imshow(win_name, img)
            key = cv2.waitKey(0)
            if key == 121:
                correct_matches.append(match)
            cv2.destroyWindow(win_name)

        filtered_matches = correct_matches

    elif code == BY_DISTANCE:
        filtered_matches = [m for m in matches if m.distance < 70]
    elif code == TOP_X_RESULTS:
        filtered_matches = sorted(matches, key=lambda x: x.distance)[:max_results]
    elif code == GROUPED_MATCHES:

        # pre-filter matches in case we get lots of poor matches
        filtered_matches = [m for m in matches if m.distance < 70]
        groups = defaultdict(list)
        for m in filtered_matches:
            # get source coords of minimap
            x0, y0 = kp1[m.queryIdx].pt
            # get corresponding coords from main map
            x1, y1 = kp2[m.trainIdx].pt
            x = int((mm.config['width'] / 2 - x0) * MINI2MAP_RATIO + x1)
            y = int((mm.config['height'] / 2 - y0) * MINI2MAP_RATIO + y1)

            new_key = (x, y)
            added = False
            items = list(groups.items())
            i = 0
            while i < len(items):
                (kx, ky), v = items[i]

                if added:
                    break

                tx0 = kx - GROUPED_MATCH_TOLERANCE
                tx1 = kx + GROUPED_MATCH_TOLERANCE
                ty0 = ky - GROUPED_MATCH_TOLERANCE
                ty1 = ky + GROUPED_MATCH_TOLERANCE

                in_bounds = (
                    tx0 <= x <= tx1 and
                    ty0 <= y <= ty1
                )
                if in_bounds:
                    # add the match to the existing key, we'll move it later
                    groups[(kx, ky)].append(m)

                    # create a new average center point between all existing points
                    avg_x = int((x + kx) // 2)
                    avg_y = int((y + ky) // 2)
                    new_key = (avg_x, avg_y)
                    if (kx, ky) != new_key:
                        for existing_match in groups[(kx, ky)]:
                            groups[new_key].append(existing_match)
                        del groups[(kx, ky)]
                    # if somehow the new average position is the same, then
                    # we're done because the current match was added earlier.

                    # set the flag so we stop iterating potential groups
                    added = True

                # increment our counter so we can check the next item
                i += 1

            if not added:
                # we must have a brand new group, so start a new list
                groups[new_key].append(m)

        # normalise the number of matches per group
        max_num_matches = max([len(v) for k, v in groups.items()], default=0)
        normalised_average = dict()
        for (k, v) in groups.items():
            average_distance = sum([m_.distance for m_ in v]) / len(v)
            normalised_average[k] = average_distance / client.screen.normalise(len(v), stop=max_num_matches)

        def test_sort_method(item):
            return item[1]

        sorted_normalised_average = sorted(
            [(k, v) for k, v in normalised_average.items()],
            # sort by normalised value, lower means more matches and lower
            key=test_sort_method)

        key, score = sorted_normalised_average[0]
        filtered_matches = groups[key]

    return filtered_matches


def main():

    c = Client('RuneLite')
    mm = c.minimap.minimap
    c.activate()
    msg_length = 200

    train_img_grey = load_map_sections(c, GREATER_CHAOS_ALTAR)

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

        filtered_matches = filter_matches(
            matches, train_img_grey, query_img_grey, kp1, kp2, GROUPED_MATCHES)

        msg.append(', '.join([f'{m.distance}' for m in filtered_matches]))

        # top_matches = top_matches[:3]
        # min_match = top_matches[0].distance
        # max_match = top_matches[-1].distance

        # msg.append(', '.join([f'{int(m.distance)}' for m in matches]))

        coords = list()
        for m in filtered_matches:
            # get source coords of minimap
            x1, y1 = kp1[m.queryIdx].pt
            # get corresponding coords from main map
            x2, y2 = kp2[m.trainIdx].pt
            coords.append(((x1, y1), (x2, y2)))

        # msg.append(', '.join([str(c) for c in coords]))
        #
        # (
        #     max_match_rect,
        #     min_avg_distance,
        #     best_match_subset,
        #     coords_subset
        # ) = find_best_match_subset(
        #     query_img, train_img, filtered_matches, kp1, kp2, c)
        #
        # # msg.append(f'matches: {len(best_match_subset)} @ {max_match_rect}')
        # #
        # median_x_ratio, median_y_ratio = calculate_median_ratios(coords_subset)

        (bx0, by0), (bx1, by1) = coords[0]
        x = int((mm.config['width'] / 2 - bx0) * MINI2MAP_RATIO + bx1) + mm.config['width']
        y = int((mm.config['height'] / 2 - by0) * MINI2MAP_RATIO + by1)

        t1 = time.time() - t1
        msg.append(f'Update {t1:.2f}')

        map_copy = train_img_grey.copy()
        img3 = map_copy
        # img3 = cv2.rectangle(map_copy, max_match_rect[0], max_match_rect[1], (255, 255, 255), 1)
        img3 = cv2.drawMatches(query_img_grey, kp1, img3, kp2, filtered_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # map_display = train_img.copy()
        img3 = cv2.circle(img3, (x, y), 4, WHITE, -1)
        img3 = cv2.circle(img3, (x, y), 2, BLACK, -1)

        msg.append(f'Position {(x, y)}')

        msg = ' - '.join(msg)
        sys.stdout.write(f'{msg[:msg_length]:{msg_length}}')
        sys.stdout.flush()

        cv2.imshow('Brute Force Matcher', img3)
        # cv2.imshow('Map Coords', marked_map_display)
        key = cv2.waitKey(0)

        # optionally reload a new map
        key_mapping = {
            49: GREATER_LUMBRIDGE,
            50: GREATER_CHAOS_ALTAR,
            51: LUMBRIDGE_ONLY,
            52: GHORROCK_TELEPORT,
        }
        if key_mapping.get(key):
            train_img_grey = load_map_sections(c, key_mapping[key])

        # cv2.destroyWindow('Brute Force Matcher')


if __name__ == '__main__':
    main()
