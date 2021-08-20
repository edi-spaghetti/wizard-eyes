from client import Client
import numpy
import cv2
import sys
import time
import os

c = Client('RuneLite')
root = os.path.join(os.path.dirname(__file__), '..', 'data')

mm = c.minimap.minimap

# build the map
# lumbridge
map_sections = [['0_49_51', '0_50_51', '0_51_51'],
                ['0_49_50', '0_50_50', '0_51_50'],
                ['0_49_49', '0_50_49', '0_51_49']]
# lumbridge castle only
# map_sections = [['0_50_50']]

# chaos alter
# map_sections = [['0_45_60', '0_46_60'],
#                 ['0_45_59', '0_46_59']]

map = numpy.concatenate(
    [numpy.concatenate(
        [cv2.imread(f'{root}/maps/{name}.png') for name in row], axis=1)
        for row in map_sections], axis=0)

# map = cv2.imread(f'{root}/maps/0_50_50.png')
map_bgra = cv2.cvtColor(map, cv2.COLOR_BGR2BGRA)
map_grey = cv2.cvtColor(map, cv2.COLOR_BGR2GRAY)
map_edged = cv2.Canny(map_grey, 50, 200)
cv2.imwrite(f'{root}/edged_map.png', map_edged)

# build the mask once, it can be re-used
img = c.screen.grab_screen(*mm.get_bbox())
grey = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
mask = numpy.zeros_like(grey)
mask_circle = cv2.circle(mask, (mask.shape[0] // 2, mask.shape[1] // 2),
                         (164 - 18) // 2, (255, 255, 255), -1)
masked_img = img.copy()
masked_img = cv2.bitwise_and(masked_img, masked_img, mask=mask_circle)
cv2.imwrite(f'{root}/masked_minimap.png', masked_img)

msg_length = 50


while True:

    sys.stdout.write('\b' * msg_length)
    msg = list()

    # sample the minimap
    img = c.screen.grab_screen(*mm.get_bbox())

    t1 = time.time()
    grey = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    BestMatchVal = float('inf')
    WorstMatchVal = -float('inf')
    BestMatchLoc = None
    WorstMatchLoc = None
    BestMatchScale = None
    WorstMatchScale = None
    # lumbridge works best at this scale
    # scale20 = numpy.linspace(0.85, 0.95, 5)
    scale20 = numpy.linspace(0.9, 1., 5)
    for scale in scale20:

        # scale the minimap screen grab
        resized_template = cv2.resize(
            grey,
            (int(img.shape[0] * scale), int(img.shape[1] * scale))
        )
        # make sure the mask is also scaled to same size
        mask_resized = cv2.resize(
            mask_circle,
            (int(mask_circle.shape[0] * scale), int(mask_circle.shape[1] * scale))
        )

        # keep track of the ratio, we'll need it later
        # r = grey.shape[1] / float(resized_template.shape[1])

        edged_template = cv2.Canny(resized_template, 50, 200)

        # edged template match
        result = cv2.matchTemplate(map_edged, edged_template, cv2.TM_CCORR_NORMED, None, mask_resized)
        # grey template match
        # result = cv2.matchTemplate(map_grey, resized_template, cv2.TM_CCORR_NORMED, None, mask_resized)
        # colour match
        # result = cv2.matchTemplate(map_bgra, resized_template, cv2.TM_CCORR_NORMED, None, mask_resized)

        # cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)
        _minVal, _maxVal, minLoc, maxLoc = cv2.minMaxLoc(result) # , None)
        matchLoc = minLoc

        if BestMatchVal > _minVal:
            BestMatchLoc = matchLoc
            BestMatchVal = _minVal
            BestMatchScale = scale
        if WorstMatchVal < _maxVal:
            WorstMatchVal = _maxVal
            WorstMatchLoc = maxLoc
            WorstMatchScale = scale

    t1 = time.time() - t1
    msg.append(f'Update {t1:.2f}')

    msg.append(f'Scale {BestMatchScale}')

    mx, my = BestMatchLoc
    px, py = (int(mx + (grey.shape[0] * BestMatchScale) // 2),
              int(my + (grey.shape[1] * BestMatchScale) // 2))
    msg.append(f'Coords ({px}, {py})')

    map_display = map_bgra.copy()
    marked_map_display = cv2.circle(
        map_display,
        (px, py),
        2, (255, 255, 255), -1
    )
    # cv2.imwrite(f'{root}/gps.png', marked_map_display)

    mx, my = WorstMatchLoc
    px, py = (int(mx + (grey.shape[0] * WorstMatchScale) // 2),
              int(my + (grey.shape[1] * WorstMatchScale) // 2))

    map_display = map_grey.copy()
    marked_map_display = cv2.circle(
        marked_map_display,
        (px, py),
        2, (0, 0, 0), -1
    )
    # cv2.imwrite(f'{root}/gps.png', marked_map_display)

    msg = ' - '.join(msg)
    sys.stdout.write(f'{msg[:msg_length]:{msg_length}}')
    sys.stdout.flush()

    cv2.imshow("Test", marked_map_display)
    cv2.waitKey(1)
