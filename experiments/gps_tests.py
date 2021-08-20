import time

import cv2
import numpy

from client import Client


LUMBRIDGE_ONLY = 1
GREATER_LUMBRIDGE = 2
GREATER_CHAOS_ALTAR = 3

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FILL = -1


def get_client():
    c = Client('RuneLite')
    c.activate()
    time.sleep(1)

    return c


def load_map_sections(client, code):

    if code == LUMBRIDGE_ONLY:
        sections = [['0_50_50']]
    elif code == GREATER_LUMBRIDGE:
        sections = [['0_49_51', '0_50_51', '0_51_51'],
                    ['0_49_50', '0_50_50', '0_51_50'],
                    ['0_49_49', '0_50_49', '0_51_49']]
    elif code == GREATER_CHAOS_ALTAR:
        sections = [['0_45_60', '0_46_60'],
                        ['0_45_59', '0_46_59']]
    else:
        raise NotImplementedError

    img = client.minimap.minimap.load_map_sections(sections)
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_grey


def get_minimap_imgs(client, apply_mask=True):

    mm = client.minimap.minimap

    img = client.screen.grab_screen(*client.minimap.minimap.get_bbox())
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    if apply_mask:
        mask = numpy.zeros_like(img_grey)

        my, mx, _ = mask.shape
        my //= 2
        mx //= 2

        radius = (mm.width // 2) - mm.config['padding']

        mask = cv2.circle(mask, (my, mx), radius, WHITE, FILL)

    else:
        mask = None

    return img_grey, mask
