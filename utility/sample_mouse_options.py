import time
from os.path import join, dirname, exists
from os import makedirs

import cv2
import numpy

from wizard_eyes.client import Client


def sample(name: str, client: Client):

    # grab whole screen and sample from it for each slot
    img = client.screen.grab_screen(*client.get_bbox())
    # we need client bbox to zero the slot coordinates
    x, y, _, _ = client.get_bbox()

    t = time.time()

    for i in range(28):

        x1, y1, x2, y2 = client.mouse_options.get_bbox()
        x1, y1, x2, y2 = client.localise(x1, y1, x2, y2)
        # numpy arrays are stored rows x columns, so flip x and y
        slot_img = img[y1:y2, x1:x2]

        # prepare path to save template into

        path = client.mouse_options.PATH_TEMPLATE.format(
            root=join(dirname(__file__), '..'),
            name=name
        )
        if not exists(dirname(path)):
            makedirs(dirname(path))

        # first save a colour copy for reference
        cv2.imwrite(path.replace('.npy', '.png'), slot_img)
        # process and save the numpy array
        processed_img = client.mouse_options.process_img(slot_img)
        numpy.save(path, processed_img)

    t = time.time() - t

    print(f'Got mouse option in {round(t, 2)} seconds')


def main():

    c = Client('RuneLite')

    print('Press enter on blank item to exit')
    while True:
        item_name = input('Item Name: ')
        if item_name:
            sample(item_name, c)
        else:
            break


if __name__ == '__main__':
    main()
