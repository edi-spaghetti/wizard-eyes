from client import Client
import cv2
import keyboard
import sys
import time
from math import sqrt

def show(img):
    cv2.imshow('Test', img)
    cv2.waitKey(0)
    cv2.destroyWindow('Test')


key_mapping = {
    97: (-1, 0),   # a
    115: (0, 1),   # s
    100: (1, 0),   # d
    119: (0, -1),  # w
}

c = Client('RuneLite')
c.activate()
mm = c.minimap.minimap

mm.create_map({(50, 50, 0)})
mm.create_map({(45, 60, 0), (46, 59, 0)})

msg_length = 200

mm.set_coordinates(31, 31, 46, 60, 0)
# mm.set_coordinates(21, 45, 50, 50, 0)
while True:

    sys.stdout.write('\b' * msg_length)
    msg = list()

    t = time.time()

    tc = None
    coords = mm.run_gps(train_chunk=tc, show=1)
    msg.append(f'Coords: {coords}')

    t = time.time() - t
    msg.append(f'Update: {t:.2f}')

    # if keyboard.is_pressed('c'):
    #     mm.set_coordinates(31, 31, 46, 60, 0)
    # if keyboard.is_pressed('l'):
    #     mm.set_coordinates(21, 45, 50, 50, 0)
    # else:

    v0, w0, x0, y0, z0 = mm.get_coordinates()
    wx0 = (mm.max_tile + 1) * x0 + v0
    wy0 = (mm.max_tile + 1) * y0 + w0

    v1, w1, x1, y1, z1 = coords
    wx1 = (mm.max_tile + 1) * x1 + v1
    wy1 = (mm.max_tile + 1) * y1 + w1

    distance = int(sqrt(abs(wx0 - wx1) ** 2 + abs(wy0 - wy1) ** 2))
    msg.append(f'Distance: {distance:.2f}')

    if distance < 4:
        mm.set_coordinates(*coords)



    #
    # if mm._show_key == 113:
    #     cv2.destroyWindow('Position in Local Zone')
    #     break

    msg = ' - '.join(msg)
    sys.stdout.write(f'{msg[:msg_length]:{msg_length}}')
    sys.stdout.flush()

    time.sleep(0.05)
