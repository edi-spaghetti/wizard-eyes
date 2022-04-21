import sys
import time
import random

import keyboard
import pyautogui
import numpy


def safety_catch(c, msg_length):
    if not c.screen.on_off_state():
        msg = f'Sleeping @ {time.time()}'
        sys.stdout.write(f'{msg:{msg_length}}')
        sys.stdout.flush()
        time.sleep(0.1)
        return True
    elif keyboard.is_pressed('p'):
        exit(1)


def weighted_random(candidates, distances):
    """
    Pick a random item from a list of candidates, weighted by distance.
    Indexes of candidates and distances must exactly match.
    """
    inverse = [1 / d for d in distances]
    normalised = [i / sum(inverse) for i in inverse]
    cum_sum = numpy.cumsum(normalised)
    r = random.random()
    for i, val in enumerate(cum_sum):
        if val > r:
            return candidates[i]


def logout(c):

    # TODO: add logout buttons to client as proper game objects
    pause = random.random() * 60
    time.sleep(pause)

    bbox = c.get_bbox()
    pyautogui.moveTo(bbox[2] - 20, bbox[1] + 40)

    time.sleep(0.1)
    pyautogui.click()

    pause = random.random() * 5
    time.sleep(pause)
    pyautogui.moveTo(c.get_bbox()[2] - 100,
                     c.get_bbox()[3] - 80)

    time.sleep(0.1)
    pyautogui.click()

    sys.stdout.write('Logged out\n')
    sys.stdout.flush()

    exit()
