import time
import sys

import keyboard

import client
from game_objects import GameObject
from script_utils import safety_catch

c = client.Client('RuneLite')

mage_protect_aoi = c.screen.gen_bbox()
mage_protect = GameObject(c, c)
mage_protect.set_aoi(*mage_protect_aoi)
keyboard.on_press_key('u', lambda _: mage_protect.click(tmin=0.2, tmax=0.3))

range_protect_aoi = c.screen.gen_bbox()
range_protect = GameObject(c, c)
range_protect.set_aoi(*range_protect_aoi)
keyboard.on_press_key('i', lambda _: range_protect.click(tmin=0.2, tmax=0.3))

melee_protect_aoi = c.screen.gen_bbox()
melee_protect = GameObject(c, c)
melee_protect.set_aoi(*melee_protect_aoi)
keyboard.on_press_key('o', lambda _: melee_protect.click(tmin=0.2, tmax=0.3))

msg_length = 100
cool_down = 0.05

print('Entering Main Loop')
while True:

    # reset for logging
    t1 = time.time()
    sys.stdout.write('\b' * msg_length)
    msg = list()
    current_prayer = None

    current_tick = time.time() // 0.6
    sub_tick = time.time() % 0.6

    if safety_catch(c, msg_length):
        continue

    mage_protect.update()
    range_protect.update()
    melee_protect.update()

    msg = ' - '.join(msg)

    sys.stdout.write(f'{msg[:msg_length]:{msg_length}}')
    sys.stdout.flush()

    time.sleep(cool_down)
