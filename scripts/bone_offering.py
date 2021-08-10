import sys
import time

import keyboard
import pyautogui

import client
from game_objects import GameObject
from script_utils import safety_catch


def main():

    # setup
    c = client.Client('RuneLite')

    # set up item names
    bone = 'dragon_bone'
    bone_selected = f'{bone}_selected'

    # set up inventory slots
    items = [bone, bone_selected]
    for i in range(28):
        c.inventory.set_slot(i, items)

    # set up altar (note, it is a variable)
    altar = GameObject(c, c)

    # get logout buttons
    pm = c.personal_menu
    logout_menu = pm.get_menu(pm.LOGOUT)
    ws = pm.get_menu(pm.WORLD_SWITCHER)
    logout_button = logout_menu.logout_button
    ws_logout = ws.logout_button

    # logging
    msg_length = 100
    t3 = time.time()

    # main loop
    print('Entering Main Loop')
    c.activate()

    while True:

        # reset for logging
        t1 = time.time()
        sys.stdout.write('\b' * msg_length)
        msg = list()

        if keyboard.is_pressed('l'):

            msg.append('Logout!')

            img = c.screen.grab_screen(*logout_button.get_bbox())
            logout_menu_open = (logout_button.identify(img) in
                                {'logout', 'logout_hover'})
            img = c.screen.grab_screen(*ws_logout.get_bbox())
            ws_menu_open = (ws_logout.identify(img) in
                            {'logout', 'logout_hover'})

            if logout_menu_open:
                logout_button.click(tmin=0.01, tmax=0.1)
            elif ws_menu_open:
                ws_logout.click(tmin=0.01, tmax=0.1)
            else:
                if c.logout_button.clicked:
                    msg.append(f'Wait Logout Menu '
                               f'{c.logout_button.time_left}')
                else:
                    msg.append(f'Click Logout Tab')
                    c.logout_button.click(tmin=0.1, tmax=0.2)

            msg = ' - '.join(msg)

            sys.stdout.write(f'{msg[:msg_length]:{msg_length}}')
            sys.stdout.flush()

            altar.clear_bbox()
            continue

        if safety_catch(c, msg_length):
            altar.clear_bbox()
            continue

        # update
        img = c.screen.grab_screen(*c.get_bbox())
        inventory = c.inventory.identify(img)
        for i in range(len(inventory)):
            c.inventory.slots[i].update()
        if altar.get_bbox() is None:
            # get current mouse position
            mp = pyautogui.position()
            x, y = mp.x, mp.y
            # expand for bbox
            size = 20
            x1 = x - size
            y1 = y - size
            x2 = x + size
            y2 = y + size
            # set the new bbox
            altar.set_aoi(x1, y1, x2, y2)

        t1 = time.time() - t1
        msg.append(f'Update {t1:.2f}')

        # action
        t2 = time.time()
        cool_down = 0.05

        if bone_selected in inventory:
            if altar.clicked:
                msg.append(f'Wait Altar {altar.time_left}')
            else:
                altar.click(tmin=0.1, tmax=0.3, pause_before_click=True)
        elif bone in inventory:
            slot = c.inventory.first({bone})
            if slot.clicked:
                msg.append(f'Wait {bone} in slot {slot.idx} {slot.time_left}')
            else:
                slot.click(tmin=0.1, tmax=0.3, shift=True, pause_before_click=True)
                msg.append(f'Clicked {bone} at index {slot.idx}')
        else:
            msg.append('No bones left')

        t2 = time.time() - t2
        msg.insert(1, f'Action {t2:.2f}')
        msg.insert(2, f'Runtime {time.time() - t3:.2f}')
        msg = ' - '.join(msg)

        sys.stdout.write(f'{msg[:msg_length]:{msg_length}}')
        sys.stdout.flush()

        time.sleep(cool_down)


if __name__ == '__main__':
    main()
