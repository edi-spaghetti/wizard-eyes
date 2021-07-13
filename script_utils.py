import sys
import time

import keyboard


def safety_catch(c, msg_length):
    if not c.screen.on_off_state():
        msg = f'Sleeping @ {time.time()}'
        sys.stdout.write(f'{msg:{msg_length}}')
        sys.stdout.flush()
        time.sleep(0.1)
        return True
    elif keyboard.is_pressed('p'):
        exit(1)
