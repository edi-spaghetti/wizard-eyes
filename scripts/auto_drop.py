import time
import sys

from client import Client
from script_utils import safety_catch, logout


def main():
    c = Client('RuneLite')

    # setup inventory slots
    oak_longbow = 'oak_longbow_unstrung'
    items = [oak_longbow]
    for i in range(28):
        c.inventory.set_slot(i, items)

    msg_length = 100

    while True:

        # reset for logging
        t1 = time.time()
        sys.stdout.write('\b' * msg_length)
        msg = list()
        cool_down = 0.05

        if safety_catch(c, msg_length):
            continue

        # update
        img = c.screen.grab_screen(*c.get_bbox())
        inventory = c.inventory.identify(img)

        if oak_longbow in inventory:

            slot = c.inventory.first([oak_longbow], clicked=False)
            if slot:
                slot.click(tmin=0.6, tmax=0.9, shift=True)
                msg.append(f'Drop {oak_longbow} {slot.idx}')
            else:
                msg.append(f'Waiting Drop')
        else:
            msg.append('Nothing to drop')

        t1 = time.time() - t1
        msg.insert(0, f'Action: {t1:.2f}')
        msg = ' - '.join(msg)

        sys.stdout.write(f'{msg[:msg_length]:{msg_length}}')
        sys.stdout.flush()

        time.sleep(cool_down)


if __name__ == '__main__':
    main()
