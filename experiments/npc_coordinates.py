"""
An experiment trying to combine npc detection on the minimap with GPS
functionality. This will allow us to determine NPCs map position,
which we can use to track their movement across frames, so we can track state
e.g. combat status
"""
import sys
import time

import cv2

from client import Client


def main():

    c = Client('RuneLite')
    mm = c.minimap.minimap
    msg_length = 200
    folder = r'C:\Users\Edi\Desktop\programming\wizard-eyes\data\npc_minimap'

    mm.create_map({(44, 153, 20), (44, 154, 20)})
    mm.set_coordinates(37, 12, 44, 153, 20)
    mm.load_templates(['npc_tag'])
    mm.load_masks(['npc_tag'])

    save = False
    images = list()

    while True:

        t = time.time()
        sys.stdout.write('\b' * msg_length)
        msg = list()

        player = c.game_screen.player
        c.update()
        player.update()
        mm.update()

        # TODO: design method to add tile base on construction
        for icon in mm._icons.values():
            icon.tile_base = 2

        if c.args.show:
            cv2.imshow('Client', c.original_img)
            k = cv2.waitKey(5)
            if k == 27:
                if save:
                    print(f'Saving {len(images)} images')
                    for i_, image in enumerate(images):
                        path = f'{folder}/img{i_}.png'
                        c.screen.save_img(image, path)
                print('destroying windows')
                cv2.destroyAllWindows()
                break

        t = time.time() - t
        msg.append(f'Update: {t:.3f}')

        msg = ' - '.join(msg)
        sys.stdout.write(f'{msg[:msg_length]:{msg_length}}')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
