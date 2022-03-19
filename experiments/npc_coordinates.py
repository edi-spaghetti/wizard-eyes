"""
An experiment trying to combine npc detection on the minimap with GPS
functionality. This will allow us to determine NPCs map position,
which we can use to track their movement across frames, so we can track state
e.g. combat status
"""
import sys

import cv2
import numpy

from client import Client
from game_objects import GameObject
from game_screen import NPC


def main():

    c = Client('RuneLite')
    mm = c.minimap.minimap
    msg_length = 200
    folder = r'C:\Users\Edi\Desktop\programming\wizard-eyes\data\npc_minimap'
    folder2 = r'C:\Users\Edi\Desktop\programming\wizard-eyes\data\main_window'

    colour_mapping = {'npc': (255, 0, 0, 255), 'npc_tag': (0, 255, 0, 255)}
    combat_mapping = {
        -1: 'unknown',
        0: 'no combat',
        1: 'combat',
        2: 'combat 2',
    }

    mm.create_map({(44, 153, 20), (44, 154, 20)})
    mm.set_coordinates(37, 12, 44, 153, 20)

    # determine centre bound box as offset from middle
    x1, y1, x2, y2 = c.get_bbox()
    x_m, y_m = (x1 + x2) / 2, (y1 + y2) / 2

    t_width = 59 - 11
    t_height = 59 - 11

    save = False
    images = list()
    npcs = dict()

    while True:

        sys.stdout.write('\b' * msg_length)
        msg = list()

        player = c.game_screen.player
        cx1, cy1, cx2, cy2 = player.get_bbox()
        c.update()
        player.update()

        # get whole image
        img_grey = c.img
        img = c._original_img

        # first, find player tile
        # TODO: find player tile if prayer on
        # TODO: find player tile if moving
        px, py, _px2, _py2 = player.tile_bbox()

        # convert relative to static bbox so we can use later
        px = px - cx1 + 1
        py = py - cy1 + 1
        _px2 = _px2 - cx1 + 1
        _py2 = _py2 - y1 + 1

        # identify npcs on minimap
        results = mm.identify(threshold=0.99)
        msg.append(f'NPCs: {len(results)}')

        # get player coords on map
        coords = mm.run_gps()
        msg.append(f'Coords: {coords}')

        # reset marker on NPCs so we know which ones we've checked
        for n in npcs.values():
            n.refresh()

        # compare existing npcs to what we found
        checked = set()
        created = 0
        exact = 0
        adjacent = 0
        for name, pixel_x, pixel_y, tile_x, tile_y in results:

            # TODO: set up methods to calculate coordinate across chunks
            # assume we are within the same chunk
            v, w, X, Y, Z = coords

            # key by pixel
            # key = (pixel_x, pixel_y, X, Y, Z)
            # key by tile coordinate
            key = (tile_x, tile_y, X, Y, Z)

            added_on_adjacent = False
            try:
                n = npcs[key]
                n.update(key)
                checked.add(key)
                exact += 1
                continue
            except KeyError:
                npc_copy = [n.key for n in npcs.values()]
                max_dist = 1
                for npc_key in npc_copy:
                    if (abs(tile_x - npc_key[0]) <= max_dist and
                            abs(tile_y - npc_key[1]) <= max_dist):
                        # move npc to updated key
                        _npc = npcs.pop(npc_key)
                        npcs[key] = _npc
                        _npc.update(key)
                        added_on_adjacent = True
                        adjacent += 1
                        continue
                #
                # # TODO: this doesn't seem to be working
                # # NPCs can move one tile per tick, so at most, they could have
                # # moved 1 tile away from their last known position
                # r = 6
                # for _ix in range(r):
                #     for _iy in range(r):
                #         ix = math.ceil(_ix - r / 2)
                #         iy = math.ceil(_iy - r / 2)
                #         if ix == 0 and iy == 0:
                #             # we don't need to check an exact match again
                #             continue
                #         ikey = (
                #             int(pixel_x + ix),
                #             int(pixel_y + iy),
                #             X, Y, Z)
                #         if ikey in npcs:
                #             n = npcs[ikey]
                #             n.update(ikey)
                #             checked.add(ikey)
                #             added_on_adjacent = True
                #             adjacent += 1
                #             continue

            # finally if we still can't find it, we must have a new one
            if key not in checked and not added_on_adjacent:
                n = NPC(c, c, name, *key, tile_base=2)
                n.update(key)
                npcs[key] = n
                created += 1

        # do one final check to remove any that are no longer on screen
        keys = list(npcs.keys())
        removed = 0
        for k in keys:
            n = npcs[k]
            if not n.checked:
                removed += 1
                npcs.pop(k)

        msg.append(f'NPCs ({len(npcs)}): {exact, adjacent, created, removed}')

        for name, pix_x, pix_y, x, y in results:

            # player marker may not have been found
            if px is None or py is None:
                continue

            # get npc from dict
            v, w, X, Y, Z = coords
            n = npcs.get((x, y, X, Y, Z))
            if not n:
                n = NPC(c, c, 'fail', 1, 2, 3, 4, 5, tile_base=2)
                n.id = 'fail'
            # name = n.name
            # x = n.key[0] // mm.tile_size
            # y = n.key[1] // mm.tile_size

            # locate NPCs on mini map
            colour = colour_mapping.get(name, (255, 255, 255))
            nx = n.mm_x
            ny = n.mm_y
            img = cv2.rectangle(img, (nx, ny), (nx + 4, ny + 4), colour, 1)

            # locate NPCs on main screen
            _x1 = cx1 - x1
            _y1 = cy1 - y1
            px1, py1, px2, py2 = n.ms_bbox()
            # convert back to relative to client bbox
            px1 = px1 - x1 + 1
            py1 = py1 - y1 + 1
            px2 = px2 - x1 + 1
            py2 = py2 - y1 + 1

            g_client = GameObject(c, c)
            g_client.set_aoi(0, 0, c.width, c.height)

            if g_client.is_inside(px1, py1) and g_client.is_inside(px2, py2):
                colour = colour_mapping.get(name)
                img = cv2.rectangle(img, (px1, py1), (px2, py2), colour, 1)

                # write the npc ID to inside the box
                cv2.putText(
                    img, n.id[:8], (px1, py1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colour, thickness=1)

                # write the npc combat status just under ID
                cv2.putText(
                    img, combat_mapping.get(n.combat_status, 'none'),
                    (px1, py1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                    colour, thickness=1
                )

                try:
                    hy, hx = n.get_hitbox()
                    hx1 = px1 + hx
                    hy1 = py1 + hy
                    cv2.circle(img, (hx1, hy1), 3, colour, thickness=1)
                    msg.append(f'{hx, hy} {(hx1, hy1)}')
                except TypeError:
                    msg.append('err')

                msg.append('1')
            else:
                msg.append('0')

        cv2.imshow('npc coords', img)
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

        msg = ' - '.join(msg)
        sys.stdout.write(f'{msg[:msg_length]:{msg_length}}')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
