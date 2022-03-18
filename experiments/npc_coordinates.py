"""
An experiment trying to combine npc detection on the minimap with GPS
functionality. This will allow us to determine NPCs map position,
which we can use to track their movement across frames, so we can track state
e.g. combat status
"""
import time
import sys
from os.path import join
from uuid import uuid4

import cv2
import numpy

from client import Client
from game_objects import GameObject


class NPC(GameObject):

    def __init__(self, client, parent, name, v, w, x, y, z):
        super(NPC, self).__init__(client, parent)

        self.id = uuid4().hex
        self.key = v, w, x, y, z
        self.name = name
        self.updated_at = time.time()
        self.checked = False

    @property
    def mm_x(self):
        """top left X pixel on the minimap"""
        mm = self.client.minimap.minimap
        mm_x1 = mm.get_bbox()[0]
        x1 = self.client.get_bbox()[0]
        x = self.key[0]

        nx = int(mm_x1 - x1 + mm.config['width'] / 2 + x * mm.tile_size)
        return nx

    @property
    def mm_y(self):
        """top left Y pixel on the minimap"""
        mm = self.client.minimap.minimap
        mm_y1 = mm.get_bbox()[1]
        y1 = self.client.get_bbox()[1]
        y = self.key[1]

        ny = int(mm_y1 - y1 + mm.config['height'] / 2 + y * mm.tile_size)
        return ny

    def refresh(self):
        self.checked = False

    def update(self, key):
        self.key = key
        self.updated_at = time.time()
        self.checked = True


c = Client('RuneLite')
mm = c.minimap.minimap
msg_length = 200
folder = r'C:\Users\Edi\Desktop\programming\wizard-eyes\data\npc_minimap'
folder2 = r'C:\Users\Edi\Desktop\programming\wizard-eyes\data\main_window'

# load templates
player_marker = cv2.imread(join(folder2, 'player_marker.png'))
player_marker_mask = cv2.imread(join(folder2, 'player_marker_mask.png'))
player_marker_grey = cv2.cvtColor(player_marker, cv2.COLOR_BGRA2GRAY)
player_marker_mask = cv2.cvtColor(player_marker_mask, cv2.COLOR_BGRA2GRAY)

colour_mapping = {'npc': (255, 0, 0, 255), 'npc_tag': (0, 255, 0, 255)}

mm.create_map({(44, 153, 20), (44, 154, 20)})
mm.set_coordinates(37, 12, 44, 153, 20)

# determine centre bound box as offset from middle
x1, y1, x2, y2 = c.get_bbox()
x_m, y_m = (x1 + x2) / 2, (y1 + y2) / 2
cx1, cy1, cx2, cy2 = int(x_m - 29), int(y_m - 17), int(x_m+29), int(y_m+41)

t_width = 59 - 11
t_height = 59 - 11

save = False
images = list()
npcs = dict()

while True:

    sys.stdout.write('\b' * msg_length)
    msg = list()

    # get whole image
    img = c.screen.grab_screen(x1, y1, x2, y2)
    img_grey = mm.process_img(img)

    # first, find player tile
    # TODO: find player tile if prayer on
    # TODO: find player tile if moving
    p_img = img_grey[cy1-y1:cy2-y1+1, cx1-x1:cx2-x1+1]
    match = cv2.matchTemplate(
        p_img, player_marker_grey, cv2.TM_CCOEFF_NORMED,
        mask=player_marker_mask
    )
    _, max_match, _, (mx, my) = cv2.minMaxLoc(match)
    if max_match > 0.99:
        px = mx
        py = my
        msg.append(f'Match: {max_match:.2f}')
    else:
        msg.append(f'No match: {max_match:.2f}')
        px = py = None

    # identify npcs on minimap
    mm_x1, mm_y1, mm_x2, mm_y2 = mm.get_bbox()
    mm_img = img_grey[mm_y1-y1:mm_y2-y1+1, mm_x1-x1:mm_x2-x1+1]
    results = mm.identify(mm_img, threshold=0.99)
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
            n = NPC(c, c, name, *key)
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
            n = NPC(c, c, 'fail', 1, 2, 3, 4, 5)
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
        px1 = (cx1 - x1) + px + (t_width * x)
        py1 = (cy1 - y1) + py + (t_height * y)
        px2 = px1 + (t_width * 2) - x
        py2 = py1 + (t_height * 2) - y

        g_client = GameObject(c, c)
        g_client.set_aoi(0, 0, c.width, c.height)

        if g_client.is_inside(px1, py1) and g_client.is_inside(px2, py2):
            colour = colour_mapping.get(name)
            img = cv2.rectangle(img, (px1, py1), (px2, py2), colour, 1)

            # write the npc ID to inside the box
            cv2.putText(
                img, n.id[:8], (px1, py1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                colour, thickness=1)

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
