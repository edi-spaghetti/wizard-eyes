import argparse
import time
from os.path import dirname, exists, join
from os import makedirs
import re
from shutil import rmtree, copy2

import cv2
import numpy

from wizard_eyes.client import Client
from wizard_eyes.file_path_utils import get_root


def sample(
        item_name: str,
        client: Client,
        copy_idx: int,
        args: argparse.Namespace,
        stackable: bool = False,
        charges: bool = False,
):
    t = time.time()

    # update so we get a fresh screen grab
    client.update()
    client.tabs.inventory.interface.update()

    stackable_mask = client.load_masks(['stackable'])['stackable']
    charges_mask = client.load_masks(['charges'])['charges']

    copy_icon = None
    for icon in client.tabs.inventory.interface.icons.values():
        i = int(re.match('[^0-9]*([0-9]+)$', icon.name).group(1))
        if icon.name == f'{icon.type}{copy_idx}':
            copy_icon = icon

        slot_img = client.get_img_at(icon.get_bbox(), mode=client.BGRA)

        # create a rough mask - will need to be verified & refined
        mask = cv2.inRange(slot_img, (38, 50, 59, 255), (46, 56, 66, 255))
        mask = cv2.bitwise_not(mask)
        if stackable:
            mask = cv2.bitwise_and(mask, stackable_mask)
        if charges:
            mask = cv2.bitwise_and(mask, charges_mask)

        # prepare path to save template into
        path = icon.resolve_path(root=get_root(), name='_')
        path = join(dirname(path), 'sample', str(i), f'{item_name}.npy')
        if not exists(dirname(path)):
            makedirs(dirname(path))

        # first save a colour copy for reference
        cv2.imwrite(path.replace('.npy', '.png'), slot_img)
        cv2.imwrite(path.replace('.npy', '_mask.png'), mask)
        # process and save the numpy array
        processed_img = icon.process_img(slot_img)
        numpy.save(path, processed_img)
        numpy.save(path.replace('.npy', '_mask.npy'), mask)

    if copy_icon is not None:
        copy_icon.load_templates([item_name])
        copy_icon.load_masks([item_name])

        for ext in ('.npy', '.png', '_mask.npy', '_mask.png'):
            target = copy_icon.resolve_path(
                root=get_root(), name=item_name).replace('.npy', ext)
            source = join(
                dirname(target), 'sample', str(copy_idx),
                f'{item_name}{ext}')

            if exists(target) and not args.overwrite:
                print(f'{target} already exists, skipping')
                continue

            if args.copy_to_inventory:
                copy2(source, target)

            if args.copy_to_bank:
                bank_target = target.replace(
                    'tabs', 'bank').replace(
                    'inventory', 'tab0')
                copy2(source, bank_target)
            if args.copy_to_equipment:
                equipment_target = target.replace('inventory', 'equipment')
                copy2(source, equipment_target)

    t = time.time() - t

    print(f'Got slots in {round(t, 2)} seconds')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--clear', action='store_true', default=False)
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument(
        '--copy-to-inventory', action='store_true', default=False)
    parser.add_argument('--copy-to-bank', action='store_true', default=False)
    parser.add_argument(
        '--copy-to-equipment', action='store_true', default=False)
    args = parser.parse_args()

    c = Client('RuneLite')

    c.update()
    c.tabs.inventory.auto_locate = True
    c.tabs.inventory.interface.create_template_groups_from_alpha_mapping([])

    c.tabs.inventory.update()

    for i, icon in enumerate(c.tabs.inventory.interface.icons.values()):
        path = icon.resolve_path(root=get_root(), name='_')
        path = dirname(path)
        path = join(path, 'sample', str(i))

        if args.clear and exists(path):
            rmtree(path)

    if not c.tabs.inventory.interface.located:
        print('Failed to locate inventory interface')
        return

    print('Press enter on blank item to exit')
    print('Usage: <item_name> [index] [-s | -stackable] [-c | -charges]')
    print('Example normal: "dragon_bones"')
    print('Example copy to inventory: "dragon_bones 1"')
    print('Example stackable: "rune_essence_noted 15 -s"')
    print('Example charges: "slayer_gem 27 -c"')
    while True:
        item_name = input('Item Name: ')
        idx = None
        stackable = False
        charges = False
        match = re.match(
            '([^\s]+)\s*([0-9]+)?\s*(-stackable|-s)?\w*(-charges|-c)?',
            item_name
        )
        if match:
            item_name = match.group(1).strip()
            if match.group(2):
                idx = int(match.group(2))
            if match.group(3):
                stackable = True
            if match.group(4):
                charges = True

        if item_name:
            sample(item_name, c, idx, args, stackable, charges)
        else:
            break


if __name__ == '__main__':
    main()
