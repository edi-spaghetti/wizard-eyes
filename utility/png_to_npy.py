import os
import argparse

import cv2
import numpy


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    parser.add_argument('files', nargs='+')
    parser.add_argument('--bgra-2-gray', action='store_true', default=False)
    args = parser.parse_args()

    if args.files == ['*']:
        files = list()
        for f in os.listdir(args.dir):
            if os.path.isfile(f'{args.dir}/{f}') and f.endswith('.png'):
                f, _ = os.path.splitext(f)
                files.append(f)
        args.files = files

    for f in args.files:
        path = f'{args.dir}/{f}.png'
        img = cv2.imread(path)

        if args.bgra_2_gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        path = f'{args.dir}/{f}.npy'
        numpy.save(path, img)
        print(f'Saved: {path}')


if __name__ == '__main__':
    main()
