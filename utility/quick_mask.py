from argparse import ArgumentParser
from os.path import dirname, join, splitext, exists

import cv2


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('lower', type=int)
    parser.add_argument('upper', type=int)

    parser.add_argument('--overwrite', default=False, action='store_true')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    path = join(dirname(__file__), '..', 'data', args.path)
    img = cv2.imread(path)

    grey = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    _, mask = cv2.threshold(grey, args.lower, args.upper, cv2.THRESH_BINARY)
    mask = cv2.bitwise_not(mask)

    # TODO: multiple threshold ranges

    filename, ext = splitext(path)
    target = f'{filename}_mask{ext}'

    if exists(target) and not args.overwrite:
        print(f'Cannot overwrite existing mask: {target}')
    else:
        cv2.imwrite(target, mask)


if __name__ == '__main__':
    main()
