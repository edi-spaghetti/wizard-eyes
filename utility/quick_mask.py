from argparse import ArgumentParser
from os.path import dirname, join, splitext, exists

import cv2
import keyboard


INTERACTIVE_CONTINUE = True


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('lower', type=int)
    parser.add_argument('upper', type=int)

    parser.add_argument('--colour-mask', action='store_true', default=False)
    parser.add_argument('--delta', type=int, default=1)

    parser.add_argument('--red-lower', type=int, default=0)
    parser.add_argument('--green-lower', type=int, default=0)
    parser.add_argument('--blue-lower', type=int, default=0)
    parser.add_argument('--red-upper', type=int, default=255)
    parser.add_argument('--green-upper', type=int, default=255)
    parser.add_argument('--blue-upper', type=int, default=255)

    parser.add_argument('--overwrite', default=False, action='store_true')
    parser.add_argument('--interactive', default=False, action='store_true')

    args = parser.parse_args()
    return args


def update_value(args, attribute, delta):
    current = getattr(args, attribute)
    setattr(args, attribute, current + delta)


def stop_interactive():
    global INTERACTIVE_CONTINUE
    INTERACTIVE_CONTINUE = False
    cv2.destroyAllWindows()


def create_mask(img, args):
    if args.colour_mask:
        mask = cv2.inRange(
            img,
            (args.blue_lower, args.green_lower, args.red_lower),
            (args.blue_upper, args.green_upper, args.red_upper),
        )
    else:
        _, mask = cv2.threshold(img, args.lower, args.upper, cv2.THRESH_BINARY)

    return mask


def interactive_threshold(img, args):

    keyboard.add_hotkey('-', lambda: update_value(args, 'delta', -1))
    keyboard.add_hotkey('+', lambda: update_value(args, 'delta', +1))

    # lower range
    keyboard.add_hotkey('a', lambda: update_value(args, 'blue_lower', -args.delta))
    keyboard.add_hotkey('q', lambda: update_value(args, 'blue_lower', args.delta))
    keyboard.add_hotkey('s', lambda: update_value(args, 'green_lower', -args.delta))
    keyboard.add_hotkey('w', lambda: update_value(args, 'green_lower', args.delta))
    keyboard.add_hotkey('d', lambda: update_value(args, 'red_lower', -args.delta))
    keyboard.add_hotkey('e', lambda: update_value(args, 'red_lower', args.delta))

    # upper range
    keyboard.add_hotkey('1', lambda: update_value(args, 'blue_upper', -args.delta))
    keyboard.add_hotkey('4', lambda: update_value(args, 'blue_upper', args.delta))
    keyboard.add_hotkey('2', lambda: update_value(args, 'green_upper', -args.delta))
    keyboard.add_hotkey('5', lambda: update_value(args, 'green_upper', args.delta))
    keyboard.add_hotkey('3', lambda: update_value(args, 'red_upper', -args.delta))
    keyboard.add_hotkey('6', lambda: update_value(args, 'red_upper', args.delta))

    keyboard.add_hotkey('esc', stop_interactive)

    if not args.colour_mask:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    mask = create_mask(img, args)
    while INTERACTIVE_CONTINUE:
        mask = create_mask(img, args)
        cv2.imshow('mask', mask)
        cv2.waitKey(1)

        if args.colour_mask:
            print(
                f'lower: {args.blue_lower, args.green_lower, args.red_lower}, '
                f'upper: {args.blue_upper, args.green_upper, args.red_upper}, '
                f'delta: {args.delta}')
        else:
           print(f'lower: {args.lower}, upper: {args.upper}')

    mask = cv2.bitwise_not(mask)
    return mask


def main():
    args = parse_args()

    path = join(dirname(__file__), '..', 'data', args.path)
    img = cv2.imread(path)

    if args.interactive:
        mask = interactive_threshold(img, args)
    else:
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
