from typing import List
from unittest.mock import ANY

from wizard_eyes.application import Application
from wizard_eyes.game_objects.game_objects import GameObject
from wizard_eyes.game_entities.screen import TileColour
from wizard_eyes.constants import COLOUR_DICT_HSV

import cv2
import numpy

import keyboard


class Tile(GameObject):
    """Game object to represent found highlighted tiles."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.idx = None
        self.name = None

    def draw(self):
        """Draw the tile on the original image.
        Right the results index to the center of the bounding box, and
        write the name of the tile below the bounding box."""
        super().draw()

        x1, y1, x2, y2 = self.get_bbox()
        x1, y1, x2, y2 = self.client.localise(x1, y1, x2, y2)
        x, y = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.putText(self.client.original_img, str(self.idx), (x, y),
                    cv2.FONT_HERSHEY_COMPLEX, 1, self.colour, 2)

        cv2.putText(self.client.original_img, str(self.name), (x, y + 20),
                    cv2.FONT_HERSHEY_COMPLEX, 1, self.colour, 2)


class FindHighlightedTilesApp(Application):
    """Demo of how to find highlighted tiles in the game screen."""

    def create_parser(self):
        """Create a parser to define a tile colour for testing."""
        parser = super().create_parser()

        parser.add_argument(
            '--item-colour',
            type=str,
            default='purple',
            help='The colour of the item to find.',
            choices=COLOUR_DICT_HSV.keys(),
        )

        return parser

    def __init__(self, *args, **kwargs):
        """Init containers for search colour, results, and how to draw them."""
        super().__init__(*args, **kwargs)
        self.tile_colours: List[TileColour] = []
        self.tiles: List[GameObject] = []
        self.draw_idx = ANY
        self.show_failures = False

    def setup(self):
        """Ensure we're actually displaying the results, configured the
        colour to search for set up hotkeys to alter display option.

        Press 'esc' to stop highlighting one particular tile.
        Press 'f' to toggle showing failures.
        Press 'tab' or 'shift+tab' to cycle forward/back through the results
        and only highlight one result at a time.

        """

        self.client.args.show = {'*bbox', 'mouse'}

        self.tile_colours.append(
            TileColour(
                name=self.args.item_colour,
                lower=COLOUR_DICT_HSV[self.args.item_colour][1],
                upper=COLOUR_DICT_HSV[self.args.item_colour][0],
            )
        )

        keyboard.add_hotkey('esc', lambda: setattr(self, 'draw_idx', ANY))
        keyboard.add_hotkey(
            'f',
            lambda: setattr(self, 'show_failures', not self.show_failures)
        )
        keyboard.add_hotkey(
            'tab', lambda: setattr(
                self, 'draw_idx',
                (self.draw_idx + 1 % len(self.tiles))
                if len(self.tiles) and self.draw_idx is not ANY else 0
            )
        )

        keyboard.add_hotkey(
            'shift+tab', lambda: setattr(
                self, 'draw_idx',
                (self.draw_idx - 1 % len(self.tiles))
                if len(self.tiles) and self.draw_idx is not ANY else 0
            )
        )

    def update(self):
        """Find highlighted tiles and display them on screen."""

        self.client.game_screen.player.update()

        self.tiles = []
        tiles = self.client.game_screen.find_highlighted_tiles(
            self.tile_colours)
        for i, (tile_colour, (x1, y1, x2, y2)) in enumerate(tiles):
            tile = Tile(self.client, self.client, data=tile_colour)
            # x1, y1, x2, y2 = self.client.globalise(x1, y1, x2, y2)
            tile.set_aoi(x1, y1, x2, y2)
            tile.idx = i

            # set the game object the opposite of the actual highlight so the
            # draw call is easier to see
            colour_bounds = numpy.uint8([[tile_colour.upper]])
            bgr = cv2.cvtColor(colour_bounds, cv2.COLOR_HSV2BGR)
            opposite = cv2.bitwise_not(bgr)
            tile.colour = tuple(int(o) for o in opposite.reshape(-1))
            tile.name = tile_colour.name

            if self.draw_idx == i:
                if 'fail' in tile.name and self.show_failures:
                    tile.update()
                elif tile.name == self.args.item_colour:
                    tile.update()
                else:
                    fuckyou = 1  # the fuck mate

            self.tiles.append(tile)

    def action(self):
        """Log results."""

        self.msg.append(f'Found {len(self.tiles)} tiles')
        self.msg.append(f'{self.draw_idx}, {self.show_failures}')


def main():
    app = FindHighlightedTilesApp()
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
