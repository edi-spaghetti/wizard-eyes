import argparse
import keyboard

import cv2

from wizard_eyes.application import Application
from wizard_eyes.script_utils import wait_lock
from wizard_eyes.constants import REDA


class TileMarker(Application):
    """Utility for marking out tile spacing at a specific level of zoom."""

    @property
    def client_init_kwargs(self):
        return {'zoom': self.args.zoom}

    def create_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--zoom', type=int)

        self.parser = parser
        return parser

    def __init__(self, *args, **kwargs):

        self.create_parser()
        self.parse_args()
        super().__init__(*args, **kwargs)

        self.x = 0
        self.y = 0

    @wait_lock
    def move_cursor(self, x, y):

        self.x += x
        self.y += y

        p = self.client.game_screen.player
        h, w, _ = p.templates[p.template_name].shape
        self.nudge_mouse(x * w, y * h)

    def nudge_mouse(self, x, y):

        mx, my = self.client.screen.mouse_xy
        mx += x
        my += y

        self.client.screen.mouse_to(mx, my)

    @wait_lock
    def add_coordinate(self):
        """Add current mouse position, relative to player, to grid."""

        # get mouse relative to client
        mx, my = self.client.screen.mouse_xy
        mx, my, _ , _ = self.client.localise(mx, my, mx, my)

        # get lower left point of player
        px, _, _, py = self.client.game_screen.player.tile_bbox()
        px, py, _, _ = self.client.localise(px, py, px, py)

        # find mouse relative position to player
        rx = mx - px + 1
        ry = my - py + 1

        # add coordinate to the grid
        self.client.game_screen.tile_marker.grid[(self.x, self.y)] = rx, ry
        self.client.game_screen.tile_marker.logger.warning(
            f'Added tile: {self.x, self.y} -> {rx, ry}'
        )

    @wait_lock
    def save_grid(self):
        tm = self.client.game_screen.tile_marker
        tm.save_grid()
        tm.logger.warning(f'grid saved to: {tm.grid_path}')

    def setup(self):
        """"""

        keyboard.add_hotkey('4', lambda: self.move_cursor(-1, 0))
        keyboard.add_hotkey('6', lambda: self.move_cursor(1, 0))
        keyboard.add_hotkey('2', lambda: self.move_cursor(0, 1))
        keyboard.add_hotkey('8', lambda: self.move_cursor(0, -1))

        keyboard.add_hotkey('w', lambda: self.nudge_mouse(0, -1))
        keyboard.add_hotkey('a', lambda: self.nudge_mouse(-1, 0))
        keyboard.add_hotkey('s', lambda: self.nudge_mouse(0, 1))
        keyboard.add_hotkey('d', lambda: self.nudge_mouse(1, 0))

        keyboard.add_hotkey('5', self.add_coordinate)
        keyboard.add_hotkey('+', self.save_grid)

    def highlight_current(self):
        tm = self.client.game_screen.tile_marker

        px, _, _, py = self.client.game_screen.player.tile_bbox()

        try:
            rx, ry = tm.grid[(self.x, self.y)]
            gx = px + rx
            gy = py + ry
            gx, gy, _, _ = self.client.localise(gx, gy, gx, gy)
        except KeyError:
            rx, ry = self.client.screen.mouse_xy
            gx, gy, _, _ = self.client.localise(rx, ry, rx, ry)

        cv2.circle(
            self.client.original_img, (gx, gy),
            self.client.game_screen.tile_marker.GRID_POINT_RADIUS,
            REDA, -1
        )

    def update(self):
        """"""
        self.client.game_screen.tile_marker.update()
        self.client.add_draw_call(self.highlight_current)

    def action(self):
        """"""
        self.msg.append(f'current: {self.x, self.y}')


def main():
    app = TileMarker()
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
