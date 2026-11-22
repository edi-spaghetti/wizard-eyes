"""Bank slot grid, tag column and bottom buttons geometry."""
from unittest import TestCase

from wizard_eyes.game_objects.bank import grid


class TestBankGrid(TestCase):

    def test_rows_follow_button_row(self):
        # measured on a 1280x720 client: buttons at 438 from the bank top
        self.assertEqual(grid.slot_rows(438), 10)
        # a taller client: more rows; no slot below the buttons
        for top in range(300, 1300, 7):
            rows = grid.slot_rows(top)
            if rows:
                _, _, _, y2 = grid.slot_bbox(rows * grid.COLUMNS - 1)
                self.assertLess(y2, top)
            _, y1, _, _ = grid.slot_bbox(rows * grid.COLUMNS)
            self.assertGreaterEqual(y1 + grid.SLOT_SIZE[1], top - 1)

    def test_slot_positions(self):
        self.assertEqual(grid.slot_bbox(0)[:2], grid.SLOT_ORIGIN)
        x1, y1, _, _ = grid.slot_bbox(13)  # row 1, column 5
        self.assertEqual((x1, y1), (grid.SLOT_ORIGIN[0] + 5 * 48,
                                    grid.SLOT_ORIGIN[1] + 36))

    def test_slot_colours_unique(self):
        colours = [grid.slot_colour(i) for i in range(60 * grid.COLUMNS)]
        self.assertEqual(len(set(colours)), len(colours))
        self.assertTrue(all(0 <= c <= 255 for col in colours for c in col))

    def test_tags(self):
        self.assertEqual(grid.tag_count(438), 7)
        _, _, _, y2 = grid.tag_bbox(6)
        self.assertLess(y2, 438 - 40)

    def test_buttons_from_all(self):
        self.assertEqual(grid.button_bbox('quantity_all', 235, 438),
                         (235, 438, 270, 473))
        self.assertEqual(grid.button_bbox('search', 235, 438)[0], 278)
        self.assertEqual(grid.button_bbox('search', 235, 438)[2] - 278, 45)
        groups = (set(grid.QUANTITY_BUTTONS) | set(grid.TOGGLE_BUTTONS)
                  | set(grid.ACTION_BUTTONS))
        self.assertEqual(groups, set(grid.BUTTONS))
