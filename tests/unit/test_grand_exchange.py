import unittest

import numpy

from wizard_eyes.grand_exchange import (SETUP_PARTS, SLOT_COLUMNS, SLOTS,
                                        columns, glyphs, read_digits,
                                        similar)


def compose(text, font, gap=1, junk=False):
    """A text mask the way the game draws it: glyphs one pixel apart."""
    parts = []
    for ch in text:
        g = font[ch]
        parts.append(g)
        parts.append(numpy.zeros((g.shape[0], gap), bool))
    if junk:  # "coins": a blob that matches no digit
        h = parts[0].shape[0]
        blob = numpy.zeros((h, 6), bool)
        blob[6:11, :] = True
        parts += [numpy.zeros((h, 3), bool), blob]
    mask = numpy.hstack(parts)
    return numpy.pad(mask, ((0, 0), (5, 5)))


class TestDigits(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.font = glyphs()

    def test_font_complete(self):
        self.assertEqual(set(self.font), set('0123456789,'))

    def test_reads_numbers(self):
        for value, text in ((1, '1'), (4700, '4,700'), (235689, '235,689'),
                            (1047890, '1,047,890'), (11000, '11,000')):
            mask = compose(text, self.font, junk=True)
            self.assertEqual(read_digits(mask, self.font), value, text)

    def test_bad_grouping(self):
        mask = compose('47,00', self.font)
        self.assertIsNone(read_digits(mask, self.font))

    def test_nothing(self):
        self.assertIsNone(read_digits(numpy.zeros((18, 30), bool),
                                      self.font))

    def test_columns(self):
        m = numpy.zeros((3, 10), bool)
        m[1, 2:4] = True
        m[0, 6] = True
        self.assertEqual(columns(m), [(2, 4), (6, 7)])


class TestLayout(unittest.TestCase):

    def test_parts_inside_interface(self):
        # the set-up screen spans x -53..429, y -6..296 from its bar
        for name, (x1, y1, x2, y2) in SETUP_PARTS.items():
            self.assertTrue(-53 <= x1 < x2 <= 429 and -6 <= y1 < y2 <= 296,
                            name)

    def test_slots(self):
        self.assertEqual(SLOTS, 8)
        self.assertEqual(SLOT_COLUMNS, 4)

    def test_similar(self):
        self.assertGreater(similar('Grimy ranarr weed', 'Grimyranarrweed'),
                           0.95)
        self.assertLess(similar('Grimy ranarr weed', 'Grimy toadflax'), 0.85)


if __name__ == '__main__':
    unittest.main()
