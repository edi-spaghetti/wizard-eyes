import unittest

import numpy

from wizard_eyes.grand_exchange import (SETUP_PARTS, bar_progress,
                                        best_name, find_main_title, template, SLOT_COLUMNS, SLOTS,
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


class TestMainScreen(unittest.TestCase):

    def bar(self, extra=None, gap=6):
        """The main bar with something bright after its title, re-centred."""
        bar = template('main_bar').copy()
        if extra is None:
            return bar
        word = bar[:, 136:239].copy()
        bar[:, 136:239] = 65
        width = 103 + gap + extra
        start = 187 - width // 2
        bar[:, start:start + 103] = word
        bar[7:15, start + 103 + gap:start + width] = 200
        return bar

    def find(self, bar):
        img = numpy.full((60, 500), 65, numpy.uint8)
        img[20:42, 50:50 + bar.shape[1]] = bar
        return find_main_title(img, template('main_bar'))

    def test_with_and_without_value(self):
        self.assertEqual(self.find(self.bar()), (50, 20))
        for extra in (30, 40, 41):
            self.assertEqual(self.find(self.bar(extra)), (50, 20))

    def test_other_screens(self):
        self.assertIsNone(self.find(template('setup_bar')))
        self.assertIsNone(self.find(numpy.full((22, 394), 65, numpy.uint8)))

    def test_progress(self):
        empty, orange = (39, 49, 56), (31, 136, 217)
        green, red = (0, 95, 0), (0, 0, 143)
        row = numpy.array([orange] * 26 + [empty] * 78)
        self.assertEqual(bar_progress(row), ('active', 0.25))
        self.assertEqual(bar_progress(numpy.array([empty] * 104)),
                         ('active', 0.0))
        self.assertEqual(bar_progress(numpy.array([green] * 104)),
                         ('complete', 1.0))
        self.assertEqual(bar_progress(numpy.array([red] * 104)),
                         ('cancelled', 1.0))

    def test_names(self):
        items = ['Grimy toadflax', 'Toadflax', 'Grimy tarromin']
        self.assertEqual(best_name('Grimy koadflax', items), items[0])
        self.assertIsNone(best_name('Rune axe', items))


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
