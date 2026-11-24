import unittest

import numpy

from wizard_eyes.game_objects.bank.withdraw_x import (judge, load,
                                                      read_amount, x_line)


class TestWithdrawX(unittest.TestCase):

    def text(self, glyphs, gap=2, drop=0):
        """A letter, a hyphen and glyphs, spaced like the jiggling text."""
        m = numpy.zeros((20, 200), bool)
        m[4:14, 5:7] = True
        m[9:10, 10:16] = True
        x = 19
        for i, g in enumerate(glyphs):
            h, w = g.shape
            y = 4 + drop * (i % 2)
            m[y:y + h, x:x + w] = g
            x += w + gap
        return m

    def test_amount_any_spacing(self):
        one, four = load('1'), load('4')
        for gap in range(0, 7):
            for drop in (0, 1, 2):
                m = self.text([one, four], gap, drop)
                self.assertEqual(read_amount(m), '14', (gap, drop))

    def test_shaded_pixel(self):
        four = load('4').copy()
        four[0, 0] = False
        self.assertEqual(read_amount(self.text([load('1'), four])), '14')

    def test_other_amounts(self):
        one, four = load('1'), load('4')
        blob = numpy.ones((10, 6), bool)
        self.assertEqual(read_amount(self.text([one])), '1')
        self.assertEqual(read_amount(self.text([one, four, four])), '144')
        self.assertEqual(read_amount(self.text([one, blob])), '1?')
        self.assertEqual(read_amount(self.text([one, four, blob])), '14?')

    def test_text_after_the_number(self):
        m = self.text([load('1'), load('4')])
        m[4:14, 80:82] = True
        self.assertEqual(read_amount(m), '14')

    def test_nothing(self):
        self.assertIsNone(read_amount(numpy.zeros((20, 200), bool)))
        self.assertIsNone(read_amount(self.text([])))

    def test_both_reads_must_agree(self):
        self.assertTrue(judge(14, '14', '14'))
        self.assertFalse(judge(14, '?', '3'))
        self.assertIsNone(judge(14, '14', '1'))
        self.assertIsNone(judge(14, '1?', '14'))
        self.assertIsNone(judge(14, None, '14'))

    def test_menu(self):
        lines = ['Withdraw-1k', 'Withdraw-1', 'Withdraw-5', 'Withdraw-10',
                 'Withdraw-1L', 'Withdraw-X', 'Withdraw-All',
                 'Withdraw-All-but-1', 'Cancel']
        self.assertEqual(x_line(lines), 5)
        self.assertEqual(x_line([i.replace('-X', '-%') for i in lines]), 5)
        self.assertIsNone(x_line(lines[:5] + lines[6:]))
        self.assertIsNone(x_line([i.replace('With', 'De') for i in lines]))


if __name__ == '__main__':
    unittest.main()
