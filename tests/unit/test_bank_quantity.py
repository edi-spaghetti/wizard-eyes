"""Bank quantities, drawn with a font like the game's."""
from unittest import TestCase

import numpy

from wizard_eyes.game_objects.bank.quantity import (
    GREEN, HEIGHT, K, M, TEXT_BOX, WHITE, YELLOW, read_quantity)


def _font():
    """Ten distinct 8-row glyphs, each one run of columns."""
    font = {}
    for d in range(10):
        g = numpy.zeros((HEIGHT, 4), dtype=bool)
        g[0] = g[-1] = True
        g[:, 0] = True
        for bit in range(4):
            g[2 + bit, 2] = bool(d >> bit & 1)
        font[str(d)] = g
    return font


FONT = _font()
BG = (50, 61, 69)


def draw(text, colour=YELLOW, at=(1, 2)):
    img = numpy.empty((TEXT_BOX[3] - TEXT_BOX[1] + 1,
                       TEXT_BOX[2] + 1, 4), dtype=numpy.uint8)
    img[:] = (*BG, 255)
    x, y = at
    for ch in text:
        g = {'K': K, 'M': M}.get(ch, FONT.get(ch))
        img[y:y + HEIGHT, x:x + g.shape[1], :3][g] = colour
        x += g.shape[1] + 2
    return img


class TestBankQuantity(TestCase):
    def read(self, img):
        return read_quantity(img, FONT)

    def test_units(self):
        self.assertEqual(self.read(draw('29797')), 29_797)
        self.assertEqual(self.read(draw('561K', WHITE)), 561_000)
        self.assertEqual(self.read(draw('992M', GREEN)), 992_000_000)

    def test_position_may_shift_a_little(self):
        self.assertEqual(self.read(draw('12', at=(3, 4))), 12)

    def test_nothing(self):
        self.assertIsNone(self.read(draw('')))

    def test_placeholder(self):
        self.assertEqual(self.read(draw('0', (24, 165, 169))), 0)
        self.assertIsNone(self.read(draw('7', (24, 165, 169))))

    def test_covered_is_not_read(self):
        img = draw('46804')
        img[:, 12:20, :3] = BG          # a digit and a bit in the middle
        self.assertIsNone(self.read(img))
        img = draw('46804')
        img[2:5, :, :3] = BG            # the top rows
        self.assertIsNone(self.read(img))

    def test_unit_missing(self):
        self.assertIsNone(self.read(draw('561', WHITE)))
        self.assertIsNone(self.read(draw('561', GREEN)))
