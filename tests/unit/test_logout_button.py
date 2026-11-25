from unittest import TestCase

import numpy

from wizard_eyes.game_objects.tabs import logout_button as lb

SHAPE = (23, 24)


def reference():
    """An X of two colours with a dark outline, like the real one."""
    y, x = numpy.mgrid[:SHAPE[0], :SHAPE[1]]
    d = numpy.minimum(abs(y - x), abs(y - (SHAPE[1] - 2 - x)))
    r = numpy.zeros(SHAPE + (4,), numpy.uint8)
    r[d <= 4] = (1, 0, 0, 255)
    r[d <= 3] = (122, 184, 224, 255)
    r[d <= 2] = (72, 152, 166, 255)
    return r


def drawn(ref, ground, opacity):
    """The X over some ground (BGR), as BGRA with alpha 0 (window grabs)."""
    ground = numpy.broadcast_to(ground, SHAPE + (3,)).astype(float)
    on = ref[..., 3:] > 0
    img = numpy.where(
        on, opacity * ref[..., :3] + (1 - opacity) * ground, ground)
    return numpy.dstack([img.round().astype(numpy.uint8),
                         numpy.zeros(SHAPE, numpy.uint8)])


class TestLogoutButton(TestCase):

    def setUp(self):
        self.ref = reference()
        rng = numpy.random.default_rng(1)
        half = numpy.zeros(SHAPE + (3,))
        half[:, 12:] = 250
        self.grounds = {
            'black': (0, 0, 0), 'snow': (250, 250, 250),
            'grass': (20, 70, 30), 'lava': (30, 130, 250),
            'yellow': (30, 210, 255), 'water': (200, 120, 40),
            'noise': rng.integers(60, 120, SHAPE + (3,)),
        }
        self.half = half

    def test_open(self):
        for name, g in {**self.grounds, 'half': self.half}.items():
            with self.subTest(name):
                self.assertEqual(lb.state(drawn(self.ref, g, 1), self.ref),
                                 'logout_selected')

    def test_closed(self):
        for name, g in self.grounds.items():
            with self.subTest(name):
                self.assertEqual(lb.state(drawn(self.ref, g, .6), self.ref),
                                 'logout')

    def test_no_button(self):
        for name, g in {**self.grounds, 'half': self.half}.items():
            with self.subTest(name):
                self.assertEqual(lb.state(drawn(self.ref, g, 0), self.ref),
                                 '')

    def test_wrong_size(self):
        img = drawn(self.ref, (0, 0, 0), 1)
        self.assertEqual(lb.state(img[:-1], self.ref), '')
        self.assertEqual(lb.state(img[:0], self.ref), '')
