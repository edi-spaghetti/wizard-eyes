"""Transports: guesses from synthetic map data, merging, recording."""
import json
import os
import tempfile
from unittest import TestCase

import numpy

from wizard_eyes.cache.collision import CollisionBuilder
from wizard_eyes.cache.maps import Location, Region, region_id
from wizard_eyes.cache.object_data import build_object_data
from wizard_eyes.cache.objects import ObjectDefinition
from wizard_eyes.navigation.collision_map import CollisionMap
from wizard_eyes.navigation.object_map import ObjectMap
from wizard_eyes.navigation.transports import (
    Transport, Transports, guess_transports, merge_recordings, save_guesses,
)

RX, RY = 50, 50          # surface region, base (3200, 3200)
UX, UY = 50, 150         # underground region, base (3200, 9600)
SX, SY = RX * 64, RY * 64
UX0, UY0 = UX * 64, UY * 64


def acts(*names):
    return list(names) + [None] * (5 - len(names))


def definitions():
    return {
        1: ObjectDefinition(1, name='Staircase', actions=acts('Climb-up'),
                            size_x=2, size_y=2),
        2: ObjectDefinition(2, name='Staircase',
                            actions=acts('Climb-down'), size_x=2, size_y=2),
        3: ObjectDefinition(3, name='Ladder', actions=acts('Climb-down')),
        4: ObjectDefinition(4, name='Ladder', actions=acts('Climb-up')),
        5: ObjectDefinition(5, name='Trapdoor', actions=acts('Climb-down')),
    }


def make_region(rx, ry, locs):
    return Region(region_id(rx, ry), numpy.zeros((4, 64, 64), numpy.uint8),
                  numpy.zeros((4, 64, 64), numpy.int16),
                  [Location(*l) for l in locs])


class TestTransports(TestCase):

    def setUp(self):
        surface = make_region(RX, RY, [
            (1, 10, 10, 0, 10, 0),   # staircase up, partner above
            (2, 10, 10, 1, 10, 0),
            (3, 30, 30, 0, 10, 0),   # ladder down, partner underground
            (5, 50, 50, 0, 22, 0),   # trapdoor with no partner below
        ])
        under = make_region(UX, UY, [
            (4, 31, 30, 0, 10, 0),   # 1 tile east of the surface ladder
        ])
        defs = definitions()
        placements, used = build_object_data([surface, under], defs)
        builder = CollisionBuilder(defs)
        builder.add_region(surface)
        builder.add_region(under)
        movement, doors = builder.all_masks()

        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        meta = numpy.array(json.dumps({'cache': {}}))
        numpy.savez_compressed(
            os.path.join(d, 'objects.npz'), meta=meta,
            definitions=numpy.array(json.dumps(used)), **placements)
        arrays = {f'm{r}': m for r, m in movement.items()}
        numpy.savez_compressed(os.path.join(d, 'movement.npz'), meta=meta,
                               **arrays)
        self.om = ObjectMap(os.path.join(d, 'objects.npz'))
        self.cm = CollisionMap(os.path.join(d, 'movement.npz'))
        self.guesses = guess_transports(self.om, self.cm)
        self.guessed = os.path.join(d, 'guessed.npz')
        save_guesses(self.guessed, self.guesses, {'cache': {}})
        self.curated = os.path.join(d, 'transports.json')
        self.recorded = os.path.join(d, 'recorded.local.json')

    def tearDown(self):
        self.cm._npz.close()
        self.tmp.cleanup()

    def load(self, **kw):
        return Transports(self.guessed, self.curated, self.recorded, **kw)

    def guess(self, tile, action):
        for g in self.guesses:
            if g.object_tile == tile and g.action == action:
                return g
        return None

    def test_stairs_between_floors(self):
        up = self.guess((SX + 10, SY + 10, 0), 'Climb-up')
        self.assertEqual(up.to[2], 1)
        self.assertEqual(up.to_radius, 2)
        self.assertGreaterEqual(up.confidence, 0.8)
        down = self.guess((SX + 10, SY + 10, 1), 'Climb-down')
        self.assertEqual(down.to[2], 0)

    def test_ladder_to_underground_and_back(self):
        down = self.guess((SX + 30, SY + 30, 0), 'Climb-down')
        self.assertGreater(down.to[1], 6400)
        self.assertLess(down.confidence, 0.5)  # partner 1 tile off
        up = self.guess((UX0 + 31, UY0 + 30, 0), 'Climb-up')
        self.assertLess(up.to[1], 6400)

    def test_no_partner_no_guess(self):
        self.assertIsNone(self.guess((SX + 50, SY + 50, 0), 'Climb-down'))

    def test_min_confidence(self):
        self.assertIsNone(self.load().get((SX + 30, SY + 30, 0),
                                          'Climb-down'))
        self.assertIsNotNone(self.load(min_confidence=0).get(
            (SX + 30, SY + 30, 0), 'Climb-down'))

    def test_record_overrides_guess(self):
        t = self.load()
        obj = self.om.find('Staircase', 'Climb-up')[0]
        t.record(obj, 'Climb-up', (SX + 9, SY + 10, 0), (SX + 11, SY + 12, 1))
        again = self.load()
        got = again.get(obj.tile, 'climb-up')
        self.assertEqual(got.source, 'observed')
        self.assertEqual(got.to, (SX + 11, SY + 12, 1))
        again.record(obj, 'Climb-up', (SX + 9, SY + 10, 0),
                     (SX + 11, SY + 12, 1))
        self.assertEqual(self.load().get(obj.tile, 'Climb-up').count, 2)

    def test_manual_beats_recording(self):
        obj = self.om.find('Staircase', 'Climb-up')[0]
        manual = Transport(obj.tile, 'Climb-up', (1, 2, 1), source='manual',
                           requirements={'quest': 'Some quest'})
        with open(self.curated, 'w') as f:
            json.dump([manual.to_json()], f)
        t = self.load()
        t.record(obj, 'Climb-up', obj.tile, (SX + 11, SY + 12, 1))
        got = self.load().get(obj.tile, 'Climb-up')
        self.assertEqual(got.source, 'manual')
        self.assertEqual(got.requirements, {'quest': 'Some quest'})

    def test_merge_recordings(self):
        obj = self.om.find('Staircase', 'Climb-up')[0]
        self.load().record(obj, 'Climb-up', obj.tile, (SX + 11, SY + 12, 1))
        added, updated, warnings = merge_recordings(self.curated,
                                                    self.recorded)
        self.assertEqual((added, updated, warnings), (1, 0, []))
        # a different landing later is reported (random landing?)
        os.remove(self.recorded)
        self.load().record(obj, 'Climb-up', obj.tile, (SX + 8, SY + 12, 1))
        added, updated, warnings = merge_recordings(self.curated,
                                                    self.recorded)
        self.assertEqual((added, updated, len(warnings)), (0, 1, 1))

    def test_edges(self):
        edges = self.load(min_confidence=0).edges(self.om, self.cm)
        stairs = [(s, to) for s, lst in edges.items() for to, _, t in lst
                  if t.object_tile == (SX + 10, SY + 10, 0)]
        self.assertTrue(stairs)
        self.assertTrue(all(s[2] == 0 and to[2] == 1 for s, to in stairs))
