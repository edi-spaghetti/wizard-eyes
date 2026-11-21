import json
import os
import shutil
import tempfile
from unittest import TestCase

from wizard_eyes.runelite_profile import (
    REQUIRED, RECOMMENDED, Rule, active_profile, check_active_profile,
    check_properties, default_rules_path, load_rules, parse_properties,
)

PROPS = r"""#RuneLite configuration
#Sat Nov 21 17:14:12 GMT 2026
runelite.stretchedmodeplugin=false
tileindicators.highlightCurrentColor=-16711681
banktaglayouts.layout_potion\ maker=227\:0,14244\:64
timestamp.format=[HH\:mm]
notes.text=line one\nline two
multi=first \
    second
unicode=café
"""


class TestParseProperties(TestCase):

    def test_escapes(self):
        p = parse_properties(PROPS)
        self.assertEqual(p['runelite.stretchedmodeplugin'], 'false')
        self.assertEqual(p['banktaglayouts.layout_potion maker'], '227:0,14244:64')
        self.assertEqual(p['timestamp.format'], '[HH:mm]')
        self.assertEqual(p['notes.text'], 'line one\nline two')
        self.assertEqual(p['multi'], 'first second')
        self.assertEqual(p['unicode'], 'café')
        self.assertNotIn('#RuneLite configuration', p)


class TestCheck(TestCase):

    def test_mismatch_missing_and_default(self):
        rules = [
            Rule('a', 'true'),
            Rule('b', 'false', default='false'),   # missing, default matches
            Rule('c', 'true'),                     # missing, no default
            Rule('d', 'x', level=RECOMMENDED),
        ]
        m = check_properties({'a': 'false', 'd': 'x'}, rules)
        self.assertEqual([x.rule.key for x in m], ['a', 'c'])
        self.assertEqual(m[0].actual, 'false')
        self.assertIsNone(m[1].actual)
        self.assertIn('not set', str(m[1]))

    def test_repo_rules_load(self):
        rules = load_rules(default_rules_path())
        self.assertTrue(any(r.level == REQUIRED for r in rules))
        self.assertTrue(any(r.level == RECOMMENDED for r in rules))


class TestActiveProfile(TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        p2 = os.path.join(self.dir, 'profiles2')
        os.makedirs(p2)
        with open(os.path.join(p2, 'profiles.json'), 'w') as f:
            json.dump({'profiles': [
                {'id': 1, 'name': 'default', 'active': False},
                {'id': 42, 'name': 'bot', 'active': True},
            ]}, f)
        with open(os.path.join(p2, 'bot-42.properties'), 'w') as f:
            f.write('x=1\ny=2\n')
        self.rules = os.path.join(self.dir, 'rules.json')
        with open(self.rules, 'w') as f:
            json.dump({'required': [{'key': 'x', 'expected': '1'}],
                       'recommended': [{'key': 'y', 'expected': '3'}]}, f)

    def test_finds_active(self):
        p = active_profile(self.dir)
        self.assertEqual((p.name, p.id), ('bot', 42))

    def test_check(self):
        r = check_active_profile(self.dir, self.rules)
        self.assertTrue(r.ok)
        self.assertEqual(r.checked, 2)
        self.assertEqual([m.rule.key for m in r.recommended_failures], ['y'])

    def test_required_failure(self):
        with open(self.rules, 'w') as f:
            json.dump({'required': [{'key': 'x', 'expected': '9'}]}, f)
        self.assertFalse(check_active_profile(self.dir, self.rules).ok)

    def test_no_runelite(self):
        r = check_active_profile(os.path.join(self.dir, 'nope'), self.rules)
        self.assertFalse(r.ok)
        self.assertIn('No active', r.error)
