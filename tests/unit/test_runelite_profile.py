import json
import os
import shutil
import tempfile
from unittest import TestCase
from unittest import mock

from wizard_eyes import runelite_profile
from wizard_eyes.runelite_profile import (
    REQUIRED, RECOMMENDED, Rule, RuneLiteRunning, active_profile,
    check_active_profile, check_properties, default_rules_path,
    import_profile, load_rules, parse_properties,
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


class TestNotRule(TestCase):

    def test_must_not_equal(self):
        rules = [Rule('sky', '!-16711686')]
        self.assertEqual(len(check_properties({'sky': '-16711686'}, rules)), 1)
        self.assertEqual(check_properties({'sky': '-1'}, rules), [])
        self.assertEqual(check_properties({}, rules), [])
        self.assertIn('must not be', str(check_properties({'sky': '-16711686'}, rules)[0]))


class TestImportProfile(TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.src = os.path.join(self.dir, 'src.properties')
        with open(self.src, 'w', encoding='utf-8') as f:
            f.write(PROPS)
        self.rl = os.path.join(self.dir, '.runelite')
        patcher = mock.patch.object(runelite_profile, 'runelite_running',
                                    return_value=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def index(self):
        path = os.path.join(self.rl, 'profiles2', 'profiles.json')
        with open(path, encoding='utf-8') as f:
            return json.load(f)

    def test_new_runelite_dir(self):
        p = import_profile(self.src, 'naughty', self.rl)
        self.assertTrue(os.path.isfile(p.path))
        self.assertEqual(open(p.path, encoding='utf-8').read(), PROPS)
        self.assertEqual(active_profile(self.rl).name, 'naughty')
        self.assertEqual(active_profile(self.rl).id, p.id)

    def test_existing_profiles_kept_and_the_import_activated(self):
        os.makedirs(os.path.join(self.rl, 'profiles2'))
        with open(os.path.join(self.rl, 'profiles2', 'profiles.json'), 'w') as f:
            json.dump({'profiles': [
                {'id': 7, 'name': 'good', 'sync': False, 'active': True,
                 'rev': -1},
                {'id': -1, 'name': '$rsprofile', 'sync': True,
                 'active': False, 'rev': -1, 'defaultForRsProfiles': []}]}, f)
        import_profile(self.src, 'naughty', self.rl)
        names = {p['name']: p['active'] for p in self.index()['profiles']}
        self.assertEqual(names, {'good': False, '$rsprofile': False,
                                 'naughty': True})

    def test_same_name_is_replaced_not_duplicated(self):
        first = import_profile(self.src, 'naughty', self.rl)
        with open(self.src, 'w', encoding='utf-8') as f:
            f.write('a=b')
        second = import_profile(self.src, 'naughty', self.rl)
        self.assertEqual(first.id, second.id)
        self.assertEqual(len(self.index()['profiles']), 1)
        self.assertEqual(open(second.path, encoding='utf-8').read(), 'a=b')

    def test_not_while_runelite_runs(self):
        with mock.patch.object(runelite_profile, 'runelite_running',
                               return_value=True):
            with self.assertRaises(RuneLiteRunning):
                import_profile(self.src, 'naughty', self.rl)
            self.assertFalse(os.path.exists(self.rl))
            import_profile(self.src, 'naughty', self.rl, force=True)
        self.assertTrue(os.path.exists(self.rl))

    def test_missing_source(self):
        with self.assertRaises(FileNotFoundError):
            import_profile(os.path.join(self.dir, 'nope'), 'naughty', self.rl)
