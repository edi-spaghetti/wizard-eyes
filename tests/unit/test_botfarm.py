import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy

from wizard_eyes.botfarm import alerts as a
from wizard_eyes.botfarm.control import Control
from wizard_eyes.botfarm.farm import ENV, Farm, NoFarm, state_folder
from wizard_eyes.botfarm.live import Live, NoPicture, to_jpeg
from wizard_eyes.botfarm.remote import (Remote, cookie_name, hashed, main,
                                        new_key)

TOKEN = '123:SECRET'
HOST = 'motherbear.example.ts.net'


class Folder(unittest.TestCase):

    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.folder = Path(d.name)


class TestFarm(Folder):

    def write(self, **d):
        farm = self.folder / 'farm'
        farm.mkdir(exist_ok=True)
        (farm / 'farm.json').write_text('﻿' + json.dumps(d), 'utf-8')
        return farm

    def test_load(self):
        farm = self.write(name='Herbs', state_dir='data/x', roles={
            'cleaner': {'vm': 'vm1'}, 'mixer': {'vm': 'vm2'}},
            alerts={'telegram': 'm', 'credential': 'c/t'})
        f = Farm.load(farm, engine=self.folder)
        self.assertEqual(f.name, 'Herbs')
        self.assertEqual(f.state_dir, self.folder / 'data' / 'x')
        self.assertEqual(f.vms, {'cleaner': 'vm1', 'mixer': 'vm2'})
        self.assertEqual(f.roles, ('cleaner', 'mixer'))
        self.assertEqual(f.telegram_credential, 'c/t')

    def test_where_from(self):
        farm = self.write(name='Herbs')
        local = self.folder / 'local'
        settings = local / 'wizard-eyes' / 'botfarm.json'
        settings.parent.mkdir(parents=True)
        env = {k: v for k, v in os.environ.items() if k != ENV}
        env['LOCALAPPDATA'] = str(local)
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(NoFarm):
                Farm.load()
            settings.write_text('﻿' + json.dumps(
                {'default': str(farm)}), 'utf-8')
            self.assertEqual(Farm.load().folder, farm.resolve())
            os.environ[ENV] = str(self.folder)      # (no farm.json there)
            with self.assertRaises(NoFarm):
                Farm.load()
            self.assertEqual(state_folder(self.folder), self.folder)


class TestTelegram(Folder):

    def setUp(self):
        super().setUp()
        self.asked, self.answers = [], []
        self.t = a.Telegram(self.folder, lambda: TOKEN, self.opener)

    def opener(self, request, timeout):
        self.asked.append((request.full_url, json.loads(request.data)))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return io.BytesIO(json.dumps(answer).encode())

    def update(self, text, kind='private', i=1):
        return {'update_id': i, 'message': {
            'text': text, 'chat': {'id': 77, 'type': kind}}}

    def pair(self, *updates):
        now = [0.0]

        def sleep(s):
            now[0] += s
        self.answers = [{'ok': True, 'result': list(updates)}] + [
            {'ok': True, 'result': []}] * 20
        return self.t.pair('1234', 10, lambda: now[0], sleep)

    def test_pairs_with_the_code_only(self):
        self.assertFalse(self.pair(
            self.update('/start'), self.update('/start 9999'),
            self.update('/start 1234', 'group')))
        self.assertIsNone(self.t.chat())
        self.assertFalse(self.t.ready)
        self.assertTrue(self.pair(self.update('/start 1234')))
        self.assertEqual(self.t.chat(), 77)
        self.assertTrue(self.t.ready)

    def test_send(self):
        self.t.path.write_text('{"chat": 77}')
        self.answers = [{'ok': True, 'result': {}}]
        self.t.send('hello')
        self.assertEqual(self.asked[0][1], {'chat_id': 77, 'text': 'hello'})

    def test_the_token_stays_out_of_errors(self):
        self.answers = [OSError(f'https://x/bot{TOKEN}/y'),
                        {'ok': False}]
        for _ in range(2):
            with self.assertRaises(a.TelegramError) as e:
                self.t.call('sendMessage')
            self.assertNotIn('SECRET', str(e.exception))
            self.assertTrue(e.exception.__suppress_context__ or
                            e.exception.__context__ is None)

    def test_no_token(self):
        t = a.Telegram(self.folder, lambda: None, self.opener)
        self.assertFalse(t.ready)
        with self.assertRaises(a.TelegramError):
            t.call('getMe')

    def test_for_farm(self):
        farm = Farm('X', self.folder, self.folder,
                    telegram_credential='no/such/credential/here')
        t = a.Telegram.for_farm(farm)
        self.assertEqual(t.path.parent, self.folder)
        self.assertIsNone(t.token())


class FakeTelegram:
    ready = True

    def __init__(self, fails=0):
        self.sent, self.fails = [], fails

    def send(self, text):
        if self.fails:
            self.fails -= 1
            raise a.TelegramError('down')
        self.sent.append(text)


class TestAlerts(unittest.TestCase):

    def make(self, **kw):
        self.now = 0.0
        self.t = FakeTelegram(**kw)
        return a.Alerts(self.t, lambda: self.now, lambda s: None)

    def drain(self, al):
        while not al.queue.empty():
            al.deliver(al.queue.get())
        return self.t.sent

    def test_not_twice(self):
        al = self.make()
        al.say('x')
        al.say('x')
        self.now += a.REPEAT_SECONDS + 1
        al.say('x')
        self.assertEqual(self.drain(al), ['x', 'x'])

    def test_tries_again(self):
        al = self.make(fails=2)
        al.say('x')
        self.assertEqual(self.drain(al), ['x'])

    def test_not_set_up(self):
        al = self.make()
        self.t.ready = False
        al.say('x')
        self.assertEqual(self.drain(al), [])

    def test_never_blocks(self):
        al = self.make()
        for i in range(200):
            al.say(str(i))


class TestWatch(unittest.TestCase):

    def setUp(self):
        self.said = []
        self.w = a.Watch(self.said.append, lambda: 100.0)

    def test_bot(self):
        self.w.bot('mixer', 'ok', 'working')
        self.w.bot('mixer', 'wait', 'no report for 8 min')
        self.assertEqual(self.said, [])
        self.w.bot('mixer', 'restart', 'restarted: frozen')
        self.w.bot('mixer', 'wait', 'being started')
        self.w.bot('mixer', 'ok', 'working')
        self.w.bot('mixer', 'ok', 'working')
        self.w.bot('cleaner', 'alert', 'out of herbs')
        self.assertEqual(self.said, [
            'Mixer: restarted: frozen', 'Mixer is working again',
            'Cleaner needs you: out of herbs'])

    def test_meetings(self):
        m = dict(id='a', state='met', updated=150.0, reasons={})
        self.w.met([m])
        m = dict(m, state='aborted', reasons={'mixer': 'no bank'})
        self.w.met([m, dict(m, id='old', updated=50.0)])
        self.w.met([m])
        self.assertEqual(self.said, ['Meeting failed. mixer: no bank'])


class TestRemote(Folder):

    def setUp(self):
        super().setUp()
        self.remote = Remote(self.folder, 'Herbitrage')

    def test_cookie_and_page_named_after_the_farm(self):
        self.assertEqual(self.remote.cookie_name, '__Host-herbitrage')
        self.assertEqual(cookie_name('My <Farm> 2'), '__Host-my-farm-2')
        self.assertIn(b'<title>Herbitrage</title>', self.remote.login_page)
        self.assertNotIn(b'<Farm>', Remote(self.folder, '<Farm>').login_page)
        key = new_key()
        cookie = self.remote.cookie(key)
        for part in ('Secure', 'HttpOnly', 'SameSite=Lax', 'Path=/'):
            self.assertIn(part, cookie)
        self.assertEqual(self.remote.key_in(
            {'Cookie': cookie.split(';')[0]}), key.replace('-', ''))
        self.assertEqual(self.remote.key_in(
            {'Authorization': 'Bearer abc'}), 'abc')
        self.assertIsNone(self.remote.key_in({'Cookie': '__Host-x=1'}))

    def test_key_and_host(self):
        key = new_key()
        self.assertFalse(self.remote.key_ok(key))
        self.assertFalse(self.remote.host_ok(HOST, None))
        self.remote.change(host=HOST, key_sha256=hashed(key))
        self.assertTrue(self.remote.key_ok(key.upper()))
        self.assertFalse(self.remote.key_ok(''))
        self.assertFalse(self.remote.key_ok(new_key()))
        self.assertTrue(self.remote.host_ok(HOST.upper(), f'https://{HOST}'))
        self.assertFalse(self.remote.host_ok(HOST, 'https://evil.example'))
        self.assertFalse(self.remote.host_ok('evil.example', None))

    def test_command(self):
        out = io.StringIO()
        folder = ['--folder', str(self.folder)]
        with contextlib.redirect_stdout(out):
            main(['host', HOST.upper() + '.', *folder])
            main(['key', *folder])
        key = out.getvalue().split()[-1]
        self.assertTrue(self.remote.key_ok(key))
        self.assertTrue(self.remote.host_ok(HOST, None))
        self.assertNotIn(key.replace('-', ''), json.dumps(
            self.remote.settings()))
        with contextlib.redirect_stdout(out):
            main(['status', *folder])
            main(['off', *folder])
        self.assertIn(f'host: {HOST}; key: set', out.getvalue())
        self.assertFalse(self.remote.key_ok(key))


def png(w=1920, h=1075):
    return cv2.imencode('.png', numpy.full((h, w, 3), 90, numpy.uint8))[1].tobytes()


class TestLive(Folder):

    def setUp(self):
        super().setUp()
        self.now, self.grabs = 100.0, []

        def grab(vm, path):
            self.grabs.append(vm)
            if vm == 'vm2':
                raise subprocess.TimeoutExpired('VBoxManage', 10)
            return png()
        self.live = Live(self.folder,
                         {'cleaner': 'vm1', 'mixer': 'vm2', 'spare': None},
                         grab, lambda: self.now)

    def test_smaller_jpeg(self):
        img = cv2.imdecode(numpy.frombuffer(to_jpeg(png()), numpy.uint8), 1)
        self.assertEqual(img.shape[:2], (538, 960))
        small = cv2.imdecode(numpy.frombuffer(to_jpeg(png(400, 300)), numpy.uint8), 1)
        self.assertEqual(small.shape[:2], (300, 400))
        with self.assertRaises(NoPicture):
            to_jpeg(b'not a png')

    def test_shared_while_fresh(self):
        a_ = self.live.picture('cleaner')
        self.now += 0.5
        self.assertIs(self.live.picture('cleaner'), a_)
        self.now += 0.5
        self.live.picture('cleaner')
        self.assertEqual(self.grabs, ['vm1'] * 2)

    def test_errors(self):
        for role in ('vm1', 'spare'):
            with self.assertRaises(ValueError):
                self.live.picture(role)
        with self.assertRaises(NoPicture):
            self.live.picture('mixer')


class Job:
    def __init__(self, args, **kw):
        self.args, self.code = args, None

    def poll(self):
        return self.code


class TestControl(Folder):

    def setUp(self):
        super().setUp()
        self.c = Control('x.ps1', self.folder, ('cleaner', 'mixer'),
                         popen=Job)

    def test_one_job_per_bot(self):
        self.c.run('mixer', 'start')
        self.assertIn('start', self.c.jobs['mixer'].args)
        with self.assertRaises(ValueError):
            self.c.run('mixer', 'stop')
        self.c.run('cleaner', 'stop')
        self.c.jobs['mixer'].code = 0
        self.c.run('mixer', 'restart')

    def test_only_known_things(self):
        for role, action in (('x', 'start'), ('mixer', 'format'),
                             (None, None), ([], 'start')):
            with self.assertRaises(ValueError):
                self.c.run(role, action)
        self.assertEqual(self.c.jobs, {})
        with self.assertRaises(ValueError):
            Control(None, self.folder, ('mixer',), popen=Job).run(
                'mixer', 'start')

    def test_status(self):
        self.c.run('mixer', 'start')
        self.c.log('mixer').write_text('﻿1 a\n\n2 OK working\n',
                                       'utf-8')
        s = self.c.status({'mixer': 100}, 150)['bots']
        self.assertTrue(s['mixer']['working'] and s['mixer']['busy'])
        self.assertEqual(s['mixer']['lines'], ['1 a', '2 OK working'])
        self.assertIsNone(s['cleaner']['age'])
        self.assertFalse(s['cleaner']['working'])


if __name__ == '__main__':
    unittest.main()
