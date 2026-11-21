import json
import os
import shutil
import tempfile
from unittest import TestCase, mock

from wizard_eyes.assets import AssetManager, IncludeRules, LocalStorage
from wizard_eyes.assets.config import AssetConfig, default_rules_path
from wizard_eyes.assets.manifest import MANIFEST_KEY, blob_key, hash_bytes


def write(root, key, data: bytes):
    path = os.path.join(root, *key.split('/'))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(data)
    return path


def read(path):
    with open(path, 'rb') as f:
        return f.read()


RULES = IncludeRules(
    include=['*.npy', '*.pickle', 'alpha.png'],
    exclude_segments=['old', 'psd', '.assets'],
    exclude_patterns=[r'^[^/]+$', r'^inventory/\d+/', r'^maps/-?\d+/',
                      ' - Copy'],
)


class AssetTestCase(TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.source = os.path.join(self.tmp, 'source')
        self.remote = os.path.join(self.tmp, 'remote')
        self.client_dir = os.path.join(self.tmp, 'client')
        os.makedirs(self.client_dir)
        self.storage = LocalStorage(self.remote)

        self.files = {
            'tabs/inventory/frame.npy': b'frame',
            'tabs/inventory/frame_mask.npy': b'mask',
            'tabs/inventory/alpha.png': b'alpha',
            'bank/dup_a.npy': b'same',
            'bank/dup_b.npy': b'same',
            'maps/meta/gielinor.pickle': b'graph',
        }
        for key, data in self.files.items():
            write(self.source, key, data)

    def publish(self):
        return AssetManager(self.source, self.storage).publish(RULES)

    def client(self, storage=None):
        return AssetManager(self.client_dir, storage or self.storage)

    def client_path(self, key):
        return os.path.join(self.client_dir, *key.split('/'))


class TestRules(TestCase):

    def test_matches(self):
        yes = ['tabs/inventory/frame.npy', 'tabs/x/alpha.png',
               'maps/meta/hosidius.pickle', 'bank/tab0/coins_mask.npy']
        no = ['test.npy', 'tabs/old/frame.npy', 'tabs/psd/a.npy',
              'tabs/inventory/frame.png', 'inventory/3/coins.npy',
              'maps/0/1_2.png', 'maps/-1/x.npy', 'bank/a - Copy.npy',
              '.assets/index.json']
        for key in yes:
            self.assertTrue(RULES.matches(key), key)
        for key in no:
            self.assertFalse(RULES.matches(key), key)

    def test_committed_rules_load(self):
        rules = IncludeRules.load(default_rules_path())
        self.assertTrue(rules.matches('tabs/inventory/frame.npy'))
        self.assertFalse(rules.matches('recordings/abc/0_1.png'))
        self.assertFalse(rules.matches('items/silhouettes/coins.npy'))
        self.assertFalse(rules.matches('npc_minimap/test1/a.npy'))
        self.assertFalse(rules.matches('maps/meta/pollnivneach - 副本.pickle'))
        self.assertFalse(rules.matches('maps/meta/hosidius.pickle.backup'))


class TestPublish(AssetTestCase):

    def test_publish_uploads_deduplicated_blobs_and_manifest(self):
        result = self.publish()

        self.assertEqual(result.total_files, 6)
        self.assertEqual(len(result.new), 6)
        self.assertEqual(result.uploaded_blobs, 5)  # dup_a == dup_b
        manifest = json.loads(read(os.path.join(self.remote, MANIFEST_KEY)))
        self.assertEqual(
            manifest['files']['bank/dup_a.npy']['sha256'], hash_bytes(b'same'))
        self.assertTrue(self.storage.exists(blob_key(hash_bytes(b'frame'))))

    def test_republish_reports_changes(self):
        self.publish()
        write(self.source, 'tabs/inventory/frame.npy', b'frame v2')
        os.remove(os.path.join(self.source, 'bank', 'dup_b.npy'))
        write(self.source, 'bank/new.npy', b'new')

        result = self.publish()

        self.assertEqual(result.changed, ['tabs/inventory/frame.npy'])
        self.assertEqual(result.removed, ['bank/dup_b.npy'])
        self.assertEqual(result.new, ['bank/new.npy'])
        self.assertEqual(result.uploaded_blobs, 2)
        # old blob kept for rollback
        self.assertTrue(self.storage.exists(blob_key(hash_bytes(b'frame'))))

    def test_dry_run_writes_nothing(self):
        before = sorted(os.walk(self.source))
        result = AssetManager(self.source, self.storage).publish(
            RULES, dry_run=True)
        self.assertEqual(result.total_files, 6)
        self.assertFalse(os.path.exists(self.remote))
        self.assertEqual(sorted(os.walk(self.source)), before)


class TestEnsure(AssetTestCase):

    def setUp(self):
        super().setUp()
        self.publish()

    def test_missing_file_is_downloaded(self):
        key = 'tabs/inventory/frame.npy'
        path = self.client_path(key)
        self.client().ensure(path)
        self.assertEqual(read(path), b'frame')

    def test_relative_path_accepted(self):
        self.client().ensure('maps/meta/gielinor.pickle')
        self.assertEqual(
            read(self.client_path('maps/meta/gielinor.pickle')), b'graph')

    def test_modified_file_is_restored(self):
        key = 'tabs/inventory/frame.npy'
        path = write(self.client_dir, key, b'stale')
        self.client().ensure(path)
        self.assertEqual(read(path), b'frame')

    def test_up_to_date_file_not_redownloaded(self):
        key = 'tabs/inventory/frame.npy'
        path = self.client_path(key)
        self.client().ensure(path)

        manager = self.client()
        with mock.patch.object(
                self.storage, 'get_bytes', wraps=self.storage.get_bytes) as m:
            manager.ensure(path)
            manager.ensure(path)
        # only the conditional manifest request, no blob fetch
        self.assertEqual([c.args[0] for c in m.call_args_list], [MANIFEST_KEY])

    def test_unchanged_manifest_uses_304(self):
        self.client().ensure('tabs/inventory/frame.npy')

        manager = self.client()
        returned = []
        original = self.storage.get_bytes

        def spy(key, if_none_match=None):
            result = original(key, if_none_match=if_none_match)
            returned.append((key, if_none_match, result[0]))
            return result

        with mock.patch.object(self.storage, 'get_bytes', side_effect=spy):
            manager.refresh_manifest()
        key, etag, body = returned[0]
        self.assertEqual(key, MANIFEST_KEY)
        self.assertIsNotNone(etag)
        self.assertIsNone(body)  # 304: unchanged, no body transferred
        self.assertIn('tabs/inventory/frame.npy', manager.manifest.files)

    def test_manifest_refreshed_once_per_process(self):
        manager = self.client()
        with mock.patch.object(
                self.storage, 'get_bytes', wraps=self.storage.get_bytes) as m:
            manager.ensure('tabs/inventory/frame.npy')
            manager.ensure('tabs/inventory/frame_mask.npy')
            manager.ensure('bank/dup_a.npy')
        manifest_calls = [c for c in m.call_args_list
                          if c.args[0] == MANIFEST_KEY]
        self.assertEqual(len(manifest_calls), 1)

    def test_new_manifest_picked_up(self):
        self.client().ensure('tabs/inventory/frame.npy')
        write(self.source, 'tabs/inventory/frame.npy', b'frame v2')
        self.publish()

        path = self.client_path('tabs/inventory/frame.npy')
        self.client().ensure(path)
        self.assertEqual(read(path), b'frame v2')

    def test_not_in_manifest_left_alone(self):
        path = write(self.client_dir, 'tabs/local_only.npy', b'mine')
        manager = self.client()
        manager.ensure(path)
        manager.ensure(os.path.join(self.client_dir, 'tabs', 'nope.npy'))
        self.assertEqual(read(path), b'mine')
        self.assertFalse(os.path.exists(
            os.path.join(self.client_dir, 'tabs', 'nope.npy')))

    def test_path_outside_data_dir_ignored(self):
        outside = write(self.tmp, 'elsewhere/x.npy', b'x')
        self.assertEqual(self.client().ensure(outside), outside)
        self.assertEqual(read(outside), b'x')

    def test_storage_error_falls_back_to_disk(self):
        key = 'tabs/inventory/frame.npy'
        path = write(self.client_dir, key, b'offline copy')
        broken = mock.MagicMock()
        broken.get_bytes.side_effect = ConnectionError('no network')

        result = self.client(storage=broken).ensure(path)

        self.assertEqual(result, path)
        self.assertEqual(read(path), b'offline copy')

    def test_offline_uses_cached_manifest(self):
        self.client().ensure('tabs/inventory/frame.npy')
        broken = mock.MagicMock()
        broken.get_bytes.side_effect = [
            ConnectionError('no network'),  # manifest
            ConnectionError('no network'),  # blob
        ]
        manager = self.client(storage=broken)
        self.assertIsNotNone(manager.refresh_manifest())
        manager.ensure('tabs/inventory/frame_mask.npy')  # must not raise
        self.assertFalse(os.path.exists(
            self.client_path('tabs/inventory/frame_mask.npy')))

    def test_corrupt_blob_rejected(self):
        sha = hash_bytes(b'frame')
        self.storage.put_bytes(blob_key(sha), b'tampered')
        path = self.client_path('tabs/inventory/frame.npy')
        self.client().ensure(path)
        self.assertFalse(os.path.exists(path))

    def test_ensure_glob(self):
        keys = self.client().ensure_glob(
            os.path.join(self.client_dir, 'bank', 'dup_*'))
        self.assertEqual(sorted(keys), ['bank/dup_a.npy', 'bank/dup_b.npy'])
        self.assertEqual(read(self.client_path('bank/dup_b.npy')), b'same')

    def test_sync_and_status(self):
        manager = self.client()
        self.assertEqual(manager.sync(prefix='tabs/'), (3, 0))
        self.assertTrue(os.path.exists(self.client_path('tabs/inventory/alpha.png')))
        self.assertFalse(os.path.exists(self.client_path('bank/dup_a.npy')))

        write(self.client_dir, 'tabs/inventory/frame.npy', b'edited')
        write(self.client_dir, 'tabs/extra.npy', b'extra')
        status = manager.status(RULES)
        self.assertEqual(status.modified, ['tabs/inventory/frame.npy'])
        self.assertEqual(sorted(status.missing), [
            'bank/dup_a.npy', 'bank/dup_b.npy', 'maps/meta/gielinor.pickle'])
        self.assertEqual(status.local_only, ['tabs/extra.npy'])
        self.assertEqual(status.ok, 2)


class TestDisabled(TestCase):

    def test_no_storage_is_noop(self):
        manager = AssetManager(tempfile.gettempdir(), None)
        self.assertEqual(manager.ensure('a/b.npy'), 'a/b.npy')
        self.assertIsNone(manager.refresh_manifest())
        self.assertEqual(manager.ensure_glob('a/*'), [])

    def test_config_env_overrides_file(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        path = os.path.join(tmp, 'assets.local.json')
        with open(path, 'w') as f:
            json.dump({'backend': 'r2', 'r2': {'bucket': 'from-file',
                                               'account_id': 'acc'}}, f)
        config = AssetConfig.load(path, environ={
            'WIZARD_ASSETS_R2_BUCKET': 'from-env'})
        self.assertEqual(config.backend, 'r2')
        self.assertEqual(config.r2['bucket'], 'from-env')
        self.assertEqual(config.r2['account_id'], 'acc')

        empty = AssetConfig.load(os.path.join(tmp, 'missing.json'), environ={})
        self.assertFalse(empty.enabled)
        self.assertIsNone(empty.create_storage())

    def test_local_backend_from_env(self):
        config = AssetConfig.load('missing.json', environ={
            'WIZARD_ASSETS_BACKEND': 'local',
            'WIZARD_ASSETS_LOCAL_DIR': '/x'})
        self.assertIsInstance(config.create_storage(), LocalStorage)
