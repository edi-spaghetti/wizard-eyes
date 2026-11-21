"""Manage remote assets for the data directory.

    python utility/assets.py publish [--dry-run]
    python utility/assets.py sync [--prefix tabs/]
    python utility/assets.py status

Storage is configured in config/assets.local.json or WIZARD_ASSETS_* env
vars, see config/assets.example.json and readme.md.
"""
import argparse
import logging
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))

from wizard_eyes.assets import AssetManager, IncludeRules  # noqa: E402
from wizard_eyes.assets.config import (  # noqa: E402
    AssetConfig, default_data_dir, default_rules_path,
)


def mb(n: int) -> str:
    return f'{n / 1e6:.1f} MB'


def print_list(label, items, limit):
    print(f'{label}: {len(items)}')
    for item in items[:limit]:
        print(f'  {item}')
    if len(items) > limit:
        print(f'  ... and {len(items) - limit} more')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--data-dir', default=default_data_dir(),
                        help='Local data directory (default: %(default)s)')
    parser.add_argument('--config', default=None,
                        help='Config file (default: config/assets.local.json)')
    parser.add_argument('--rules', default=default_rules_path(),
                        help='Include/exclude rules (default: %(default)s)')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--limit', type=int, default=20,
                        help='Max paths to list per category')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('publish', help='Upload changed assets and manifest')
    p.add_argument('--dry-run', action='store_true',
                   help='Show what would happen without writing anything')
    p = sub.add_parser('sync', help='Download all (or some) assets')
    p.add_argument('--prefix', default='',
                   help='Only sync keys starting with this, e.g. tabs/')
    sub.add_parser('status', help='Compare local files with the manifest')

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(levelname)s %(message)s')

    config = AssetConfig.load(args.config)
    storage = config.create_storage()
    if storage is None and not (args.command == 'publish' and args.dry_run):
        parser.error('No asset backend configured. Copy '
                     'config/assets.example.json to config/assets.local.json '
                     'or set WIZARD_ASSETS_BACKEND.')

    manager = AssetManager(args.data_dir, storage)
    rules = IncludeRules.load(args.rules)
    print(f'Data dir: {manager.data_dir}')
    print(f'Storage:  {storage!r}')

    if args.command == 'publish':
        r = manager.publish(rules, dry_run=args.dry_run)
        print_list('New', r.new, args.limit)
        print_list('Changed', r.changed, args.limit)
        print_list('Removed', r.removed, args.limit)
        print(f'Unchanged: {r.unchanged}')
        verb = 'Would upload' if args.dry_run else 'Uploaded'
        print(f'Manifest: {r.total_files} files, {mb(r.total_bytes)}')
        print(f'{verb} {r.uploaded_blobs} blobs, {mb(r.uploaded_bytes)}')
    elif args.command == 'sync':
        downloaded, failed = manager.sync(args.prefix)
        print(f'Downloaded {downloaded}, failed {failed}')
        return 1 if failed else 0
    elif args.command == 'status':
        r = manager.status(rules)
        print(f'Up to date: {r.ok}')
        print_list('Missing locally', r.missing, args.limit)
        print_list('Modified locally', r.modified, args.limit)
        print_list('Local only (not published)', r.local_only, args.limit)
    return 0


if __name__ == '__main__':
    sys.exit(main())
