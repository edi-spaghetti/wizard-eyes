"""Build map data (collision) from the game cache.

    python utility/map_data.py build [--cache-dir DIR] [--data-dir DIR]
    python utility/map_data.py status

Reads the cache RuneLite keeps up to date (~/.runelite/jagexcache/oldschool/
LIVE by default) and writes data/maps/collision/movement.npz. Rebuild after
each game update (``status`` says when), then publish with
``utility/assets.py publish``.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))

from wizard_eyes.assets.config import default_data_dir  # noqa: E402
from wizard_eyes.cache.collision import build_movement  # noqa: E402
from wizard_eyes.cache.maps import MAP_INDEX, load_regions  # noqa: E402
from wizard_eyes.cache.objects import CONFIG_INDEX, load_objects  # noqa: E402
from wizard_eyes.cache.store import CacheStore, default_cache_dir  # noqa: E402
from wizard_eyes.navigation.collision_map import (  # noqa: E402
    DEFAULT_PATH, read_meta,
)

OUTPUT = DEFAULT_PATH[len('data/'):]


def cache_revisions(store: CacheStore) -> dict:
    return {
        'maps': store.table(MAP_INDEX).revision,
        'configs': store.table(CONFIG_INDEX).revision,
    }


def build(args):
    start = time.time()
    out = os.path.join(args.data_dir, OUTPUT)
    with CacheStore(args.cache_dir) as store:
        revisions = cache_revisions(store)
        print(f'Cache: {store.path} {revisions}')
        objects = load_objects(store)
        print(f'Object definitions: {len(objects)}')
        errors = {}
        movement, doors = build_movement(load_regions(store, errors), objects)

    print(f'Regions: {len(movement)} ({len(doors)} with doors)')
    for rid, err in errors.items():
        print(f'  skipped region {rid}: {err}')

    meta = {
        'built_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'cache': revisions,
        'regions': len(movement),
        'skipped': sorted(errors),
    }
    arrays = {f'm{rid}': grid for rid, grid in movement.items()}
    arrays.update({f'd{rid}': grid for rid, grid in doors.items()})
    arrays['meta'] = numpy.array(json.dumps(meta))

    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + '.tmp.npz'
    numpy.savez_compressed(tmp, **arrays)
    os.replace(tmp, out)
    print(f'Wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB) '
          f'in {time.time() - start:.0f}s')


def status(args):
    out = os.path.join(args.data_dir, OUTPUT)
    with CacheStore(args.cache_dir) as store:
        current = cache_revisions(store)
    print(f'Game cache: {current}')
    if not os.path.exists(out):
        print(f'No map data at {out}: run build')
        return 1
    meta = read_meta(out)
    print(f'Map data:   {meta["cache"]} (built {meta["built_at"]})')
    if meta['cache'] != current:
        print('Out of date: run build, then publish')
        return 1
    print('Up to date')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--cache-dir', default=str(default_cache_dir()),
                        help='Game cache directory (default: %(default)s)')
    parser.add_argument('--data-dir', default=default_data_dir(),
                        help='Data directory to write to '
                             '(default: %(default)s)')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('build', help='Build collision data from the cache')
    sub.add_parser('status', help='Check the data matches the game cache')
    args = parser.parse_args(argv)
    return {'build': build, 'status': status}[args.command](args) or 0


if __name__ == '__main__':
    sys.exit(main())
