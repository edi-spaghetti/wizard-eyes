"""Build map data (collision, objects, transport guesses) from the game cache.

    python utility/map_data.py build [--cache-dir DIR] [--data-dir DIR]
    python utility/map_data.py status
    python utility/map_data.py transports status
    python utility/map_data.py transports merge

Reads the cache RuneLite keeps up to date (~/.runelite/jagexcache/oldschool/
LIVE by default) and writes data/maps/collision/movement.npz (movement and
doors), data/maps/objects/objects.npz (named / usable objects) and
data/maps/transports/guessed.npz (guessed stair / ladder destinations).
Rebuild after each game update (``status`` says when), then publish with
``utility/assets.py publish``.

``transports merge`` folds transports recorded in game on this machine
(data/maps/transports/recorded.local.json in the repo) into the curated
data/maps/transports/transports.json of --data-dir, ready to publish.
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
from wizard_eyes.cache.collision import CollisionBuilder  # noqa: E402
from wizard_eyes.cache.object_data import build_object_data  # noqa: E402
from wizard_eyes.cache.maps import MAP_INDEX, load_regions  # noqa: E402
from wizard_eyes.cache.objects import CONFIG_INDEX, load_objects  # noqa: E402
from wizard_eyes.cache.store import CacheStore, default_cache_dir  # noqa: E402
from wizard_eyes.file_path_utils import get_root  # noqa: E402
from wizard_eyes.navigation import (  # noqa: E402
    collision_map, object_map, transports,
)

OUTPUTS = {
    'collision': collision_map.DEFAULT_PATH[len('data/'):],
    'objects': object_map.DEFAULT_PATH[len('data/'):],
    'transport guesses': transports.GUESSED_PATH[len('data/'):],
}


def cache_revisions(store: CacheStore) -> dict:
    return {
        'maps': store.table(MAP_INDEX).revision,
        'configs': store.table(CONFIG_INDEX).revision,
    }


def write_npz(out: str, arrays: dict):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + '.tmp.npz'
    numpy.savez_compressed(tmp, **arrays)
    os.replace(tmp, out)
    print(f'Wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)')


def build(args):
    start = time.time()
    with CacheStore(args.cache_dir) as store:
        revisions = cache_revisions(store)
        print(f'Cache: {store.path} {revisions}')
        objects = load_objects(store)
        print(f'Object definitions: {len(objects)}')
        errors = {}
        builder = CollisionBuilder(objects)

        def regions():
            # one pass over the regions feeds both builds
            for region in load_regions(store, errors):
                builder.add_region(region)
                yield region

        placements, definitions = build_object_data(regions(), objects)
        movement, doors = builder.all_masks()

    print(f'Regions: {len(movement)} ({len(doors)} with doors)')
    for rid, err in errors.items():
        print(f'  skipped region {rid}: {err}')
    print(f'Objects: {len(placements["id"])} placements, '
          f'{len(definitions)} definitions')

    meta = {
        'built_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'cache': revisions,
        'regions': len(movement),
        'skipped': sorted(errors),
    }
    arrays = {f'm{rid}': grid for rid, grid in movement.items()}
    arrays.update({f'd{rid}': grid for rid, grid in doors.items()})
    arrays['meta'] = numpy.array(json.dumps(meta))
    write_npz(os.path.join(args.data_dir, OUTPUTS['collision']), arrays)

    arrays = dict(placements)
    arrays['definitions'] = numpy.array(json.dumps(definitions))
    arrays['meta'] = numpy.array(json.dumps(
        dict(meta, objects=len(placements['id']))))
    write_npz(os.path.join(args.data_dir, OUTPUTS['objects']), arrays)

    om = object_map.ObjectMap(os.path.join(args.data_dir, OUTPUTS['objects']))
    cm = collision_map.CollisionMap(
        os.path.join(args.data_dir, OUTPUTS['collision']))
    guesses = transports.guess_transports(om, cm)
    out = os.path.join(args.data_dir, OUTPUTS['transport guesses'])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    transports.save_guesses(out, guesses, dict(meta, transports=len(guesses)))
    print(f'Transport guesses: {len(guesses)}; wrote {out} '
          f'({os.path.getsize(out) / 1e6:.1f} MB)')
    print(f'Done in {time.time() - start:.0f}s')


def status(args):
    with CacheStore(args.cache_dir) as store:
        current = cache_revisions(store)
    print(f'Game cache: {current}')
    result = 0
    for name, rel in OUTPUTS.items():
        out = os.path.join(args.data_dir, rel)
        if not os.path.exists(out):
            print(f'{name}: missing ({out}): run build')
            result = 1
            continue
        meta = collision_map.read_meta(out)
        state = 'up to date' if meta['cache'] == current else 'OUT OF DATE'
        print(f'{name}: {meta["cache"]} built {meta["built_at"]}: {state}')
        if meta['cache'] != current:
            result = 1
    if result:
        print('Run build, then publish')
    return result


def transports_cmd(args):
    curated = os.path.join(args.data_dir,
                           transports.CURATED_PATH[len('data/'):])
    recorded = os.path.join(get_root(), transports.RECORDED_PATH)
    if args.action == 'merge':
        added, updated, warnings = transports.merge_recordings(
            curated, recorded)
        for w in warnings:
            print(f'  warning: {w}')
        print(f'Merged {recorded} into {curated}: {added} added, '
              f'{updated} updated')
        if os.path.exists(recorded):
            os.replace(recorded, recorded + '.merged')
            print(f'Local recordings moved to {recorded}.merged; publish '
                  f'with utility/assets.py --data-dir {args.data_dir} '
                  f'publish')
        return 0
    guessed = os.path.join(args.data_dir, OUTPUTS['transport guesses'])
    t = transports.Transports(guessed=guessed, curated=curated,
                              recorded=recorded, min_confidence=0.0)
    by_source = {}
    for x in t:
        by_source[x.source] = by_source.get(x.source, 0) + 1
    print(f'Transports: {len(t)} ({by_source})')
    for x in t:
        if x.source != 'guess':
            print(f'  {x.source:8} {x.name!r} {x.action} {x.object_tile} '
                  f'-> {x.to} (x{x.count})')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--cache-dir', default=str(default_cache_dir()),
                        help='Game cache directory (default: %(default)s)')
    parser.add_argument('--data-dir', default=default_data_dir(),
                        help='Data directory to write to '
                             '(default: %(default)s)')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('build', help='Build collision and object data from '
                                 'the cache')
    sub.add_parser('status', help='Check the data matches the game cache')
    tp = sub.add_parser('transports', help='Recorded / curated transports')
    tp.add_argument('action', choices=('status', 'merge'))
    args = parser.parse_args(argv)
    return {'build': build, 'status': status,
            'transports': transports_cmd}[args.command](args) or 0


if __name__ == '__main__':
    sys.exit(main())
