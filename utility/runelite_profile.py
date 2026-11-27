"""Check the active RuneLite profile against config/runelite_required.json.

Usage:
    python utility/runelite_profile.py            # check the active profile
    python utility/runelite_profile.py --file X   # check a .properties file
    python utility/runelite_profile.py --import-profile [X] [--name N]
        # import X (default: the published reference profile, data/runelite/
        # naughty.properties) as the active profile N (default naughty), unless
        # it already is the active one and passes (RuneLite must be closed)

Exit code is 0 if all required settings match, 1 otherwise.
"""
import argparse
import os
import sys

from wizard_eyes.file_path_utils import ensure_local, get_root
from wizard_eyes.runelite_profile import (
    CheckResult, Profile, RuneLiteRunning, active_profile,
    check_active_profile, check_properties, import_profile, load_properties,
    load_rules,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--runelite-dir', help='defaults to ~/.runelite')
    parser.add_argument('--rules', help='defaults to config/runelite_required.json')
    parser.add_argument('--file', help='check this .properties file instead '
                                       'of the active profile')
    parser.add_argument('--import-profile', nargs='?', const='', metavar='FILE',
                        help='import FILE (default: the published reference '
                             'profile) as the active profile, unless that is '
                             'already so and it passes the check')
    parser.add_argument('--name', default='naughty',
                        help='profile name for --import-profile')
    args = parser.parse_args(argv)

    if args.import_profile is not None:
        current = active_profile(args.runelite_dir)
        if current and current.name == args.name and os.path.isfile(current.path):
            if check_active_profile(args.runelite_dir, args.rules).ok:
                print(f'Profile {args.name} is already active and passes')
                return 0
        source = args.import_profile or os.path.join(
            get_root(), 'data', 'runelite', 'naughty.properties')
        ensure_local(source)
        try:
            profile = import_profile(source, args.name, args.runelite_dir)
        except (RuneLiteRunning, FileNotFoundError) as e:
            print(f'ERROR: {e}')
            return 1
        print(f'Imported {profile.name} ({profile.path})')
        args.file = None

    if args.file:
        rules = load_rules(args.rules)
        result = CheckResult(
            Profile(name=args.file, id=0, path=args.file),
            check_properties(load_properties(args.file), rules), len(rules))
    else:
        result = check_active_profile(args.runelite_dir, args.rules)

    if result.error:
        print(f'ERROR: {result.error}')
        return 1

    print(f'Profile:  {result.profile.name}')
    print(f'Checked:  {result.checked} settings')
    for label, items in (('Required mismatches', result.required_failures),
                         ('Recommended mismatches', result.recommended_failures)):
        print(f'{label}: {len(items)}')
        for m in items:
            print(f'  {m}')
    print('OK' if result.ok else 'FAIL')
    return 0 if result.ok else 1


if __name__ == '__main__':
    sys.exit(main())
