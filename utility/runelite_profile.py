"""Check the active RuneLite profile against config/runelite_required.json.

Usage:
    python utility/runelite_profile.py            # check the active profile
    python utility/runelite_profile.py --file X   # check a .properties file

Exit code is 0 if all required settings match, 1 otherwise.
"""
import argparse
import sys

from wizard_eyes.runelite_profile import (
    CheckResult, Profile, check_active_profile, check_properties,
    load_properties, load_rules,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--runelite-dir', help='defaults to ~/.runelite')
    parser.add_argument('--rules', help='defaults to config/runelite_required.json')
    parser.add_argument('--file', help='check this .properties file instead '
                                       'of the active profile')
    args = parser.parse_args(argv)

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
