"""Check the active RuneLite profile has the settings the engine relies on.

RuneLite keeps profiles in ``~/.runelite/profiles2``: ``profiles.json`` lists
them (the active one has ``"active": true``) and each profile's settings live
in ``<name>-<id>.properties``.

The settings the engine cares about are listed in
``config/runelite_required.json``. Only those keys are compared, so the rest of
your profile (bank tags, menu swaps, other plugins) can be whatever you like.
"""
import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .file_path_utils import get_root

logger = logging.getLogger('wizard_eyes.runelite_profile')

REQUIRED = 'required'
RECOMMENDED = 'recommended'


def default_runelite_dir() -> str:
    return os.environ.get(
        'WIZARD_RUNELITE_DIR',
        os.path.join(os.path.expanduser('~'), '.runelite'))


def default_rules_path() -> str:
    return os.path.join(get_root(), 'config', 'runelite_required.json')


# --------------------------------------------------------------------------
# .properties parsing
# --------------------------------------------------------------------------

def _unescape(s: str) -> str:
    out = []
    i = 0
    while i < len(s):
        c = s[i]
        if c != '\\' or i == len(s) - 1:
            out.append(c)
            i += 1
            continue
        n = s[i + 1]
        if n == 'u' and i + 5 < len(s):
            try:
                out.append(chr(int(s[i + 2:i + 6], 16)))
                i += 6
                continue
            except ValueError:
                pass
        out.append({'n': '\n', 't': '\t', 'r': '\r', 'f': '\f'}.get(n, n))
        i += 2
    return ''.join(out)


def _split_key(line: str):
    """Split a logical properties line at the first unescaped = or :."""
    i = 0
    while i < len(line):
        c = line[i]
        if c == '\\':
            i += 2
            continue
        if c in '=:':
            return line[:i], line[i + 1:]
        i += 1
    return line, ''


def parse_properties(text: str) -> Dict[str, str]:
    """Parse a Java .properties file (enough of the format for RuneLite)."""
    result = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].lstrip()
        i += 1
        if not line or line[0] in '#!':
            continue
        # join continuation lines (odd number of trailing backslashes)
        while (len(line) - len(line.rstrip('\\'))) % 2 == 1 and i < len(lines):
            line = line[:-1] + lines[i].lstrip()
            i += 1
        key, value = _split_key(line)
        result[_unescape(key.strip())] = _unescape(value.strip())
    return result


def load_properties(path: str) -> Dict[str, str]:
    with open(path, 'r', encoding='latin-1') as f:
        return parse_properties(f.read())


# --------------------------------------------------------------------------
# finding the active profile
# --------------------------------------------------------------------------

@dataclass
class Profile:
    name: str
    id: int
    path: str


def active_profile(runelite_dir: Optional[str] = None) -> Optional[Profile]:
    """Return the profile RuneLite has marked active, or None."""
    runelite_dir = runelite_dir or default_runelite_dir()
    profiles_dir = os.path.join(runelite_dir, 'profiles2')
    index = os.path.join(profiles_dir, 'profiles.json')
    if not os.path.isfile(index):
        return None
    with open(index, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for p in data.get('profiles', []):
        if p.get('active'):
            path = os.path.join(profiles_dir, f"{p['name']}-{p['id']}.properties")
            return Profile(name=p['name'], id=p['id'], path=path)
    return None


# --------------------------------------------------------------------------
# importing a profile (part of setting a machine up)
# --------------------------------------------------------------------------

class RuneLiteRunning(RuntimeError):
    """RuneLite rewrites its profile files when it closes: not while it runs."""


def runelite_running() -> bool:
    """Is a RuneLite client (RuneLite.exe) running on this machine?"""
    try:
        out = subprocess.run(['tasklist', '/FO', 'CSV', '/NH'],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             timeout=30).stdout.decode('utf-8', 'replace')
    except (OSError, subprocess.SubprocessError):
        return False
    return any(line.lower().startswith('"runelite.exe"')
               for line in out.splitlines())


def import_profile(properties_path: str, name: str = 'naughty',
                   runelite_dir: Optional[str] = None, activate: bool = True,
                   force: bool = False) -> Profile:
    """Make ``properties_path`` the RuneLite profile ``name`` (replacing one
    of that name) and, by default, the active one. Same files as the
    client's own "Import" button: ``profiles2/<name>-<id>.properties`` and its
    entry in ``profiles.json``.

    :param force: go ahead although RuneLite is running (it would overwrite
        the files when it closes).
    """
    if not os.path.isfile(properties_path):
        raise FileNotFoundError(properties_path)
    if not force and runelite_running():
        raise RuneLiteRunning('RuneLite is running: close it first')
    runelite_dir = runelite_dir or default_runelite_dir()
    profiles_dir = os.path.join(runelite_dir, 'profiles2')
    os.makedirs(profiles_dir, exist_ok=True)
    index = os.path.join(profiles_dir, 'profiles.json')
    data = {'profiles': []}
    if os.path.isfile(index):
        with open(index, 'r', encoding='utf-8') as f:
            data = json.load(f)
    profiles = data.setdefault('profiles', [])
    entry = next((p for p in profiles if p.get('name') == name), None)
    if entry is None:
        entry = {'id': int(time.time() * 1000), 'name': name, 'sync': False,
                 'active': False, 'rev': -1}
        profiles.append(entry)
    path = os.path.join(profiles_dir, f"{name}-{entry['id']}.properties")
    shutil.copyfile(properties_path, path)
    if activate:
        for p in profiles:
            p['active'] = p is entry
    tmp = index + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, separators=(',', ':'))
    os.replace(tmp, index)
    logger.info(f'RuneLite profile "{name}" imported'
                f'{" and activated" if activate else ""}')
    return Profile(name=name, id=entry['id'], path=path)


def fix_profile(path: str, rules: List['Rule']) -> List[str]:
    """Set the keys of a profile file that break an "expected" rule to the
    expected value (rules saying "!value" are left alone). Returns the keys
    changed. Not while RuneLite runs."""
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    props = parse_properties(text)
    newline = '\r\n' if '\r\n' in text else '\n'
    lines = text.split(newline)
    changed = []
    for rule in rules:
        if rule.expected.startswith('!'):
            continue
        actual = props.get(rule.key, rule.default)
        if actual is not None and actual.strip() == rule.expected:
            continue
        for i, line in enumerate(lines):
            if not line.startswith('#') and line.split('=', 1)[0] == rule.key:
                lines[i] = f'{rule.key}={rule.expected}'
                break
        else:
            lines.append(f'{rule.key}={rule.expected}')
        changed.append(rule.key)
    if changed:
        with open(path, 'w', encoding='utf-8', newline='') as f:
            f.write(newline.join(lines))
    return changed


# --------------------------------------------------------------------------
# checking settings
# --------------------------------------------------------------------------

@dataclass
class Rule:
    key: str
    expected: str
    level: str = REQUIRED
    default: Optional[str] = None
    """Value RuneLite uses when the key isn't in the profile, if known."""
    reason: str = ''


@dataclass
class Mismatch:
    rule: Rule
    actual: Optional[str]

    def __str__(self):
        actual = 'not set' if self.actual is None else repr(self.actual)
        if self.rule.expected.startswith('!'):
            text = (f'{self.rule.key}: must not be '
                    f'{self.rule.expected[1:]!r}, got {actual}')
        else:
            text = (f'{self.rule.key}: expected {self.rule.expected!r}, '
                    f'got {actual}')
        if self.rule.reason:
            text += f' ({self.rule.reason})'
        return text


@dataclass
class CheckResult:
    profile: Optional[Profile]
    mismatches: List[Mismatch] = field(default_factory=list)
    checked: int = 0
    error: str = ''

    @property
    def required_failures(self) -> List[Mismatch]:
        return [m for m in self.mismatches if m.rule.level == REQUIRED]

    @property
    def recommended_failures(self) -> List[Mismatch]:
        return [m for m in self.mismatches if m.rule.level != REQUIRED]

    @property
    def ok(self) -> bool:
        return not self.error and not self.required_failures


def load_rules(path: Optional[str] = None) -> List[Rule]:
    path = path or default_rules_path()
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    rules = []
    for level in (REQUIRED, RECOMMENDED):
        for entry in raw.get(level, []):
            rules.append(Rule(
                key=entry['key'],
                expected=str(entry['expected']),
                level=level,
                default=(None if entry.get('default') is None
                         else str(entry['default'])),
                reason=entry.get('reason', ''),
            ))
    return rules


def check_properties(props: Dict[str, str], rules: List[Rule]) -> List[Mismatch]:
    mismatches = []
    for rule in rules:
        actual = props.get(rule.key, rule.default)
        if rule.expected.startswith('!'):
            # "!value" means anything except value (unset is fine)
            bad = actual is not None and actual.strip() == rule.expected[1:]
        else:
            bad = actual is None or actual.strip() != rule.expected
        if bad:
            mismatches.append(Mismatch(rule, props.get(rule.key)))
    return mismatches


def check_active_profile(runelite_dir: Optional[str] = None,
                         rules_path: Optional[str] = None) -> CheckResult:
    """Compare the active RuneLite profile against the engine's rules."""
    try:
        rules = load_rules(rules_path)
    except (OSError, ValueError) as e:
        return CheckResult(None, error=f'Could not load profile rules: {e}')

    try:
        profile = active_profile(runelite_dir)
    except (OSError, ValueError) as e:
        return CheckResult(None, error=f'Could not read RuneLite profiles: {e}')
    if profile is None:
        return CheckResult(None, error='No active RuneLite profile found')

    try:
        props = load_properties(profile.path)
    except OSError as e:
        return CheckResult(profile, error=f'Could not read profile: {e}')

    return CheckResult(profile, check_properties(props, rules), len(rules))


def log_profile_check(log: Optional[logging.Logger] = None) -> CheckResult:
    """Run the check and log the outcome. Never raises."""
    log = log or logger
    try:
        result = check_active_profile()
    except Exception as e:  # never stop a bot over this
        log.warning(f'RuneLite profile check failed: {e}')
        return CheckResult(None, error=str(e))

    if result.error:
        log.warning(f'RuneLite profile check skipped: {result.error}')
        return result

    name = result.profile.name
    for m in result.required_failures:
        log.warning(f'RuneLite profile "{name}" setting mismatch: {m}')
    for m in result.recommended_failures:
        log.info(f'RuneLite profile "{name}" recommended setting: {m}')
    if result.ok:
        log.info(f'RuneLite profile "{name}" OK '
                 f'({result.checked} settings checked)')
    return result
