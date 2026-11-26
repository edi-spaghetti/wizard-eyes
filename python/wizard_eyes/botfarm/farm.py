"""A farm's definition, as the botfarm PowerShell scripts read it
(``botfarm/lib/BotFarm.psm1`` ``Get-Farm``).

The farm folder: the one given, else ``$WIZARD_EYES_FARM`` (set for a farm's
services), else this machine's default farm
(``%LOCALAPPDATA%\\wizard-eyes\\botfarm.json``, set by ``botfarm/setup.ps1``).
"""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

from ..file_path_utils import get_root

ENV = 'WIZARD_EYES_FARM'
DEFAULT_CREDENTIAL = 'wizard-eyes/telegram'


class NoFarm(Exception):
    """No farm given and none registered, or no farm.json there."""


def machine_file() -> Path:
    return Path(os.environ.get('LOCALAPPDATA', '')) / 'wizard-eyes' / \
        'botfarm.json'


def farm_folder(folder: Union[str, Path, None] = None) -> Path:
    if not folder:
        folder = os.environ.get(ENV)
    if not folder:
        try:
            folder = json.loads(machine_file().read_text('utf-8-sig')).get(
                'default')
        except (OSError, ValueError, AttributeError):
            folder = None
    if not folder:
        raise NoFarm('no farm given, and no default farm on this machine '
                     '(botfarm\\setup.ps1 sets one)')
    folder = Path(folder)
    if not (folder / 'farm.json').is_file():
        raise NoFarm(f'no farm.json in {folder}')
    return folder.resolve()


@dataclass
class Farm:
    name: str
    folder: Path
    state_dir: Path
    vms: Dict[str, str] = field(default_factory=dict)
    """role -> VM name"""
    telegram_credential: str = DEFAULT_CREDENTIAL

    @property
    def roles(self):
        return tuple(self.vms)

    @classmethod
    def load(cls, folder: Union[str, Path, None] = None,
             engine: Optional[Path] = None) -> 'Farm':
        folder = farm_folder(folder)
        d = json.loads((folder / 'farm.json').read_text('utf-8-sig'))
        engine = Path(engine or get_root())
        alerts = d.get('alerts') or {}
        return cls(
            name=d.get('name') or folder.name,
            folder=folder,
            state_dir=engine / d.get('state_dir', 'data/botfarm'),
            vms={role: r.get('vm') for role, r in
                 (d.get('roles') or {}).items()},
            telegram_credential=alerts.get('credential') or
            DEFAULT_CREDENTIAL)


def state_folder(folder: Union[str, Path, None],
                 farm: Union[str, Path, None] = None) -> Path:
    """For the commands: ``--folder`` if given, else the farm's state
    folder."""
    return Path(folder) if folder else Farm.load(farm).state_dir
