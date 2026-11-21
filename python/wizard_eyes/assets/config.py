"""Asset storage configuration.

Settings come from ``config/assets.local.json`` (gitignored, see
``config/assets.example.json``), overridden by environment variables:

- WIZARD_ASSETS_BACKEND: ``r2``, ``local`` or empty/``none`` (disabled)
- WIZARD_ASSETS_LOCAL_DIR: folder used by the ``local`` backend
- WIZARD_ASSETS_R2_ACCOUNT_ID, WIZARD_ASSETS_R2_BUCKET,
  WIZARD_ASSETS_R2_ACCESS_KEY_ID, WIZARD_ASSETS_R2_SECRET_ACCESS_KEY

If no backend is configured, remote assets are disabled and the engine reads
local files exactly as before.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

from ..file_path_utils import get_root
from .storage import LocalStorage, R2Storage, Storage

ENV_PREFIX = 'WIZARD_ASSETS_'
R2_FIELDS = ('account_id', 'bucket', 'access_key_id', 'secret_access_key')


def default_config_path() -> str:
    return os.path.join(get_root(), 'config', 'assets.local.json')


def default_rules_path() -> str:
    return os.path.join(get_root(), 'config', 'assets_include.json')


def default_data_dir() -> str:
    return os.path.join(get_root(), 'data')


@dataclass
class AssetConfig:
    backend: str = ''
    local_dir: str = ''
    r2: Dict[str, str] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return self.backend not in ('', 'none')

    @classmethod
    def load(cls, path: Optional[str] = None,
             environ: Optional[Dict[str, str]] = None) -> 'AssetConfig':
        path = path or default_config_path()
        environ = os.environ if environ is None else environ

        raw = {}
        if os.path.isfile(path):
            with open(path, 'r', encoding='utf-8') as f:
                raw = json.load(f)

        backend = raw.get('backend', '') or ''
        local_dir = raw.get('local_dir', '') or ''
        r2 = {k: raw.get('r2', {}).get(k, '') for k in R2_FIELDS}

        backend = environ.get(ENV_PREFIX + 'BACKEND', backend)
        local_dir = environ.get(ENV_PREFIX + 'LOCAL_DIR', local_dir)
        for k in R2_FIELDS:
            r2[k] = environ.get(f'{ENV_PREFIX}R2_{k.upper()}', r2[k])

        return cls(backend=backend.strip().lower(), local_dir=local_dir, r2=r2)

    def create_storage(self) -> Optional[Storage]:
        if not self.enabled:
            return None
        if self.backend == 'local':
            if not self.local_dir:
                raise ValueError('local backend needs local_dir')
            return LocalStorage(self.local_dir)
        if self.backend == 'r2':
            missing = [k for k in R2_FIELDS if not self.r2.get(k)]
            if missing:
                raise ValueError(f'r2 backend missing settings: {missing}')
            return R2Storage(**{k: self.r2[k] for k in R2_FIELDS})
        raise ValueError(f'Unknown asset backend: {self.backend!r}')
