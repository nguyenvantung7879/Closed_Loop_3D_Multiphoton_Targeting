"""Explicit local source and cache locations for biological experimental data."""
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class ExperimentalStorage:
    source_root: Path
    biology_cache_root: Path
    current_cache_version: str
    corrected_lf_folder: str

    def cache_paths(self, version=None):
        name = version or self.current_cache_version
        if not name or name in ('.', '..') or any(c in name for c in '/\\:'):
            raise ValueError('Cache version must be one directory name')
        root = self.biology_cache_root / name
        return root / 'lf_ps', root / 'tf_targets'


def load_experimental_storage(path=None):
    config = Path(path) if path is not None else Path(__file__).resolve().parents[2] / 'experimental_storage.json'
    settings = json.loads(config.read_text(encoding='utf-8'))
    return ExperimentalStorage(Path(settings['source_root']), Path(settings['biology_cache_root']),
                               settings['current_cache_version'], settings['corrected_lf_folder'])
