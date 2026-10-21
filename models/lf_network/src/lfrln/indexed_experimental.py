"""Full-frame registered LF/PS pairs with one axial window per LF filename.

The cache stores each native PS stack once. Frames at different LF indices
are separate examples, never repeats of a single target. No spatial transform,
per-frame LF normalization, or additional LF noise is applied here.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import h5py
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

from .data import load_lf_tensor
from .paired_experimental import (
    best_integer_registration, block_reduce_2d, extract_specimen_id,
    registered_reference_window,
)

SCHEMA = 'lfrln.indexed_fullframe_ps.v1'


@dataclass(frozen=True)
class IndexedPSConfig:
    pitch: int = 11
    shape: tuple[int, int] = (517, 517)
    depth: int = 91
    stack_depth: int = 131
    allow_partial_stacks: bool = False
    lf_count_scale: float = 3000.0
    ps_background_quantile: float = .10
    ps_scale_quantile: float = .999
    validation_specimens: tuple[str, ...] = ('JY06', 'F04')
    testing_specimens: tuple[str, ...] = ('F01', 'JY09', 'F06')
    # Gross-misalignment exclusion is an input-only QC heuristic, not proof
    # of 3D registration. No correction is silently fitted or applied.
    minimum_projection_correlation: float = .15
    alignment_improvement_tolerance: float = .08
    maximum_shift_lenslets: int = 2
    lf_folder: str | None = None

    def __post_init__(self):
        if self.depth <= 0 or self.depth % 2 != 1 or self.stack_depth < self.depth:
            raise ValueError('Require a positive odd target depth within the stack')
        if self.pitch <= 0 or any(v <= 0 or v % self.pitch for v in self.shape):
            raise ValueError('LF shape must contain complete lenslets')
        if not np.isfinite(self.lf_count_scale) or self.lf_count_scale <= 0:
            raise ValueError('LF count scale must be finite and positive')
        if not 0 <= self.ps_background_quantile < self.ps_scale_quantile <= 1:
            raise ValueError('Invalid PS quantiles')
        if set(self.validation_specimens) & set(self.testing_specimens):
            raise ValueError('Validation/testing specimen groups overlap')
        if self.lf_folder is not None and (not self.lf_folder or self.lf_folder in ('.', '..')
                                           or '/' in self.lf_folder or '\\' in self.lf_folder
                                           or ':' in self.lf_folder):
            raise ValueError('LF folder must be one relative directory name')


def _field(text: str, name: str) -> str:
    match = re.search(re.escape(name) + r':?\s*\n\s*<([^>]+)>', text)
    return match.group(1).strip() if match else ''


def _stamp(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {'path': str(path.resolve()), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def discover_indexed_ps(root: str | Path, config: IndexedPSConfig) -> tuple[list[dict], list[dict]]:
    root = Path(root)
    records, excluded = [], []
    lf_root = root / '03_Light Field/Biosamples'
    if not lf_root.is_dir():
        raise FileNotFoundError(lf_root)
    for directory in sorted(lf_root.iterdir()):
        lf_directory = directory / (config.lf_folder or str(config.pitch))
        lf_paths = sorted(lf_directory.glob('*.mat'))
        if not lf_paths:
            continue
        name = directory.name
        metadata_path = directory / 'parametersLF.txt'
        text = metadata_path.read_text(encoding='utf-8-sig')
        status = _field(text, 'Data status')
        if any(word in status.lower() for word in ('fail', 'bad', 'unusable')):
            excluded.append({'acquisition': name, 'frames': len(lf_paths), 'reason': status})
            continue
        specimen = extract_specimen_id(name)
        if _field(text, 'Sample ID') != specimen:
            raise ValueError(f'Specimen metadata/filename mismatch: {name}')
        ps_directory = root / '01_Point Scanning/Biosamples' / name / 'VR_TIFF_slices'
        ps_paths = [ps_directory / f'VR_slice_{i:03d}.tif' for i in range(config.stack_depth)]
        if not config.allow_partial_stacks and any(not p.is_file() for p in ps_paths):
            raise FileNotFoundError(f'Incomplete PS stack: {ps_directory}')
        ps_metadata = ps_directory.parent / 'parameters.txt'
        ps_text = ps_metadata.read_text(encoding='utf-8-sig')
        if any(word in _field(ps_text, 'Data status').lower() for word in ('fail', 'bad', 'unusable')):
            excluded.append({'acquisition': name, 'frames': len(lf_paths), 'reason': 'PS metadata failed'})
            continue
        split = ('validation' if specimen in config.validation_specimens else
                 'testing' if specimen in config.testing_specimens else 'training')
        for lf_path in lf_paths:
            index = int(lf_path.stem)
            first,last=index-config.depth//2,index+config.depth//2
            if config.allow_partial_stacks and (first<0 or last>=config.stack_depth or any(not ps_paths[i].is_file() for i in range(max(0,first),min(config.stack_depth,last+1)))):
                excluded.append(dict(acquisition=name,frames=1,lf_index=index,reason='Incomplete retained PS window; no renumbering or invented target planes'))
                continue
            window = registered_reference_window(ps_directory, index, prefix='VR_slice_', depth=config.depth)
            tf_directory = root / '02_Temporal Focusing/Biosamples' / name / 'TF_TIFF_slices'
            tf_window = (tuple(tf_directory/f'TF_slice_{i:03d}.tif' for i in range(first,last+1)) if config.allow_partial_stacks else registered_reference_window(tf_directory, index, prefix='TF_slice_', depth=config.depth)) if tf_directory.is_dir() else ()
            records.append({
                'name': f'{name}__lf{index:06d}', 'acquisition': name, 'specimen': specimen,
                'split': split, 'lf_index': index, 'first_slice': index - config.depth // 2,
                'last_slice': index + config.depth // 2, 'lf_path': str(lf_path),
                'ps_directory': str(ps_directory), 'ps_first': str(window[0]), 'ps_last': str(window[-1]),
                'tf_first': str(tf_window[0]) if tf_window else None,
                'tf_last': str(tf_window[-1]) if tf_window else None,
                'ps_output_voxel_size': _field(ps_text, 'Output voxel size'),
                'lf_metadata': str(metadata_path), 'ps_metadata': str(ps_metadata),
            })
            registration = lf_directory / 'registration_manifest.json'
            if registration.is_file():
                records[-1]['lf_registration_manifest'] = str(registration)
    if not records:
        raise ValueError('No indexed registered LF/PS pairs found')
    present = {r['specimen'] for r in records}
    missing = (set(config.validation_specimens) | set(config.testing_specimens)) - present
    if missing and not config.allow_partial_stacks:
        raise ValueError(f'Requested held-out specimens are missing: {sorted(missing)}')
    return records, excluded


def _alignment(lf: np.ndarray, ps: np.ndarray, config: IndexedPSConfig) -> dict:
    a = block_reduce_2d(lf, (config.pitch, config.pitch), reduction='sum')
    b = block_reduce_2d(ps.sum(axis=0), (config.pitch, config.pitch), reduction='mean')
    corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.std() > 0 and b.std() > 0 else 0.
    best = best_integer_registration(a, b, max_shift=3)
    gain = best['correlation'] - corr
    gross = gain > config.alignment_improvement_tolerance and (
        best['orientation'] != 'rot0' or
        max(abs(best['shift_x']), abs(best['shift_y'])) > config.maximum_shift_lenslets)
    # _orientation_variants names the unchanged orientation 'rot0'.
    eligible = np.isfinite(corr) and corr >= config.minimum_projection_correlation and not gross
    return {'zero_shift_correlation': corr, 'best': best, 'improvement': gain,
            'eligible': bool(eligible), 'reason': ('accepted coarse QC' if eligible else
             'low projection correlation or gross orientation/shift discrepancy')}


def _write_audit(root: Path, manifest: dict) -> None:
    rows = []
    for sample in manifest['samples']:
        alignment = sample['alignment']
        rows.append({**{key: sample[key] for key in
            ('name', 'specimen', 'split', 'lf_index', 'first_slice', 'last_slice', 'eligible')},
            **sample['lf_statistics'], 'correlation': alignment['zero_shift_correlation'],
            'best_orientation': alignment['best']['orientation'],
            'best_dy': alignment['best']['shift_y'], 'best_dx': alignment['best']['shift_x'],
            'correlation_improvement': alignment['improvement']})
    with (root / 'pair_audit.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_indexed_ps_cache(
    source_root: str | Path, output_root: str | Path, *, config: IndexedPSConfig,
    registered_baseline: np.ndarray, baseline_provenance: dict,
) -> dict:
    """Build/reuse an immutable, fingerprinted acquisition cache.

    Source sizes/mtime and configuration are checked on reuse. Original PS
    uint16 pixels and raw LF counts are retained; baseline subtraction and
    the fixed divisor are performed once when a sample is read.
    """
    output_root = Path(output_root)
    baseline = np.asarray(registered_baseline, dtype=np.float32)
    if baseline.shape != config.shape or not np.isfinite(baseline).all():
        raise ValueError('Registered baseline must be finite and match LF shape')
    records, excluded = discover_indexed_ps(source_root, config)
    paths = {Path(r[k]) for r in records for k in ('lf_path', 'lf_metadata', 'ps_metadata')}
    paths.update(Path(r['lf_registration_manifest']) for r in records if 'lf_registration_manifest' in r)
    for directory in {r['ps_directory'] for r in records}:
        paths.update(p for i in range(config.stack_depth) if (p:=Path(directory)/f'VR_slice_{i:03d}.tif').is_file())
    config_contract = asdict(config)
    if config.lf_folder is None:
        config_contract.pop('lf_folder')  # Keep default cache fingerprints compatible.
    contract = {'schema': SCHEMA, 'config': config_contract, 'sources': [_stamp(p) for p in sorted(paths)],
                'baseline_sha256': hashlib.sha256(baseline.tobytes()).hexdigest(),
                'baseline_provenance': baseline_provenance,
                'pairing': [{k: r[k] for k in ('name', 'specimen', 'split', 'first_slice', 'last_slice')}
                            for r in records]}
    fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    manifest_path = output_root / 'manifest.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('fingerprint') != fingerprint:
            raise ValueError('Indexed cache sources/configuration changed; choose a new cache directory')
        for cache in manifest['acquisitions'].values():
            if not (output_root / cache['file']).is_file():
                raise FileNotFoundError(output_root / cache['file'])
        _write_audit(output_root, manifest)
        return manifest
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f'Incomplete/existing cache; choose a new directory: {output_root}')
    output_root.mkdir(parents=True, exist_ok=True)
    np.save(output_root / 'registered_baseline.npy', baseline)
    acquisitions = {}
    names = sorted({r['acquisition'] for r in records})
    for number, name in enumerate(names):
        local = [r for r in records if r['acquisition'] == name]
        directory = Path(local[0]['ps_directory'])
        retained=[i for i in range(config.stack_depth) if (directory/f'VR_slice_{i:03d}.tif').is_file()]
        values=np.stack([tifffile.imread(directory/f'VR_slice_{i:03d}.tif') for i in retained])
        if values.dtype!=np.uint16 or values.shape[1:]!=config.shape:raise ValueError('Invalid retained PS planes')
        stack=np.zeros((config.stack_depth,*config.shape),dtype=np.uint16);stack[retained]=values
        if stack.shape != (config.stack_depth, *config.shape) or stack.dtype != np.uint16:
            raise ValueError(f'Expected native uint16 PS stack, got {stack.shape}/{stack.dtype}: {name}')
        background = float(np.quantile(values, config.ps_background_quantile))
        scale = float(np.quantile(values, config.ps_scale_quantile)) - background
        if scale <= 0:
            raise ValueError(f'PS target has no usable intensity range: {name}')
        cache_file = f'{name}.h5'
        with h5py.File(output_root / cache_file, 'w') as handle:
            handle.create_dataset('ps', data=stack, compression='lzf', chunks=(1, *config.shape))
            group = handle.create_group('lf')
            for row in local:
                lf = load_lf_tensor(row['lf_path'], 'LF')[0].numpy().astype(np.float32)
                if lf.shape != config.shape or not np.isfinite(lf).all():
                    raise ValueError(f'Invalid registered LF counts: {row["lf_path"]}')
                group.create_dataset(str(row['lf_index']), data=lf, compression='lzf')
                signal = np.maximum(lf - baseline, 0.)
                window = np.maximum(stack[row['first_slice']:row['last_slice'] + 1].astype(np.float32)
                                    - background, 0.)
                row['alignment'] = _alignment(signal, window, config)
                row['eligible'] = row['alignment']['eligible']
                row['lf_statistics'] = {'raw_max': float(lf.max()), 'raw_mean': float(lf.mean()),
                    'input_max': float(signal.max() / config.lf_count_scale),
                    'input_q999': float(np.quantile(signal, .999) / config.lf_count_scale),
                    'zero_fraction': float(np.mean(signal == 0)),
                    'saturated_fraction': float(np.mean(lf >= 65535.))}
                if row['lf_statistics']['saturated_fraction'] > .001:
                    row['eligible'] = False
                    row['alignment']['reason'] = 'More than 0.1% saturated LF pixels'
                row['cache_file'] = cache_file
            handle.attrs['acquisition'] = name
            handle.attrs['ps_background'] = background
            handle.attrs['ps_scale'] = scale
        acquisitions[name] = {'file': cache_file, 'specimen': local[0]['specimen'],
            'split': local[0]['split'], 'ps_background': background, 'ps_scale': scale,'retained_plane_indices':retained}
        print(f'Cached {number + 1}/{len(names)}: {name} | '
              f'{sum(r["eligible"] for r in local)}/{len(local)} pass coarse QC', flush=True)
    counts = dict(Counter(r['split'] for r in records if r['eligible']))
    if any(counts.get(split, 0) == 0 for split in ('training', 'validation', 'testing')):
        raise ValueError(f'Each specimen-separated split needs usable pairs: {counts}')
    manifest = {'schema': SCHEMA, 'fingerprint': fingerprint, 'contract': contract,
        'samples': records, 'acquisitions': acquisitions, 'excluded_metadata': excluded,
        'split_counts': counts, 'all_pair_count': len(records),
        'notes': ['PS stacks stored once; indexed LF frames are not repeats.',
                  'LF divisor is a provisional checkpoint-unit convention, not measured photon calibration.',
                  'Coarse 2D QC does not certify axial registration or target modality equivalence.',
                  'TF paths are reference-only; TF values do not enter training or selection.']}
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
    _write_audit(output_root, manifest)
    return manifest


class IndexedPSDataset(Dataset):
    """Read a single LF and its exact centered PS window from an indexed cache."""
    def __init__(self, root: str | Path, *, split: str, include_metadata: bool = False):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text(encoding='utf-8'))
        if self.manifest['schema'] != SCHEMA:
            raise ValueError('Require the indexed full-frame PS cache schema')
        self.config = self.manifest['contract']['config']
        self.samples = [r for r in self.manifest['samples'] if r['split'] == split and r['eligible']]
        if not self.samples:
            raise ValueError(f'No usable pairs for split {split}')
        self.baseline = np.load(self.root / 'registered_baseline.npy')
        if hashlib.sha256(self.baseline.tobytes()).hexdigest() != self.manifest['contract']['baseline_sha256']:
            raise ValueError('Cached registered baseline changed')
        self.include_metadata = include_metadata

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]
        with h5py.File(self.root / row['cache_file'], 'r') as handle:
            if handle.attrs['acquisition'] != row['acquisition']:
                raise ValueError('Cached acquisition does not match the indexed pair')
            raw = np.asarray(handle['lf'][str(row['lf_index'])], dtype=np.float32)
            target = np.asarray(handle['ps'][row['first_slice']:row['last_slice'] + 1], dtype=np.float32)
            target = np.maximum(target - float(handle.attrs['ps_background']), 0.) / float(handle.attrs['ps_scale'])
        lf = np.maximum(raw - self.baseline, 0.) / self.config['lf_count_scale']
        if lf.shape != tuple(self.config['shape']) or not np.isfinite(lf).all():
            raise ValueError(f'Invalid cached LF frame: {row["name"]}')
        if target.shape != (self.config['depth'], *self.config['shape']):
            raise ValueError(f'Incorrect indexed PS window: {row["name"]}')
        result = {'lf': torch.from_numpy(lf[None]), 'gt': torch.from_numpy(target),
                  'name': row['name'], 'acquisition': row['acquisition'], 'specimen': row['specimen'],
                  'lf_index': row['lf_index'], 'first_slice': row['first_slice'], 'last_slice': row['last_slice']}
        if self.include_metadata:
            result['metadata'] = row
        return result
