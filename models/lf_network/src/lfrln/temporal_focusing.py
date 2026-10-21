"""Fixed updated TF response and a separate, immutable registered TF cache.

TF is an observation of the final raw emitter volume, not a voxel GT replacement
and not an H1 training target in this comparison. Convolution uses ascending
registered z coordinates. No extra square, reflection, or GT edge replication.
"""
from pathlib import Path
import hashlib
import json
import re

import h5py
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

from .point_scanning import PSForwardOperator
from .indexed_experimental import IndexedPSDataset


def read_tf_response(path):
    path = Path(path)
    with h5py.File(path, 'r') as f:
        kernel = np.asarray(f['I_emPSF_TF'], dtype=np.float64).transpose(0, 2, 1).copy()
        spacing = (float(f['dz_Sample'][()].item())*1e6,
                   float(f['dx_Sample'][()].item())*1e6, float(f['dx_Sample'][()].item())*1e6)
    if kernel.ndim != 3 or any(n % 2 != 1 for n in kernel.shape):
        raise ValueError('Expected odd 3D TF response support')
    if not np.isfinite(kernel).all() or kernel.min() < 0 or kernel.sum() <= 0:
        raise ValueError('Invalid TF intensity response')
    mass = float(kernel.sum())
    kernel /= mass
    record = dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  matlab_variable='I_emPSF_TF', axes='MATLAB Y,X,Z -> HDF5 Z,X,Y -> torch Z,Y,X',
                  shape=list(kernel.shape), spacing_um=list(spacing), original_sum=mass,
                  response='Already combined emission intensity times TF excitation profile; no extra squaring',
                  orientation='As supplied for ascending registered z; no symmetry-based flip',
                  reflection_relative_l2=float(np.linalg.norm(kernel-kernel[::-1, ::-1, ::-1])/np.linalg.norm(kernel)),
                  target='T(final raw emitter volume) compared with measured registered TF; H1 unchanged')
    return torch.from_numpy(kernel.astype(np.float32)), record


def tf_operator(path, shape, *, device='cpu', expected_spacing_um=None, minimum_coverage=.99):
    kernel, record = read_tf_response(path)
    if expected_spacing_um is not None and not np.allclose(record['spacing_um'], expected_spacing_um, atol=1e-8, rtol=0):
        raise ValueError('TF response and chosen registered grid differ')
    operator = PSForwardOperator(kernel, shape, device=device, spacing_um=record['spacing_um'],
                                 minimum_coverage=minimum_coverage)
    operator.source_record = record
    return operator


def _stamp(path):
    path = Path(path)
    stat = path.stat()
    return dict(path=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def build_tf_cache(ps_root, output_root, *, grid_note, scan_order='ascending', max_acquisitions=None, allow_partial_stacks=False):
    """Never modify the accepted PS cache or silently substitute another TF window.

    All eligible PS/LF frames must have TF; no changing splits/QC to favor a TF
    comparison. The acquisition-local display normalization is a background
    heuristic, not a dark calibration. Raw uint16 TF is retained in the cache.
    """
    ps_root, output_root = Path(ps_root), Path(output_root)
    manifest = json.loads((ps_root/'manifest.json').read_text(encoding='utf8'))
    if scan_order != 'ascending':
        raise ValueError('This registered pairing requires ascending slice indices; resolve export order explicitly')
    rows = [r for r in manifest['samples'] if r['eligible']]
    names = sorted({r['acquisition'] for r in rows})
    if max_acquisitions is not None:
        names = names[:max_acquisitions]
    files, acquisitions = [], {}
    for name in names:
        local = [r for r in rows if r['acquisition'] == name]
        directory = Path(local[0]['tf_first']).parent
        paths = [directory/f'TF_slice_{i:03d}.tif' for i in range(131)]
        if not allow_partial_stacks and len(list(directory.glob('TF_slice_*.tif'))) != 131:
            raise ValueError(f'Expected exactly 131 registered TF slices: {directory}')
        for row in local:
            if Path(row['tf_first']) != paths[row['first_slice']] or Path(row['tf_last']) != paths[row['last_slice']]:
                raise ValueError('TF pairing differs from exact LF-centered PS pairing')
        metadata = directory.parent/'parameters.txt'
        metadata_text = metadata.read_text(encoding='utf8', errors='replace')
        status = re.search(r'Data status:\s*<([^>]*)>', metadata_text, re.I)
        if not status or status.group(1).strip().lower() != 'good':
            raise ValueError(f'TF acquisition is not marked Good: {metadata}')
        confinement = re.search(r'Illumination Axial Confinment[^\n]*\n\s*<([^>]*)>', metadata_text)
        acquisitions[name] = dict(directory=str(directory), metadata=str(metadata),
                                  confinement_metadata=confinement.group(1) if confinement else None,
                                  split=local[0]['split'], specimen=local[0]['specimen'])
        retained=[i for i,p in enumerate(paths) if p.is_file()]
        if not retained:raise ValueError(f'No retained TF planes: {name}')
        acquisitions[name]['retained_plane_indices']=retained
        files.extend(_stamp(p) for p in [metadata, *paths] if p.is_file())
    contract = dict(schema='registered_tf_comparison_v1', ps_fingerprint=manifest['fingerprint'],
                    grid_note=grid_note, scan_order=scan_order, sources=files,
                    pairs=[{k:r[k] for k in ('name','acquisition','lf_index','first_slice','last_slice','split')}
                           for r in rows if r['acquisition'] in acquisitions],
                    intensity='Raw uint16 retained. q10 subtraction/clamp and q99.9 range only for shape comparison/display; no TF dark calibration.',
                    max_acquisitions=max_acquisitions,allow_partial_stacks=allow_partial_stacks)
    fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    dest = output_root/'manifest.json'
    if dest.exists():
        existing = json.loads(dest.read_text(encoding='utf8'))
        if existing['fingerprint'] != fingerprint:
            raise ValueError('TF data/grid/pairing changed; use a new TF cache directory')
        for value in existing['acquisitions'].values():
            if not (output_root/value['file']).is_file():
                raise FileNotFoundError(output_root/value['file'])
        return existing
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f'Incomplete TF cache: use a fresh directory: {output_root}')
    output_root.mkdir(parents=True, exist_ok=True)
    for index, (name, row) in enumerate(acquisitions.items()):
        retained=row['retained_plane_indices']
        values=np.stack([tifffile.imread(Path(row['directory'])/f'TF_slice_{i:03d}.tif') for i in retained])
        if values.dtype!=np.uint16 or values.shape[1:]!=(517,517):raise ValueError('Invalid retained TF planes')
        stack=np.zeros((131,517,517),dtype=np.uint16);stack[retained]=values
        if stack.shape != (131, 517, 517) or stack.dtype != np.uint16:
            raise ValueError(f'TF shape/dtype differs from the registered contract: {name}: {stack.shape}, {stack.dtype}')
        background, maximum = np.quantile(values, [.1, .999])
        scale = float(maximum-background)
        if scale <= 0:
            raise ValueError(f'TF has no usable contrast: {name}')
        row.update(file=f'{name}.h5', background=float(background), scale=scale,
                   saturation_fraction=float(np.mean(values == 65535)))
        with h5py.File(output_root/row['file'], 'w') as f:
            f.create_dataset('tf', data=stack, chunks=(1,517,517), compression='lzf')
            f.attrs['acquisition'] = name
        print(f'TF cache {index+1}/{len(acquisitions)}: {name}', flush=True)
    result = dict(fingerprint=fingerprint, contract=contract, acquisitions=acquisitions)
    dest.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf8')
    return result


class IndexedPSTFComparison(Dataset):
    def __init__(self, ps_root, tf_root, *, split):
        self.ps = IndexedPSDataset(ps_root, split=split)
        self.root = Path(tf_root)
        self.manifest = json.loads((self.root/'manifest.json').read_text(encoding='utf8'))
        if self.manifest['contract']['ps_fingerprint'] != self.ps.manifest['fingerprint']:
            raise ValueError('TF cache was paired against a different PS cache')
        if any(r['acquisition'] not in self.manifest['acquisitions'] for r in self.ps.samples):
            raise ValueError('TF cache is incomplete for this split')

    def __len__(self):
        return len(self.ps)

    def __getitem__(self, index):
        sample = self.ps[index]
        row = self.manifest['acquisitions'][sample['acquisition']]
        with h5py.File(self.root/row['file'], 'r') as f:
            if f.attrs['acquisition'] != sample['acquisition']:
                raise ValueError('TF cache acquisition mismatch')
            tf = f['tf'][sample['first_slice']:sample['last_slice']+1].astype(np.float32)
        if tf.shape != tuple(sample['gt'].shape):
            raise ValueError('TF and PS windows differ')
        sample['tf_reference'] = torch.from_numpy(np.maximum(tf-row['background'], 0.)/row['scale'])
        available=set(row.get('retained_plane_indices',range(131)))
        valid=torch.tensor([i in available for i in range(sample['first_slice'],sample['last_slice']+1)],dtype=torch.bool)
        sample['tf_valid_mask']=valid[:,None,None].expand_as(sample['tf_reference'])
        if not valid.any():raise ValueError('No measured TF planes in this window')
        return sample
