"""Readable biological acquisition and LF-centred pair indexes for cache sets."""
from collections import Counter
import csv
import json
from pathlib import Path


def _csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_biology_catalog(ps_root, tf_root, output_root):
    ps_root, tf_root, output_root = map(Path, (ps_root, tf_root, output_root))
    ps = json.loads((ps_root / 'manifest.json').read_text(encoding='utf-8'))
    tf = json.loads((tf_root / 'manifest.json').read_text(encoding='utf-8'))
    if tf['contract']['ps_fingerprint'] != ps['fingerprint']:
        raise ValueError('PS/TF cache fingerprints do not match')
    output_root.mkdir(parents=True, exist_ok=True)
    pairs = []
    for row in ps['samples']:
        name = row['acquisition']
        target = tf['acquisitions'].get(name)
        present = set(target.get('retained_plane_indices', [])) if target else set()
        indices = range(row['first_slice'], row['last_slice'] + 1)
        pairs.append(dict(pair_id=row['name'], acquisition=name, specimen=row['specimen'], split=row['split'],
                          eligible=row['eligible'], lf_filename=Path(row['lf_path']).name,
                          centre_plane=row['lf_index'], first_plane=row['first_slice'], last_plane=row['last_slice'],
                          layer_count=row['last_slice'] - row['first_slice'] + 1,
                          lf_source=row['lf_path'], ps_source=row['ps_directory'],
                          ps_lf_cache=str(ps_root / row['cache_file']),
                          lf_hdf5_dataset=f"/lf/{row['lf_index']}", ps_hdf5_dataset='/ps',
                          tf_cache=str(tf_root / target['file']) if target else '', tf_hdf5_dataset='/tf' if target else '',
                          missing_tf_planes=';'.join(str(i) for i in indices if i not in present) if target else 'not cached: no eligible pairs',
                          qc_reason=row['alignment']['reason']))
    acquisitions = []
    for name, entry in ps['acquisitions'].items():
        local = [p for p in pairs if p['acquisition'] == name]
        usable = sum(r['eligible'] for r in local)
        target = tf['acquisitions'].get(name)
        acquisitions.append(dict(acquisition=name, specimen=entry['specimen'], split=entry['split'],
                                 candidate_pairs=len(local), usable_pairs=usable, rejected_pairs=len(local)-usable,
                                 lf_filenames=';'.join(r['lf_filename'] for r in local),
                                 ps_lf_file=entry['file'], tf_file=target['file'] if target else '',
                                 tf_status='cached' if target else 'not cached: no eligible pairs'))
    _csv(output_root / 'pairs.csv', pairs)
    _csv(output_root / 'acquisitions.csv', acquisitions)
    summary = dict(data_kind='biological samples only; no calibration beads',
                   ps_lf_acquisition_files=len(acquisitions), tf_acquisition_files=len(tf['acquisitions']),
                   candidate_pairs=len(pairs), usable_pairs=sum(p['eligible'] for p in pairs),
                   rejected_pairs=sum(not p['eligible'] for p in pairs),
                   usable_splits=dict(Counter(p['split'] for p in pairs if p['eligible'])),
                   ps_root=str(ps_root), tf_root=str(tf_root))
    (output_root / 'catalog.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    depth = ps['contract']['config']['depth']
    half = depth // 2
    (output_root / 'README.md').write_text(
        '# Biological LF / PS / TF training cache\n\n'
        'This cache contains biological samples. Bead calibration data are stored separately under '
        '`F:/00_MicoscopyDatasets/03_Light Field/CalibrationSamples`.\n\n'
        f"There are {summary['candidate_pairs']} candidate LF-centred pairs from {len(acquisitions)} acquisitions. "
        f"{summary['usable_pairs']} pairs pass QC; {summary['rejected_pairs']} are retained in the index with rejection reasons. "
        f"Usable split counts: {summary['usable_splits']}.\n\n"
        f"`lf_ps/` stores {len(acquisitions)} acquisition files: one native PS stack and a group of selected LF frames per file. "
        f"`tf_targets/` stores {len(tf['acquisitions'])} native TF acquisition stacks. "
        'An acquisition with no usable pairs has no TF cache file.\n\n'
        '`pairs.csv` has one row for every LF-centred pair, with source files, HDF5 paths, split, QC and missing TF planes. '
        '`acquisitions.csv` has one row per acquisition and the number of pairs it supplies.\n\n'
        f'LF filename j identifies the centre PS/TF plane j. The volume is planes j-{half} through j+{half} inclusive ({depth} layers). '
        f'For LF {half:06d}.mat, use planes 000-{depth-1:03d}; '
        f'for LF {half+20:06d}.mat, use planes 020-{depth+19:03d}. '
        'Missing TF planes are masked in their original positions.\n\n'
        'The cache is a snapshot. New source data must be registered and refreshed into a new cache version.\n',
        encoding='utf-8')
    return summary
