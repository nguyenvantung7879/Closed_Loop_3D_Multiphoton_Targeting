"""Notebook 17's reproducible LF comparisons, without any training side effects."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gc
import hashlib
import json
import time

import h5py
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from . import SyntheticPairDataset, load_h1_h2_training_checkpoint
from .indexed_experimental import IndexedPSDataset
from .paired_experimental import common_band_relative_l2
from .reconstruction_audit import (
    file_record, fingerprint, reconstruct_rl, reconstruct_phase_space,
)


FAMILIES = ('1um_spheres', '2um_spheres', '10um_shells', 'multiscale_lines',
            'random_spheres', 'random_shells')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=str), encoding='utf-8')


@dataclass
class AuditCase:
    key: str
    lf: torch.Tensor                    # [1,Y,X], already in network signal units
    target: torch.Tensor | None        # known artificial emitter only
    ps_reference: torch.Tensor | None  # never scored as raw emitter truth
    provenance: dict

    def __post_init__(self):
        if self.provenance['kind'] == 'measured_biology' and self.target is not None:
            raise ValueError('Measured PS is a modality reference, not a raw emitter target')
        if self.lf.ndim != 3 or self.lf.shape[0] != 1:
            raise ValueError('Require LF [1,Y,X]')


def case_catalog(root, *, split='validation', families=FAMILIES, synthetic_per_family=1,
                 biological_per_specimen=1, include_proxy=True):
    """Deterministic IDs; None means all samples. Split is never selected by metrics."""
    root = Path(root)
    syn = SyntheticPairDataset(root/'data/synthetic_python/h2_raw_clean_v2_aberr_517',
                               split=split, augment=False)
    paired = IndexedPSDataset(root/'data/paired_experimental/lf_ps_indexed_fullframe_em800_20260908_v1',
                              split=split)
    rows = []
    for family in families:
        ids = [i for i, p in enumerate(syn.files) if family in p.name]
        if not ids:
            raise ValueError(f'Missing family {family}')
        for i in ids if synthetic_per_family is None else ids[:synthetic_per_family]:
            rows.append(dict(key=f'syn_{Path(syn.files[i]).stem}', kind='synthetic',
                             split=split, family=family, index=i, name=Path(syn.files[i]).name,
                             source=file_record(syn.files[i])))
    counts = {}
    for i, row in enumerate(paired.samples):
        specimen = row['specimen']
        if biological_per_specimen is not None and counts.get(specimen, 0) >= biological_per_specimen:
            continue
        counts[specimen] = counts.get(specimen, 0) + 1
        base = dict(split=split, family='biological', index=i, name=row['name'],
                    specimen=specimen, acquisition=row['acquisition'], lf_index=row['lf_index'],
                    ps_window=[row['first_slice'], row['last_slice']],
                    cache_fingerprint=paired.manifest['fingerprint'],
                    source=file_record(paired.root/row['cache_file']),
                    baseline=file_record(paired.root/'registered_baseline.npy'))
        rows.append(dict(base, key=f'measured_{row["name"]}', kind='measured_biology'))
        if include_proxy:
            rows.append(dict(base, key=f'proxy_{row["name"]}', kind='ps_shaped_artificial'))
    return rows


def add_noise_cases(rows, *, peaks=(500.,3000.,12000.), seed=20260909):
    """Same deterministic EM800 realization is retained for every method in a case."""
    result=list(rows)
    for ordinal,row in enumerate(rows):
        if row['kind']!='synthetic':continue
        for level in peaks:
            result.append(dict(row,key=row['key']+f'_em800_peak{level:g}',
                               noise_peak_adu=float(level),noise_seed=seed+ordinal))
    return result


@torch.no_grad()
def load_case(root, row, operator):
    root = Path(root)
    if row['kind'] == 'synthetic':
        ds = SyntheticPairDataset(root/'data/synthetic_python/h2_raw_clean_v2_aberr_517',
                                  split=row['split'], augment=False)
        sample = ds[row['index']]
        target, reference = sample['gt'].float(), None
        # Reproject every artificial object through exactly the audit operator.
        lf = operator(target[None].to(operator.device))[0].cpu()
    else:
        ds = IndexedPSDataset(root/'data/paired_experimental/lf_ps_indexed_fullframe_em800_20260908_v1',
                              split=row['split'])
        sample = ds[row['index']]
        if sample['name'] != row['name']:
            raise ValueError('Pair index/name changed')
        if row['kind'] == 'measured_biology':
            lf, target, reference = sample['lf'].float(), None, sample['gt'].float()
        else:
            target, reference = sample['gt'].float(), None
            lf = operator(target[None].to(operator.device))[0].cpu()
    if 'noise_peak_adu' in row:
        from .datasets import MixedLFNoiseDataset
        from .noise_profiles import load_noise_profile
        camera,preprocessing,_=load_noise_profile(root/'data/noise/em800_2s_20260908_provisional/em800_2s_20260908_provisional.json')
        level=row['noise_peak_adu']
        noisy=MixedLFNoiseDataset([dict(lf=lf,lf_feedback=lf,gt=target)],noise_probability=1.,
            camera_config=camera,preprocess_config=preprocessing,deterministic=True,
            seed=row['noise_seed'],peak_signal_adu_range=(level,level))
        lf=noisy[0]['lf']
    return AuditCase(row['key'], lf, target, reference, row)


def morphology(volume, *, pitch=11, spacing=(.5, .21306818181818185, .21306818181818185)):
    a = volume.detach().float().cpu().numpy()
    xy = a.sum(0); total = max(float(xy.sum()), 1e-12)
    border = 2*pitch
    phase = xy.reshape(xy.shape[0]//pitch, pitch, xy.shape[1]//pitch, pitch).mean((0, 2))
    zp = a.sum((1, 2)); z = (np.arange(len(zp))-(len(zp)-1)/2)*spacing[0]
    return dict(edge_flux_fraction=float(1-xy[border:-border, border:-border].sum()/total),
                lenslet_phase_cv=float(phase.std()/max(float(phase.mean()), 1e-12)),
                axial_centroid_um=float((zp*z).sum()/max(float(zp.sum()), 1e-12)))


@torch.no_grad()
def score_volume(volume, case, operator, *, cutoff_domain=False):
    out = morphology(volume, pitch=operator.config.nnum)
    if cutoff_domain:
        return out
    v = volume[None].to(operator.device).float()
    y = case.lf[None].to(operator.device).float(); ay = operator(v)
    norm = lambda x: torch.linalg.vector_norm(x.float()).clamp_min(1e-12)
    scale = (ay*y).sum()/ay.square().sum().clamp_min(1e-12)
    out.update(lf_relative_l2=float(norm(ay-y)/norm(y)),
               lf_relative_l2_after_scalar_fit=float(norm(scale*ay-y)/norm(y)),
               fitted_lf_scale=float(scale))
    if case.target is not None:
        target = case.target[None].to(operator.device)
        border = 2*operator.config.nnum
        out['volume_relative_l2'] = float(norm(v-target)/norm(target))
        out['interior_volume_relative_l2'] = float(
            norm((v-target)[..., border:-border, border:-border]) /
            norm(target[..., border:-border, border:-border]))
        out['coarse_shape_mse'] = float(common_band_relative_l2(v, target))
    if case.ps_reference is not None:
        out['ps_reference_coarse_shape_mse'] = float(common_band_relative_l2(
            v, case.ps_reference[None].to(operator.device)))
    return out


def model_record(path):
    return dict(file_record(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def open_run(root, output, settings, catalog, checkpoints):
    """Changed inputs/settings/code require a new run; completed cases are reusable."""
    root, output = Path(root), Path(output)
    sources = [root/'src/lfrln/reconstruction_audit.py', Path(__file__),
               root/'data/kernels/lf_forward_40x_f11_aberr_08242026.bin',
               root/'data/kernels/phase_space_40x_f11_aberr_08242026_z91.h5',
               root/'data/synthetic_python/h2_raw_clean_v2_aberr_517/manifest.json']
    if any('noise_peak_adu' in row for row in catalog):
        sources.append(root/'data/noise/em800_2s_20260908_provisional/em800_2s_20260908_provisional.json')
    contract = dict(version=1, settings=settings, catalog=catalog,
                    checkpoints={k:model_record(p) for k,p in checkpoints.items()},
                    sources=[file_record(p) for p in sources], training_executed=False)
    if any(row['split'] == 'testing' for row in catalog):
        frozen = settings.get('validation_selection_record')
        if not frozen or not Path(frozen).is_file():
            raise ValueError('Testing requires a saved validation-selection JSON first')
        selection = json.loads(Path(frozen).read_text(encoding='utf-8'))
        for name in ('rl_iterations', 'phase_sweeps'):
            if selection[name] != settings[name]:
                raise ValueError(f'Testing {name} differs from validation selection')
        contract['validation_selection'] = model_record(frozen)
    token = fingerprint(contract)
    output.mkdir(parents=True, exist_ok=True)
    path = output/'contract.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))['fingerprint'] != token:
            raise ValueError('Run contract changed; choose a new OUTPUT directory')
    elif any(output.iterdir()):
        raise FileExistsError('Nonempty directory without an audit contract')
    else:
        write_json(path, dict(fingerprint=token, **contract))
    return token


def _save_array(handle, name, tensor):
    handle.create_dataset(name, data=tensor.detach().cpu().numpy().astype('f4'), compression='lzf')


@torch.no_grad()
def run_case(case, operator, phase_cache, checkpoints, output, settings, token, progress=print):
    output = Path(output); path = output/f'{case.key}.h5'; report_path = output/f'{case.key}.json'
    if path.exists() and report_path.exists():
        report = json.loads(report_path.read_text(encoding='utf-8'))
        with h5py.File(path) as f:
            if f.attrs.get('fingerprint') != token:
                raise ValueError('Saved case contract differs')
        if report['fingerprint'] != token:
            raise ValueError('Saved report contract differs')
        progress(f'Reusing {case.key}')
        return report
    temporary = path.with_suffix('.building.h5')
    report = dict(case=case.provenance, fingerprint=token, methods={},
                  lf_statistics=dict(mean=float(case.lf.mean()), peak=float(case.lf.max())),
                  target_role='known artificial emitter' if case.target is not None else 'PS modality reference only')
    with h5py.File(temporary, 'w') as f:
        f.attrs['fingerprint'] = token
        f.attrs['case'] = json.dumps(case.provenance)
        _save_array(f, 'lf', case.lf)
        if case.target is not None:
            _save_array(f, 'target', case.target)
            report['target_morphology'] = morphology(case.target)
        if case.ps_reference is not None:
            _save_array(f, 'ps_reference', case.ps_reference)
        def retain(name, volume, metadata, *, cutoff=False):
            _save_array(f, 'predictions/'+name, volume)
            metrics = score_volume(volume, case, operator, cutoff_domain=cutoff)
            report['methods'][name] = dict(metadata, **metrics,
                                          output_domain='cutoff' if cutoff else 'raw')
            if not cutoff:
                projected = operator(volume[None].to(operator.device))[0].cpu()
                _save_array(f, 'reprojected_lf/'+name, projected)
            progress(f'{case.key} | {name} | LF {metrics.get("lf_relative_l2", "cutoff domain")}')
        for name, checkpoint in checkpoints.items():
            model, payload = load_h1_h2_training_checkpoint(checkpoint, device='cpu')
            model = model.to(operator.device).eval()
            y = case.lf[None].to(operator.device)
            if operator.device.type == 'cuda':
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with torch.autocast(device_type=operator.device.type, enabled=operator.device.type=='cuda', dtype=torch.float16):
                stages = model.predict_stages(y)
            if operator.device.type == 'cuda': torch.cuda.synchronize()
            metadata = dict(seconds=time.perf_counter()-started, epoch=int(payload.get('epoch', -1)),
                            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if operator.device.type=='cuda' else None,
                            timing_scope='one inference, excludes checkpoint loading; no separate warmup')
            retain(name, stages['merged'][0].float().cpu(), metadata)
            if settings.get('save_stages', True):
                for stage in ('h1', 'h2'):
                    retain(name+'_'+stage, stages[stage][0].float().cpu(),
                           dict(parent_network=name), cutoff=stage=='h1')
            del model, payload, stages, y
            gc.collect(); torch.cuda.empty_cache()
        for update, name in [('volume', '3d_rl_volume'), ('measurement', '3d_rl_measurement')]:
            volume, metadata = reconstruct_rl(case.lf, operator,
                iterations=settings['rl_iterations'], update=update)
            metadata['implementation'] = ('local deconv_3D_RL.m volume-ratio update with matched A transpose; '
                'full frame retained, common interior metric provided' if update=='volume' else
                'measurement-ratio RL with A transpose and sensitivity normalization')
            retain(name, volume, metadata)
            del volume
        volume, metadata = reconstruct_phase_space(case.lf, phase_cache, gain=operator.gain,
            sweeps=settings['phase_sweeps'], device=operator.device)
        retain('phase_space_rl', volume, metadata)
        del volume
    temporary.replace(path)
    write_json(report_path, report)
    return report


def plot_case(path):
    import matplotlib.pyplot as plt
    path = Path(path)
    report = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    methods = [k for k in report['methods'] if not k.endswith(('_h1', '_h2'))]
    with h5py.File(path) as f:
        reference = 'target' if 'target' in f else 'ps_reference'
        names = [reference]+methods
        fig, axes = plt.subplots(3, len(names), figsize=(3.2*len(names), 9),
                                 gridspec_kw={'height_ratios':[2.4, 1, 1]}, layout='constrained')
        for column, name in enumerate(names):
            a = f[name if column==0 else 'predictions/'+name][:]
            z,y,x = a.shape; dz,dy,dx = .5,.21306818181818185,.21306818181818185
            planes = [(a.max(0), [-x*dx/2,x*dx/2,-y*dy/2,y*dy/2], 'y'),
                      (a.max(1), [-x*dx/2,x*dx/2,-z*dz/2,z*dz/2], 'z'),
                      (a.max(2), [-y*dy/2,y*dy/2,-z*dz/2,z*dz/2], 'z')]
            for row,(im,extent,label) in enumerate(planes):
                ax=axes[row,column]
                ax.imshow(im, origin='lower', extent=extent, aspect='equal', cmap='magma',
                          vmin=0, vmax=max(float(np.quantile(im,.999)),1e-12))
                ax.set_ylabel(label+' (µm)'); ax.set_xlabel(('y' if row==2 else 'x')+' (µm)')
            title = ('PS reference (filtered)' if name=='ps_reference' else
                     'Artificial emitter g' if name=='target' else name.replace('_',' '))
            axes[0,column].set_title(title,fontsize=10)
        fig.suptitle(report['case']['kind']+' | '+report['case']['name']+
            '\nXY / XZ / YZ MIPs; independent 99.9% display scales; common nominal LF grid')
        fig.savefig(path.with_suffix('.png'),dpi=110)
        lf = f['lf'][:].squeeze()
        residual_fig, ax = plt.subplots(2,len(methods)+1,figsize=(3.2*(len(methods)+1),6),layout='constrained')
        lim=max(float(np.quantile(lf,.999)),1e-12)
        ax[0,0].imshow(lf,cmap='gray',vmin=0,vmax=lim);ax[0,0].set_title('Observed LF')
        ax[1,0].axis('off')
        ax[1,0].text(0,.5,'Top: common observed scale\nBottom: (A x − y)/peak(y)\nUnfitted physical scale',fontsize=9)
        for c,name in enumerate(methods,1):
            ay=f['reprojected_lf/'+name][:].squeeze()
            ax[0,c].imshow(ay,cmap='gray',vmin=0,vmax=lim);ax[0,c].set_title(name.replace('_',' '),fontsize=9)
            ax[1,c].imshow((ay-lf)/max(float(lf.max()),1e-12),cmap='coolwarm',vmin=-1,vmax=1)
        for a in ax.flat:a.set_xticks([]);a.set_yticks([])
        residual_fig.savefig(path.with_name(path.stem+'_lf.png'),dpi=110)
    return fig,residual_fig


def perturb_emitter(target, *, brightness=1., smoothing_um=0., spacing=(.5,.21306818181818185,.21306818181818185)):
    """One-factor controls: physical object brightness OR flux-preserving artificial blur."""
    if brightness<=0 or smoothing_um<0 or (brightness!=1. and smoothing_um!=0.):
        raise ValueError('Use one positive brightness factor or one nonnegative blur at a time')
    if smoothing_um==0:
        return target.float()*brightness
    a=target.float().numpy()
    out=gaussian_filter(a, sigma=tuple(smoothing_um/s for s in spacing),mode='reflect')
    out*=float(a.sum(dtype=np.float64))/max(float(out.sum(dtype=np.float64)),1e-12)
    return torch.from_numpy(out)


@torch.no_grad()
def run_regressions(root, rows, operator, checkpoints, output, *, brightness=(.25,.5,1.,2.),
                    smoothing_um=(.5,1.), progress=print):
    """Metrics and compact stage MIPs on existing controls; no test-set selection."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    contract=dict(rows=rows, brightness=brightness,smoothing_um=smoothing_um,
                  checkpoints={k:model_record(p) for k,p in checkpoints.items()},
                  operator_gain=operator.gain,source=file_record(__file__),
                  audit_source=file_record(Path(__file__).with_name('reconstruction_audit.py')))
    token=fingerprint(contract);saved=output/'regression.json'
    if saved.exists():
        result=json.loads(saved.read_text(encoding='utf-8'))
        if result['fingerprint']!=token:raise ValueError('Changed regression contract; choose a new directory')
        return result['rows']
    result=[]
    variants=[dict(factor='brightness',value=b,brightness=b) for b in brightness]
    variants += [dict(factor='smoothing_um',value=b,smoothing_um=b) for b in smoothing_um]
    with h5py.File(output/'stage_mips.h5','w') as f:
        for model_name,path in checkpoints.items():
            model,payload=load_h1_h2_training_checkpoint(path,device='cpu')
            model=model.to(operator.device).eval()
            for row in rows:
                base=load_case(root,row,operator)
                if base.target is None:raise ValueError('Regressions require artificial emitter truth')
                for variant in variants:
                    target=perturb_emitter(base.target,**{k:v for k,v in variant.items() if k not in ('factor','value')})
                    lf=operator(target[None].to(operator.device))
                    case=AuditCase(base.key,lf[0].cpu(),target,None,base.provenance)
                    with torch.autocast(device_type=operator.device.type,enabled=operator.device.type=='cuda',dtype=torch.float16):
                        stages=model.predict_stages(lf)
                    for stage in ('h1','h2','merged'):
                        v=stages[stage][0].float().cpu()
                        metrics=score_volume(v,case,operator,cutoff_domain=stage=='h1')
                        record=dict(case=base.key,model=model_name,stage=stage,
                                    factor=variant['factor'],value=variant['value'],
                                    lf_mean=float(lf.mean()),lf_peak=float(lf.max()),**metrics)
                        result.append(record)
                        key=f'{model_name}/{base.key}/{variant["factor"]}_{variant["value"]}/{stage}'
                        _save_array(f,key+'/xy',v.max(0).values)
                        _save_array(f,key+'/xz',v.max(1).values)
                    progress(f'Regression {model_name} | {base.key} | {variant["factor"]}={variant["value"]}')
                    del stages,lf,v,target
                del base
            del model,payload
            gc.collect();torch.cuda.empty_cache()
    write_json(saved,dict(fingerprint=token,contract=contract,rows=result))
    return result


def plot_regressions(rows, output):
    import matplotlib.pyplot as plt
    cases=list(dict.fromkeys(r['case'] for r in rows));models=list(dict.fromkeys(r['model'] for r in rows))
    fig,axes=plt.subplots(len(cases),4,figsize=(16,3.5*len(cases)),squeeze=False,layout='constrained')
    for i,case in enumerate(cases):
        for j,(factor,metric) in enumerate([('brightness','volume_relative_l2'),('brightness','lenslet_phase_cv'),
                                           ('smoothing_um','volume_relative_l2'),('smoothing_um','lenslet_phase_cv')]):
            for model in models:
                data=[r for r in rows if r['case']==case and r['model']==model and r['stage']=='merged' and r['factor']==factor]
                if factor=='smoothing_um':
                    base=next(r for r in rows if r['case']==case and r['model']==model and r['stage']=='merged' and r['factor']=='brightness' and r['value']==1.)
                    data=[dict(base,value=0.)]+data
                axes[i,j].plot([r['value'] for r in data],[r[metric] for r in data],'o-',label=model)
            axes[i,j].set_xlabel(factor);axes[i,j].set_ylabel(metric);axes[i,j].grid(alpha=.2)
            axes[i,j].set_title(case[:60],fontsize=8)
            if i==0:axes[i,j].legend(fontsize=7)
    fig.suptitle('One factor at a time; unchanged emitter/LF scale convention; no noise added\n'
                 'Artificial smoothing is not an estimate of the PS imaging response')
    fig.savefig(Path(output)/'regression_curves.png',dpi=125)
    return fig
