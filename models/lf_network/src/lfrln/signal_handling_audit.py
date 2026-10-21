"""Frozen-checkpoint signal controls, independent of training and cache contracts.

Changes to normalization here are interventions on a trained model, not trained
replacement architectures. Calibration uses geometric TRAINING inputs only.
Biological-shaped artificial objects are used only for evaluation.
"""
from contextlib import contextmanager
from pathlib import Path
import gc,json
import h5py,numpy as np,torch
from . import load_h1_h2_training_checkpoint
from .reconstruction_audit import FixedLFOperator
from .reconstruction_benchmark import model_record
from .point_scanning import PSForwardOperator,ps_forward_relative_mse
from .indexed_experimental import IndexedPSDataset

VARIANTS = {
    'baseline':'Current network',
    'ratio_norm_off':'Remove normalized-ratio contribution',
    'bp_fixed_norm':'Fixed BP normalization (no retraining)',
    'signal_scale':'Normalize LF signal; restore output scale',
}


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf8')


def stats(value):
    a=value.detach().float()
    channel_var=a.var((-2,-1),unbiased=False)
    return dict(mean=float(a.mean()),std=float(a.std()),rms=float(a.square().mean().sqrt()),
                median_spatial_std=float(channel_var.sqrt().median()),
                fraction_channel_variance_below_1e_minus5=float((channel_var<1e-5).float().mean()))


def lf_stats(lf):
    a=lf.detach().float();peak=a.max().clamp_min(1e-12)
    return dict(mean=float(a.mean()),peak=float(peak),mean_over_peak=float(a.mean()/peak),
                fraction_above_tenth_peak=float((a>.1*peak).float().mean()))


def volume_scores(v,t):
    v,t=v.float(),t.float();energy=t.square().sum().clamp_min(1e-20)
    gain=(v*t).sum()/v.square().sum().clamp_min(1e-20)
    return dict(volume_relative_l2=float(((v-t).square().sum()/energy).sqrt()),
        fitted_output_gain=float(gain),shape_relative_l2=float(((v*gain-t).square().sum()/energy).sqrt()))


def cases(root,output):
    root,output=Path(root),Path(output)
    cache=root/'training_runs/experimental_auxcontext_boundary_aux_context_probe_20260915_v1/deconvolution_validation'
    items=[]
    for p in sorted(cache.glob('*.h5')):
        if p.stem.startswith(('syn_','proxy_','measured_')):
            kind='measured' if p.stem.startswith('measured_') else 'artificial'
            items.append(dict(name=p.stem,path=str(p),kind=kind,scattering=False if kind=='artificial' else None))
    # Extended fluorescence is a known geometric emitter, not a scattering model.
    base=cache/'syn_python_000002_10um_shells.h5'
    new_files=[output/f'shells_uniform_{k}.h5' for k in (1,4)]
    if not all(p.exists() for p in new_files):
        manifest=json.loads((root/'data/synthetic_python/h2_raw_clean_v2_aberr_517/manifest.json').read_text())
        op=FixedLFOperator(root/'data/kernels/lf_forward_40x_f11_aberr_08242026.bin',manifest['optical_gain'],device='cuda',dtype=torch.float16)
        with h5py.File(base) as f:target=torch.from_numpy(f['target'][:]).cuda().float()
        for multiplier,p in zip((1,4),new_files):
            if p.exists():continue
            gt=target+multiplier*target.mean()
            with torch.no_grad():lf=op(gt[None])[0].cpu().numpy()
            with h5py.File(p,'w') as f:
                f.create_dataset('target',data=gt.cpu().numpy(),compression='lzf')
                f.create_dataset('lf',data=lf,compression='lzf')
                f.attrs['description']='Geometric shells plus uniform fluorescence; same fixed LF operator, no scattering, no noise'
                f.attrs['uniform_level']=float(multiplier*target.mean())
            del gt
        del target,op;gc.collect();torch.cuda.empty_cache()
    items.extend(dict(name=p.stem,path=str(p),kind='artificial',scattering=False) for p in new_files)
    return items


@torch.no_grad()
def calibrate(model,root):
    folder=Path(root)/'data/synthetic_python/h2_raw_clean_v2_aberr_517/training'
    families=('1um_spheres','2um_spheres','10um_shells','multiscale_lines','random_spheres','random_shells')
    paths=[p for fam in families for p in sorted(folder.glob('*_'+fam+'.h5'))[:2]]
    if len(paths)!=12:raise ValueError('Require two geometric training images per family')
    first,second,rms=[],[],[]
    def collect(module,args):
        a=args[0].float();first.append(a.mean((0,2,3)).cpu());second.append(a.square().mean((0,2,3)).cpu())
    hook=model.h1.bp_output_norm.register_forward_pre_hook(collect)
    try:
        for path in paths:
            with h5py.File(path) as f:y=torch.from_numpy(f['lf'][:])[None].cuda().float()
            rms.append(float(y.square().mean().sqrt()))
            model.h1(y)
    finally:hook.remove()
    mean=torch.stack(first).mean(0);variance=(torch.stack(second).mean(0)-mean.square()).clamp_min(0)
    return dict(mean=mean[None,:,None,None].cuda(),variance=variance[None,:,None,None].cuda(),
                reference_lf_rms=float(np.median(rms)),training_files=[str(p) for p in paths])


@contextmanager
def intervene(model,variant,calibration):
    handles=[]
    if variant=='ratio_norm_off':
        handles.append(model.h1.ratio_norm.register_forward_hook(lambda module,args,result:torch.zeros_like(result)))
    elif variant=='bp_fixed_norm':
        def fixed(module,args,result):
            v=(args[0]-calibration['mean'])/torch.sqrt(calibration['variance']+module.eps)
            return v*module.weight[None,:,None,None]+module.bias[None,:,None,None]
        handles.append(model.h1.bp_output_norm.register_forward_hook(fixed))
    try:yield
    finally:
        for handle in handles:handle.remove()


@torch.no_grad()
def run_signal_audit(root,checkpoint,output,*,gains=(.25,1.,4.),progress=print):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=True)
    catalog=cases(root,output)
    model,payload=load_h1_h2_training_checkpoint(checkpoint,device='cpu');model=model.cuda().eval()
    calibration=calibrate(model,root)
    ps=PSForwardOperator.from_mat(root.parent/'forward model/PSFs/I_PSF_40XNA1_920nm2p_.mat',(91,517,517),device='cuda')
    record=dict(checkpoint=model_record(checkpoint),epoch=int(payload['epoch']),precision='FP32',
        variants=VARIANTS,gains=list(gains),calibration={k:v for k,v in calibration.items() if k not in ('mean','variance')},
        normalized_ratio_residual_scale=float(model.h1.config.normalized_ratio_residual_max*model.h1.normalized_ratio_residual_logit.tanh()),
        training_updates=0,notice='FP32 network inference. LF simulation uses the existing fixed operator with FP16 kernel storage. Frozen interventions are not evidence for replacing trained modules without retraining. Synthetic controls contain no scattering. Measured scaling preserves existing SNR.',
        catalog=catalog,rows=[])
    np.savez_compressed(output/'bp_training_statistics.npz',mean=calibration['mean'].cpu().numpy(),variance=calibration['variance'].cpu().numpy())
    mips={}
    for item in catalog:
        with h5py.File(item['path']) as f:
            original=torch.from_numpy(f['lf'][:])[None].cuda().float()
            reference=torch.from_numpy(f['ps_reference' if item['kind']=='measured' else 'target'][:])[None].cuda().float()
        for gain in gains:
            observed=original*gain
            for variant in VARIANTS:
                scale=float(observed.square().mean().sqrt())/calibration['reference_lf_rms'] if variant=='signal_scale' else 1.
                scale=max(scale,1e-6)
                norm_trace={}
                def norm_hook(module,args,result):
                    norm_trace.update(before=stats(args[0]),after=stats(result))
                with intervene(model,variant,calibration):
                    trace=model.h1.bp_output_norm.register_forward_hook(norm_hook)
                    try:stages=model.predict_stages(observed/scale,return_intermediates=True)
                    finally:trace.remove()
                predicted=stages['merged'].float()*scale
                h1=stages['h1_intermediates']
                if not torch.isfinite(predicted).all():raise FloatingPointError('Nonfinite output in '+item['name'])
                row=dict(case=item['name'],kind=item['kind'],gain=float(gain),variant=variant,
                    lf=lf_stats(observed),conditioning_scale=scale,h1_ratio=stats(h1['ratio']),
                    h1_normalized_ratio=stats(h1['normalized_ratio']),bp_normalization=norm_trace)
                if 'normalized_ratio_residual' in h1:
                    row['normalized_to_raw_bp_rms']=float(h1['normalized_ratio_residual'].square().mean().sqrt()/h1['raw_attended_ratio'].square().mean().sqrt().clamp_min(1e-12))
                if item['kind']=='measured':row['ps_shape_mse']=float(ps_forward_relative_mse(predicted,reference,ps))
                else:row.update(volume_scores(predicted,reference*gain))
                record['rows'].append(row)
                if gain==1.:
                    mips[item['name']+'__'+variant]=predicted[0].amax(0).cpu().numpy()
                del predicted,stages,h1
        progress('Signal tests: '+item['name'])
        write(output/'signal_audit.json',record)
        del original,reference,observed;gc.collect();torch.cuda.empty_cache()
    np.savez_compressed(output/'signal_mips.npz',**mips)
    del model,ps,calibration;gc.collect();torch.cuda.empty_cache()
    return record


def summary_table(record):
    import pandas as pd
    return pd.DataFrame([dict(case=r['case'],kind=r['kind'],brightness=r['gain'],method=VARIANTS[r['variant']],
        shape_error=r.get('shape_relative_l2'),PS_agreement_error=r.get('ps_shape_mse'),
        h1_ratio_mean=r['h1_ratio']['mean'],h1_ratio_spatial_std=r['h1_ratio']['median_spatial_std']) for r in record['rows']])


@torch.no_grad()
def validate_uniform_structure(root,checkpoint,output,progress=print):
    """Separate loss of shell shape from a missing constant fluorescence level."""
    root,output=Path(root),Path(output)
    catalog={r['name']:r for r in cases(root,output)}
    model,_=load_h1_h2_training_checkpoint(checkpoint,device='cpu');model=model.cuda().eval()
    rows=[]
    for name in ('syn_python_000002_10um_shells','shells_uniform_1','shells_uniform_4'):
        with h5py.File(catalog[name]['path']) as f:
            y=torch.from_numpy(f['lf'][:])[None].cuda().float()
            target=torch.from_numpy(f['target'][:])[None].cuda().float()
        predicted=model.predict_stages(y)['merged'].float()
        centered_predicted=predicted-predicted.mean()
        centered_target=target-target.mean()
        scores=volume_scores(centered_predicted,centered_target)
        rows.append(dict(case=name,shape_error_after_constant_and_gain_correction=scores['shape_relative_l2'],
                         fitted_gain=scores['fitted_output_gain'],target_mean=float(target.mean()),
                         prediction_mean=float(predicted.mean())))
        progress(str(rows[-1]))
        del y,target,predicted,centered_predicted,centered_target
    result=dict(checkpoint=model_record(checkpoint),precision='FP32',training_updates=0,rows=rows,
        definition='Subtract each volume mean, fit one output gain, and compute relative L2 error against the centered known target. This removes constant-offset and gain error; the three centered targets are the same shells.')
    write(output/'uniform_structure.json',result)
    del model;gc.collect();torch.cuda.empty_cache()
    return result


@torch.no_grad()
def validate_ratio_contribution(root,checkpoint,output,progress=print):
    """All validation frames, preserving the original equal-specimen weighting."""
    from collections import defaultdict
    root,output=Path(root),Path(output)
    model,_=load_h1_h2_training_checkpoint(checkpoint,device='cpu');model=model.cuda().eval()
    dataset=IndexedPSDataset(root/'data/paired_experimental/lf_ps_indexed_fullframe_em800_20260908_v1',split='validation')
    op=PSForwardOperator.from_mat(root.parent/'forward model/PSFs/I_PSF_40XNA1_920nm2p_.mat',(91,517,517),device='cuda')
    rows=[]
    for i in range(len(dataset)):
        s=dataset[i];lf=s['lf'][None].cuda().float();target=s['gt'][None].cuda().float()
        for variant in ('baseline','ratio_norm_off'):
            with intervene(model,variant,{}):prediction=model.predict_stages(lf)['merged'].float()
            rows.append(dict(case=s['name'],acquisition=s['acquisition'],specimen=s['specimen'],variant=variant,
                ps_shape_mse=float(ps_forward_relative_mse(prediction,target,op))))
            del prediction
        del lf,target,s
        if (i+1)%10==0:progress(f'Normalized-ratio contribution: {i+1}/{len(dataset)} validation frames')
    summary={}
    for variant in ('baseline','ratio_norm_off'):
        acq=defaultdict(list);specimen={}
        for r in rows:
            if r['variant']==variant:acq[r['acquisition']].append(r['ps_shape_mse']);specimen[r['acquisition']]=r['specimen']
        specs=defaultdict(list)
        for a,values in acq.items():specs[specimen[a]].append(float(np.mean(values)))
        means={s:float(np.mean(v)) for s,v in specs.items()}
        summary[variant]=dict(by_specimen=means,mean=float(np.mean(list(means.values()))))
    report=dict(checkpoint=model_record(checkpoint),frames=len(dataset),precision='FP32',training_updates=0,
        note='Frozen removal of the normalized-ratio residual only; BP normalization and all weights retained. Core-only PS mask, frame/acquisition/specimen equal weighting.',rows=rows,summary=summary)
    write(output/'ratio_validation.json',report);progress(str(summary));return report
