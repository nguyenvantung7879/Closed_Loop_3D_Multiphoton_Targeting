"""Compare a new joint checkpoint with the verified notebook17 baselines.

Old caches are reused only for identical input pixels, source records, optics,
method settings and checkpoint hashes. They are never modified.
"""
from pathlib import Path
import gc
import json
import time
import h5py
import numpy as np
import torch

from . import load_h1_h2_training_checkpoint
from .reconstruction_audit import file_record, fingerprint
from .reconstruction_benchmark import run_case, score_volume, write_json, _save_array, model_record


def matching_baseline_cache(root, folder, case, settings):
    """Return a verified old report or None; never silently accept changed pixels."""
    root,folder=Path(root),Path(folder)
    contract_path=folder/'contract.json'
    path=folder/(case.key+'.h5');report_path=folder/(case.key+'.json')
    if not all(p.exists() for p in (contract_path,path,report_path)):return None
    contract=json.loads(contract_path.read_text(encoding='utf-8'))
    report=json.loads(report_path.read_text(encoding='utf-8'))
    if fingerprint(report['case'])!=fingerprint(case.provenance):return None
    for key in ('rl_iterations','phase_sweeps'):
        if contract['settings'][key]!=settings[key]:return None
    # The previous scientific implementation and optical files must remain intact.
    if any(file_record(source['path'])!=source for source in contract['sources']):return None
    if report['fingerprint']!=contract['fingerprint']:raise ValueError('Old case/contract mismatch')
    with h5py.File(path) as f:
        if f.attrs['fingerprint']!=contract['fingerprint']:raise ValueError('Old H5/contract mismatch')
        if not np.array_equal(f['lf'][:],case.lf.numpy()):return None
        role='target' if case.target is not None else 'ps_reference'
        reference=case.target if case.target is not None else case.ps_reference
        if not np.array_equal(f[role][:],reference.numpy()):return None
    return dict(path=path,report=report,contract=contract)


@torch.no_grad()
def compare_joint_case(root, case, operator, phase_cache, checkpoints, output, settings, token,
                       *, baseline_cache, progress=print):
    output=Path(output);path=output/(case.key+'.h5');report_path=path.with_suffix('.json')
    if path.exists() and report_path.exists():
        report=json.loads(report_path.read_text(encoding='utf-8'))
        with h5py.File(path) as f:
            if f.attrs['fingerprint']!=token or report['fingerprint']!=token:
                raise ValueError('Joint comparison cache contract changed')
        progress('Reusing completed '+case.key)
        return report
    previous=matching_baseline_cache(root,baseline_cache,case,settings)
    if previous is None:
        progress('Computing all methods for '+case.key)
        return run_case(case,operator,phase_cache,checkpoints,output,settings,token,progress=progress)
    previous_path=previous['path'];old=previous['report'];old_contract=previous['contract']
    temporary=path.with_suffix('.building.h5')
    report=dict(case=case.provenance,fingerprint=token,methods={},lf_statistics=old['lf_statistics'],
                target_role=old['target_role'],baseline_reuse_source=file_record(previous_path))
    if 'target_morphology' in old:report['target_morphology']=old['target_morphology']
    with h5py.File(previous_path) as inp,h5py.File(temporary,'w') as out:
        out.attrs['fingerprint']=token;out.attrs['case']=json.dumps(case.provenance)
        for name in ('lf','target','ps_reference'):
            if name in inp:inp.copy(name,out,name=name)
        out.create_group('predictions');out.create_group('reprojected_lf')
        def copy_method(old_name,new_name):
            inp.copy('predictions/'+old_name,out['predictions'],name=new_name)
            if old_name in inp['reprojected_lf']:
                inp.copy('reprojected_lf/'+old_name,out['reprojected_lf'],name=new_name)
            report['methods'][new_name]=dict(old['methods'][old_name],reused_from=str(previous_path),
                timing_note='Recorded reconstruction runtime from notebook17, not time to read this cache')
        for name in ('3d_rl_volume','3d_rl_measurement','phase_space_rl'):copy_method(name,name)
        for name,checkpoint in checkpoints.items():
            checkpoint_hash=model_record(checkpoint)['sha256']
            alias=next((k for k,v in old_contract['checkpoints'].items() if v['sha256']==checkpoint_hash),None)
            if alias is not None:
                copy_method(alias,name)
                if settings.get('save_stages',True):
                    for stage in ('h1','h2'):copy_method(alias+'_'+stage,name+'_'+stage)
                progress('Verified cached network '+name+' | '+case.key)
                continue
            model,payload=load_h1_h2_training_checkpoint(checkpoint,device='cpu')
            model=model.to(operator.device).eval()
            if operator.device.type=='cuda':torch.cuda.synchronize()
            started=time.perf_counter()
            with torch.autocast(device_type=operator.device.type,dtype=torch.float16,enabled=operator.device.type=='cuda'):
                stages=model.predict_stages(case.lf[None].to(operator.device))
            if operator.device.type=='cuda':torch.cuda.synchronize()
            seconds=time.perf_counter()-started
            for stage in (('merged','h1','h2') if settings.get('save_stages',True) else ('merged',)):
                key=name if stage=='merged' else name+'_'+stage
                volume=stages[stage][0].float().cpu()
                _save_array(out,'predictions/'+key,volume)
                metrics=score_volume(volume,case,operator,cutoff_domain=stage=='h1')
                report['methods'][key]=dict(metrics,output_domain='cutoff' if stage=='h1' else 'raw',
                    epoch=int(payload['epoch']),seconds=seconds if stage=='merged' else None)
                if stage!='h1':_save_array(out,'reprojected_lf/'+key,operator(volume[None].to(operator.device))[0].cpu())
            del model,payload,stages,volume
            gc.collect();torch.cuda.empty_cache()
            progress('Reconstructed '+name+' | '+case.key)
    # Stable comparison order places reference, networks, and then iterative baselines.
    order=[]
    for name in checkpoints:
        order.append(name)
        if settings.get('save_stages',True):order.extend((name+'_h1',name+'_h2'))
    order+=['3d_rl_volume','3d_rl_measurement','phase_space_rl']
    report['methods']={name:report['methods'][name] for name in order}
    temporary.replace(path);write_json(report_path,report)
    return report


def plot_fixed_slices(path):
    """Common reference-selected slices; never choose a plane from a prediction."""
    import matplotlib.pyplot as plt
    path=Path(path);report=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    methods=[k for k in report['methods'] if not k.endswith(('_h1','_h2'))]
    with h5py.File(path) as f:
        role='target' if 'target' in f else 'ps_reference';reference=f[role][:]
        nz,ny,nx=reference.shape
        profiles=(reference.sum((1,2)),reference.sum((0,2)),reference.sum((0,1)))
        center=tuple(int(np.clip(np.rint((np.arange(len(p))*p).sum()/max(float(p.sum()),1e-12)),0,len(p)-1)) for p in profiles)
        z,y,x=center;dz,dy,dx=.5,.21306818181818185,.21306818181818185
        names=[role]+methods
        fig,axes=plt.subplots(3,len(names),figsize=(3.2*len(names),9),layout='constrained',
                              gridspec_kw={'height_ratios':[2.4,1,1]})
        for c,name in enumerate(names):
            a=reference if c==0 else f['predictions/'+name][:]
            scale=max(float(np.quantile(a,.999)),1e-12)
            planes=[(a[z],[-nx*dx/2,nx*dx/2,-ny*dy/2,ny*dy/2],'y'),
                    (a[:,y,:],[-nx*dx/2,nx*dx/2,-nz*dz/2,nz*dz/2],'z'),
                    (a[:,:,x],[-ny*dy/2,ny*dy/2,-nz*dz/2,nz*dz/2],'z')]
            for r,(im,extent,label) in enumerate(planes):
                axes[r,c].imshow(im,cmap='magma',origin='lower',extent=extent,vmin=0,vmax=scale,aspect='equal')
                axes[r,c].set_xlabel(('y' if r==2 else 'x')+' (µm)');axes[r,c].set_ylabel(label+' (µm)')
            axes[0,c].set_title(('PS reference (filtered)' if role=='ps_reference' else 'Artificial emitter') if c==0 else name.replace('_',' '),fontsize=9)
        fig.suptitle(f'{report["case"]["kind"]}: fixed slices at reference intensity centroid (z,y,x)={center}\n'
                     'Same planes for every method; independent volume 99.9% display scales; nominal LF grid')
        fig.savefig(path.with_name(path.stem+'_slices.png'),dpi=110)
    return fig
