"""Repeated, synchronized end-to-end reconstruction timing; no optimizer."""
from datetime import datetime,timezone
from pathlib import Path
import gc
import json
import time
import numpy as np
import torch

from . import load_h1_h2_training_checkpoint
from .reconstruction_audit import FixedLFOperator,build_phase_psf_cache,reconstruct_rl,reconstruct_phase_space,file_record
from .reconstruction_benchmark import load_case,model_record,write_json


def time_reconstruction(call, *, device, warmups=1, repeats=3):
    """Time the full call, including return of a dense float32 CPU volume."""
    device=torch.device(device)
    if warmups<1 or repeats<2:raise ValueError('Use at least one warmup and two measured repeats')
    sync=lambda:torch.cuda.synchronize(device) if device.type=='cuda' else None
    for _ in range(warmups):
        result=call();sync();del result
    records=[]
    for _ in range(repeats):
        sync()
        baseline=torch.cuda.memory_allocated(device) if device.type=='cuda' else None
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        started=time.perf_counter()
        volume=call()
        sync();seconds=time.perf_counter()-started
        if volume.device.type!='cpu' or volume.dtype!=torch.float32:
            raise ValueError('Every timed method must return the float32 CPU volume')
        peak=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None
        record=dict(seconds=seconds,baseline_allocated_gib=baseline/2**30 if baseline is not None else None,
                    peak_allocated_gib=peak/2**30 if peak is not None else None,
                    incremental_peak_allocated_gib=(peak-baseline)/2**30 if peak is not None else None)
        if not torch.isfinite(volume).all():raise FloatingPointError('Nonfinite timed reconstruction')
        records.append(record);del volume
    elapsed=np.array([r['seconds'] for r in records])
    return dict(warmups=warmups,repeats=repeats,trials=records,
                median_seconds=float(np.median(elapsed)),min_seconds=float(elapsed.min()),
                max_seconds=float(elapsed.max()),mean_seconds=float(elapsed.mean()),
                peak_allocated_gib=max(r['peak_allocated_gib'] for r in records) if device.type=='cuda' else None,
                incremental_peak_allocated_gib=max(r['incremental_peak_allocated_gib'] for r in records) if device.type=='cuda' else None)


@torch.no_grad()
def benchmark_comparison(root, comparison_directory, *, case_key=None, warmups=1,repeats=3,
                         device='cuda', progress=print):
    root,folder=Path(root),Path(comparison_directory);device=torch.device(device)
    contract=json.loads((folder/'contract.json').read_text(encoding='utf-8'))
    catalog=contract['catalog'];settings=contract['settings']
    for record in contract['sources']:
        if file_record(record['path'])!=record:
            raise ValueError('Optics or reconstruction implementation changed since the comparison')
    row=next((r for r in catalog if r['key']==case_key),None) if case_key else next(r for r in catalog if r['kind']=='measured_biology')
    if row is None:raise ValueError('Timing case is absent from the comparison catalog')
    for key in ('source','baseline'):
        if key in row and file_record(row[key]['path'])!=row[key]:
            raise ValueError('Timing input changed since the reconstruction comparison')
    checkpoints={k:Path(v['path']) for k,v in contract['checkpoints'].items()}
    for k,p in checkpoints.items():
        if model_record(p)['sha256']!=contract['checkpoints'][k]['sha256']:
            raise ValueError('Checkpoint changed since the reconstruction comparison')
    manifest=json.loads((root/'data/synthetic_python/h2_raw_clean_v2_aberr_517/manifest.json').read_text())
    if device.type=='cuda':torch.cuda.synchronize(device)
    started=time.perf_counter()
    operator=FixedLFOperator(root/'data/kernels/lf_forward_40x_f11_aberr_08242026.bin',
        float(manifest['optical_gain']),device=device,dtype=torch.float16)
    if device.type=='cuda':torch.cuda.synchronize(device)
    physics_setup=time.perf_counter()-started
    started=time.perf_counter()
    phase_cache=root/'data/kernels/phase_space_40x_f11_aberr_08242026_z91.h5'
    build_phase_psf_cache(root/'data/psf/lf_psf_40x_f11_aberr_08242026.h5',phase_cache)
    phase_setup=time.perf_counter()-started
    case=load_case(root,row,operator);y=case.lf[None].to(device).float()
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),case=row,
        comparison_contract=file_record(folder/'contract.json'),comparison_fingerprint=contract['fingerprint'],
        checkpoints={k:model_record(p) for k,p in checkpoints.items()},
        source=file_record(__file__),torch_version=torch.__version__,cuda_version=torch.version.cuda,
        gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
        cpu_threads=torch.get_num_threads(),input_shape=list(y.shape),output_shape=[91,*y.shape[-2:]],
        warmups=warmups,repeats=repeats,methods={},
        setup=dict(shared_3d_operator_seconds=physics_setup,phase_cache_validation_or_build_seconds=phase_setup),
        scope='Prepared LF on GPU -> float32 CPU volume; synchronized wall time. Excludes shared optics/checkpoint loading, plotting, external scoring and saved-volume I/O.',
        implementation_costs='Includes method-internal initialization and diagnostics. Phase-space includes angular interpolation and streamed phase-PSF reads, with a warmed file cache after warmup.',
        memory_scope='PyTorch allocated memory in this process, not total device use. Incremental peak is above the allocation immediately before the timed call.',
        environment_note='Sequential methods on the current shared workstation; timings are not an isolated-device throughput guarantee.',
        stopping=dict(rl_iterations=settings['rl_iterations'],phase_sweeps=settings['phase_sweeps']))
    if device.type=='cuda':
        free,total=torch.cuda.mem_get_info(device)
        report['device_memory_before_trials_gib']=dict(free=free/2**30,total=total/2**30)
    for name,path in checkpoints.items():
        started=time.perf_counter()
        model,payload=load_h1_h2_training_checkpoint(path,device='cpu')
        model=model.to(device).eval()
        if device.type=='cuda':torch.cuda.synchronize(device)
        setup=time.perf_counter()-started
        def network_call():
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.float16):
                stages=model.predict_stages(y)
            return stages['merged'][0].float().cpu()
        progress(f'Timing {name}: {warmups} warmup + {repeats} repeats')
        record=time_reconstruction(network_call,device=device,warmups=warmups,repeats=repeats)
        report['methods'][name]=dict(record,checkpoint_loading_seconds=setup,epoch=int(payload['epoch']),
                                    reconstruction='one network forward',precision='AMP float16; float32 CPU output')
        del model,payload
        gc.collect();torch.cuda.empty_cache()
    for update,name in (('volume','3d_rl_volume'),('measurement','3d_rl_measurement')):
        def rl_call():return reconstruct_rl(y,operator,iterations=settings['rl_iterations'],update=update)[0]
        progress(f'Timing {name}: {warmups} warmup + {repeats} repeats')
        report['methods'][name]=dict(time_reconstruction(rl_call,device=device,warmups=warmups,repeats=repeats),
            reconstruction=f'{settings["rl_iterations"]} iterations',precision='float16 convolution; float32 updates/output')
    def phase_call():return reconstruct_phase_space(y,phase_cache,gain=operator.gain,
        sweeps=settings['phase_sweeps'],device=device)[0]
    progress(f'Timing phase_space_rl: {warmups} warmup + {repeats} repeats')
    report['methods']['phase_space_rl']=dict(time_reconstruction(phase_call,device=device,warmups=warmups,repeats=repeats),
        reconstruction=f'{settings["phase_sweeps"]} complete angular sweeps',precision='float32 linear FFT')
    output=folder/'timing';output.mkdir(exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    report['report_path']=str(output/f'timing_{stamp}.json')
    write_json(report['report_path'],report);write_json(output/'latest_timing.json',report)
    del operator,y,case
    gc.collect();torch.cuda.empty_cache()
    return report


def plot_timing(report):
    import matplotlib.pyplot as plt
    names=list(report['methods']);rows=[report['methods'][k] for k in names]
    median=np.array([r['median_seconds'] for r in rows])
    error=np.array([median-np.array([r['min_seconds'] for r in rows]),np.array([r['max_seconds'] for r in rows])-median])
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    ax.barh([k.replace('_',' ') for k in names],median,xerr=error,alpha=.8,capsize=4)
    ax.set_xscale('log');ax.set_xlabel('Seconds per reconstructed volume (log scale); median and observed range')
    ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    ax.set_title(f'{report["gpu"]} | {report["warmups"]} warmup + {report["repeats"]} repeats\n'
                 f'3D RL: {report["stopping"]["rl_iterations"]} iterations; phase-space: {report["stopping"]["phase_sweeps"]} sweeps')
    fig.savefig(Path(report['report_path']).with_suffix('.png'),dpi=130)
    return fig
