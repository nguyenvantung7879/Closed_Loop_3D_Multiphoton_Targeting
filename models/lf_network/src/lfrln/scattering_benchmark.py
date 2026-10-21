"""Matched scattering, widespread fluorescence and EMCCD-noise method comparison.

This is a development diagnostic on known geometric validation objects, not a
calibrated scattering model or a held-out biological performance claim.
"""
from dataclasses import replace
from pathlib import Path
import gc
import json
import time

import h5py
import numpy as np
import torch

from .artifacts import sha256_file, write_json
from .scattering_controls import ScatteringConfig, scatter_lf
from .noise_profiles import load_noise_profile
from .synthetic import simulate_camera, preprocess_lf_counts
from .reconstruction_audit import FixedLFOperator, reconstruct_rl, reconstruct_phase_space, fingerprint
from . import load_h1_h2_training_checkpoint


CASE_FILES = {
    'shells': 'syn_python_000002_10um_shells.h5',
    'beads': 'syn_python_000001_2um_spheres.h5',
}
CONDITIONS = {
    'clear': (0., None),
    'diffuse50': (0., ScatteringConfig(diffuse_fraction=.5, angular_sigma_fraction=.6)),
    'diffuse75_broad': (0., ScatteringConfig(diffuse_fraction=.75, angular_sigma_fraction=10)),
    'uniform4': (4., None),
    'uniform4_diffuse50': (4., ScatteringConfig(diffuse_fraction=.5, angular_sigma_fraction=.6)),
}
LABELS = {'clear':'Clear', 'diffuse50':'50% redistributed',
          'diffuse75_broad':'75% redistributed, near-uniform angles',
          'uniform4':'Added uniform emitters: 4 × object mean',
          'uniform4_diffuse50':'Uniform emitters + 50% redistributed'}
METHODS = ('network', 'isra', 'poisson_rl', 'phase_space_rl')
METHOD_LABELS = {'network':'Current network (epoch 10)', 'isra':'3D ISRA (50 updates)',
                 'poisson_rl':'3D Poisson RL (50 updates)', 'phase_space_rl':'Phase-space RL (2 sweeps)'}


def expected_condition(clear_lf, uniform_lf, multiplier, config):
    """uniform_lf is A(ones * mean(original object)); scattering acts on total light."""
    expected = np.asarray(clear_lf, dtype=np.float64) + multiplier * np.asarray(uniform_lf, dtype=np.float64)
    if config is None:
        return expected, {'retained_lf_sum_ratio': 1., 'detected_redistributed_fraction': 0.}
    mixed, parts = scatter_lf(expected, config, return_components=True)
    return mixed, {'retained_lf_sum_ratio': parts['output_sum']/parts['input_sum'],
        'detected_redistributed_fraction': parts['scattered_sum']/parts['output_sum']}


def acquisition_scale(clear_lf, clear_peak_adu):
    """Known clear-reference acquisition scale; NEVER computed from a perturbed/noisy LF."""
    peak = float(np.max(clear_lf))
    if peak <= 0 or clear_peak_adu <= 0:
        raise ValueError('Positive clear reference and count level required')
    return float(clear_peak_adu/peak)


def observe(expected, camera, preprocessing, *, scale, seed):
    """Draw once; all reconstructions consume this exact same nonnegative tensor."""
    camera = replace(camera, signal_adu_per_lf_unit=scale, gain_variation=(1.,1.))
    preprocessing = replace(preprocessing, count_scale=scale)
    mean = torch.from_numpy(np.asarray(expected, dtype=np.float32))[None]
    counts, metadata = simulate_camera(mean, camera, generator=torch.Generator().manual_seed(seed))
    signed = preprocess_lf_counts(counts, replace(preprocessing, clamp_nonnegative=False), camera_config=camera)
    observed = signed.clamp_min(0)
    return observed, dict(seed=seed, count_scale=scale, expected_peak_adu=float(mean.max())*scale,
        expected_mean_adu=float(mean.mean())*scale, noise_rms_before_clamp=float((signed-mean).square().mean().sqrt()),
        negative_fraction_before_clamp=float((signed<0).float().mean()),
        expected_signal_mean=float(mean.mean()), input_mean=float(observed.mean()), camera=metadata)


def structural_scores(volume, target, object_support):
    """Full-volume structure after one fitted gain/constant; also retain absolute error.

    Gain/offset fits are evaluation only. Support is fixed from the original known
    geometry, so added uniform emitters do not redefine everything as foreground.
    """
    v, t = volume.float(), target.float()
    vm, tm = v.mean(), t.mean()
    vc, tc = v-vm, t-tm
    energy = tc.square().sum().clamp_min(1e-20)
    gain = ((vc*tc).sum()/vc.square().sum().clamp_min(1e-20)).clamp_min(0)
    residual = gain*vc-tc
    norm = t.square().sum().clamp_min(1e-20)
    bg = gain*vc[~object_support]
    stats = dict(shape_error=float((residual.square().sum()/energy).sqrt()),
        relative_volume_error=float(((v-t).square().sum()/norm).sqrt()),
        fitted_gain=float(gain), fitted_offset=float(tm-gain*vm),
        prediction_mean=float(vm), target_mean=float(tm),
        outside_structure_error_fraction=float(residual[~object_support].square().sum()/energy),
        outside_structure_std=float(bg.std(unbiased=False)),
        foreground_structure_rmse=float(residual[object_support].square().mean().sqrt()))
    return stats, gain


def projections(volume, target, gain, support):
    v=volume.numpy();t=target.numpy()
    z=int(t.sum((1,2)).argmax());y=int(t.sum((0,2)).argmax())
    # Remove fitted constant only for the shape panels; raw projections are also saved.
    shape=float(gain)*(v-v.mean())
    return dict(raw_xy=v.max(0), raw_xz=v.max(1), shape_xy=shape.max(0), shape_xz=shape.max(1),
                slice_xy=shape[z],slice_xz=shape[:,y], z_slice=np.array(z),y_slice=np.array(y))


def input_record(x):
    import hashlib
    a=np.ascontiguousarray(x, dtype=np.float32)
    return dict(sha256=hashlib.sha256(a.tobytes()).hexdigest(),mean=float(a.mean()),max=float(a.max()),
                sum=float(a.sum()),mean_over_peak=float(a.mean()/max(float(a.max()),1e-20)))


@torch.inference_mode()
def run(root, output, *, cases=('shells','beads'), peaks=(500.,3000.), seeds=(20260918,),
        iterations=50, sweeps=2, methods=METHODS, progress=print):
    root,output=Path(root),Path(output)
    checkpoint=root/'training_runs/experimental_auxcontext_boundary_aux_context_probe_20260915_v1/best_model.pt'
    cache=checkpoint.parent/'deconvolution_validation'
    profile=root/'data/noise/em800_2s_20260908_provisional/em800_2s_20260908_provisional.json'
    kernel=root/'data/kernels/lf_forward_40x_f11_aberr_08242026.bin'
    phase_cache=root/'data/kernels/phase_space_40x_f11_aberr_08242026_z91.h5'
    camera,preprocess,noise_profile=load_noise_profile(profile)
    manifest=json.loads((root/'data/synthetic_python/h2_raw_clean_v2_aberr_517/manifest.json').read_text())
    contract=dict(version=1,checkpoint_sha256=sha256_file(checkpoint),cases=list(cases),peaks=list(peaks),seeds=list(seeds),
        iterations=iterations,phase_sweeps=sweeps,methods=list(methods),
        conditions={k:dict(uniform_multiplier=m,redistribution=c.record() if c else None) for k,(m,c) in CONDITIONS.items()},
        source_sha256={str(p):sha256_file(p) for p in (Path(__file__),root/'src/lfrln/scattering_controls.py',
            root/'src/lfrln/reconstruction_audit.py',root/'src/lfrln/emccd.py',root/'src/lfrln/synthetic.py',profile)},
        input_sha256={str(cache/CASE_FILES[c]):sha256_file(cache/CASE_FILES[c]) for c in cases},
        optical_gain=float(manifest['optical_gain']),kernel_sha256=sha256_file(kernel),phase_cache_sha256=sha256_file(phase_cache),
        precision='Network FP32; existing fixed 3D projector FP16 storage and accumulation, float32 multiplicative updates; phase FFT FP32',
        noise_scale='For each object/count level, derive scale from CLEAR LF once, then use that same scale/divisor for every condition. Expected LF units preserved. Never normalize by perturbed or noisy peaks.',
        noise_status=noise_profile['provenance']['gain_status'],
        reconstruction_model='Every primary method uses its existing CLEAR forward model/weights; scattering is an unmodeled perturbation. No retraining or correction fitted to GT.',
        uniform_target='Full emitter target includes added uniform fluorescence. Structure metric subtracts volume means and fits one nonnegative output gain. Same centered geometry in all conditions.',
        training_updates=0,split='validation development controls; not independent biological test',
        timing='Descriptive whole-call timing including CPU return; not a counterbalanced performance benchmark')
    token=fingerprint(contract)
    output.mkdir(parents=True,exist_ok=True)
    contract_path=output/'contract.json'
    if contract_path.exists():
        if json.loads(contract_path.read_text())['fingerprint']!=token:raise ValueError('Changed contract; choose a new output directory')
    else:write_json(contract_path,dict(fingerprint=token,**contract))
    (output/'cases').mkdir(exist_ok=True)
    operator=FixedLFOperator(kernel,manifest['optical_gain'],dtype=torch.float16)
    model,payload=load_h1_h2_training_checkpoint(checkpoint,device='cpu');model=model.cuda().eval()
    report_path=output/'results.json'
    report=json.loads(report_path.read_text()) if report_path.exists() else dict(fingerprint=token,epoch=int(payload['epoch']),rows=[],completed=False)
    if report['fingerprint']!=token:raise ValueError('Result fingerprint differs')
    done={(r['case'],r['condition'],r['noise'],r['seed'],r['method']) for r in report['rows']}
    torch.set_num_threads(4)
    started=time.perf_counter()
    for case in cases:
        with h5py.File(cache/CASE_FILES[case]) as f:
            target=torch.from_numpy(f['target'][:]).float();clear=f['lf'][:].squeeze().astype(np.float64)
        support=target>max(float(target.max())*.05,1e-10)
        # Exact zero-padded A(uniform volume); a uniform object need not make a flat detector image.
        uniform=operator(torch.full_like(target,float(target.mean()))[None].cuda())[0,0].cpu().numpy().astype(float)
        np.savez_compressed(output/f'{case}_reference.npz',target_xy=(target-target.mean()).amax(0).numpy(),
            target_xz=(target-target.mean()).amax(1).numpy(),clear_lf=clear,uniform_increment_lf=uniform)
        for condition,(multiplier,config) in CONDITIONS.items():
            expected,transport=expected_condition(clear,uniform,multiplier,config)
            truth=target+multiplier*target.mean()
            acquisitions=[('noiseless',0,None)]+[(f'peak{level:g}',seed,level) for level in peaks for seed in seeds]
            for noise,seed,level in acquisitions:
                key=f'{case}__{condition}__{noise}__seed{seed}'
                input_path=output/'cases'/f'{key}_input.npz'
                if input_path.exists():
                    data=np.load(input_path);observed=torch.from_numpy(data['lf'].copy())[None]
                    acquisition=json.loads(str(data['acquisition']))
                else:
                    if level is None:observed=torch.from_numpy(expected.astype('float32'))[None];acquisition=dict(noise='none')
                    else:observed,acquisition=observe(expected,camera,preprocess,scale=acquisition_scale(clear,level),seed=seed)
                    np.savez_compressed(input_path,lf=observed[0].numpy(),expected=expected.astype('float32'),acquisition=json.dumps(acquisition))
                identity=input_record(observed.numpy())
                progress(f'INPUT {key} | mean/peak={identity["mean_over_peak"]:.4g}',flush=True)
                for method in methods:
                    item=(case,condition,noise,seed,method)
                    if item in done:continue
                    torch.cuda.synchronize();begin=time.perf_counter()
                    if method=='network':
                        stages=model.predict_stages(observed[None].cuda(),return_intermediates=True)
                        volume=stages['merged'][0].float().cpu()
                        ratio=stages['h1_intermediates']['ratio'].float()
                        detail=dict(h1_ratio_mean=float(ratio.mean()),h1_ratio_std=float(ratio.std()))
                        del stages,ratio
                    elif method in ('isra','poisson_rl'):
                        volume,detail=reconstruct_rl(observed,operator,iterations=iterations,
                            update='volume' if method=='isra' else 'measurement')
                    else:
                        volume,detail=reconstruct_phase_space(observed,phase_cache,gain=operator.gain,sweeps=sweeps)
                    torch.cuda.synchronize();elapsed=time.perf_counter()-begin
                    if not torch.isfinite(volume).all():raise FloatingPointError(key+' '+method)
                    metrics,gain=structural_scores(volume,truth,support)
                    arrays=projections(volume,truth,gain,support)
                    np.savez_compressed(output/'cases'/f'{key}_{method}.npz',**arrays)
                    row=dict(case=case,condition=condition,noise=noise,seed=seed,method=method,
                        input=identity,acquisition=acquisition,transport=transport,**metrics,
                        elapsed_seconds=elapsed,details=detail)
                    report['rows'].append(row);done.add(item)
                    report['elapsed_this_invocation_seconds']=time.perf_counter()-started
                    write_json(report_path,report)
                    progress(f'  {method}: shape={metrics["shape_error"]:.4f}; {elapsed:.1f} s',flush=True)
                    del volume,arrays;gc.collect();torch.cuda.empty_cache()
            del truth
        del target,support
    assert sha256_file(checkpoint)==contract['checkpoint_sha256']
    report.update(completed=True,checkpoint_unchanged=True,training_updates=0)
    write_json(report_path,report)
    return report
