"""Fixed-stage, multi-start bead fits and shared sampling diagnostics.

Whole-bead validation excludes those beads from shared-parameter fitting.
Their own nuisance poses use fit depths only; test depths remain unused.
"""
from dataclasses import dataclass
import numpy as np
import torch
from .bead_calibration import MovingSphere,fit_positive_photometry


@dataclass(frozen=True)
class PoseFitConfig:
    iterations:int=60
    learning_rate:float=.04
    z_initializations:tuple=(-4.,0.,4.)
    signs:tuple=(1,-1)


def fixed_stage_positions(q,frames,sign,config):
    if q.shape!=(3,):raise ValueError('Only dy,dx,z0 may vary; stage scale is fixed')
    dy,dx,z0=2*q[0].tanh(),2*q[1].tanh(),6*q[2].tanh()
    frames=torch.as_tensor(frames,device=q.device,dtype=q.dtype)
    z=sign*(frames-config.reference_frame)*config.stage_step_um+z0
    if float(z.detach().abs().max())+config.diameter_um/2>22.5:
        raise ValueError('Finite bead exceeds current LF depth support')
    return torch.stack((z,dy.expand_as(z),dx.expand_as(z)),-1)


def render(case,q,sign,projector,config,sampling=None,frames=None):
    frames=frames or config.fit_frames
    pattern=projector(case['sphere'](fixed_stage_positions(q,frames,sign,config)))
    if sampling is not None:
        pattern=sampling(pattern,roi_start_yx=case['record']['roi_start_yx'],full_shape_yx=(517,517))
    return pattern


def fit_pose(case,projector,config,fit_config=PoseFitConfig(),sampling=None,progress=None):
    count=len(config.fit_frames);measured=case['measured'][:count];candidates=[]
    if sampling is not None:sampling.requires_grad_(False)
    for sign in fit_config.signs:
        for initial_z in fit_config.z_initializations:
            q=torch.zeros(3,device=projector.device);q[2]=float(np.arctanh(initial_z/6));q.requires_grad_(True)
            optimizer=torch.optim.Adam([q],lr=fit_config.learning_rate)
            best=float('inf');best_q=None;history=[]
            for iteration in range(fit_config.iterations+1):
                optimizer.zero_grad(set_to_none=True)
                pattern=render(case,q,sign,projector,config,sampling)
                loss=fit_positive_photometry(pattern,measured)[0].mean()
                if not torch.isfinite(loss):raise FloatingPointError('Invalid pose loss')
                value=float(loss.detach());history.append(value)
                if value<best:best=value;best_q=q.detach().clone()
                if iteration==fit_config.iterations:break
                loss.backward()
                if not torch.isfinite(q.grad).all():raise FloatingPointError('Invalid pose gradient')
                optimizer.step()
            candidates.append(dict(sign=sign,initial_z_um=initial_z,loss=best,q=best_q,history=history))
            if progress:progress(f"{case['key']} sign {sign:+d}, initial z {initial_z:+g}: {best:.5f}")
    selected=min(candidates,key=lambda c:c['loss'])
    return selected,candidates


def summarize_pose(case,selected,candidates,projector,config,sampling=None):
    frames=config.fit_frames+config.held_out_frames;count=len(config.fit_frames)
    q=selected['q'];sign=selected['sign']
    with torch.no_grad():
        pattern=render(case,q,sign,projector,config,sampling,frames)
        mse,corr,g,b,fitted=fit_positive_photometry(pattern,case['measured'])
        positions=fixed_stage_positions(q,frames,sign,config).cpu().numpy()
    # Multi-start spread is a stability diagnostic, not a statistical CI.
    near=[c for c in candidates if c['loss']<=selected['loss']+max(.005,.02*selected['loss'])]
    centers=np.asarray([fixed_stage_positions(c['q'],(config.reference_frame,),c['sign'],config)[0].cpu().numpy() for c in near])
    spread=np.ptp(centers,axis=0)
    rows=[dict(frame=frame,split='pose_fit' if i<count else 'unseen_depth',
        relative_mse=float(mse[i]),correlation=float(corr[i]),
        center_zyx_um=positions[i].tolist(),profiled_gain=float(g[i]),background_adu=float(b[i])) for i,frame in enumerate(frames)]
    result=dict(case=case['key'],data=case['record'],depth_sign=sign,z_scale=1.,stage_step_um=config.stage_step_um,
        fit_relative_mse=float(mse[:count].mean()),held_out_depth_relative_mse=float(mse[count:].mean()),frames=rows,
        near_optimal_multistart_spread_zyx_um=spread.tolist(),near_optimal_signs=sorted(set(c['sign'] for c in near)),
        candidates=[dict(sign=c['sign'],initial_z_um=c['initial_z_um'],fit_loss=c['loss'],history=c['history'],
            center_zyx_um=fixed_stage_positions(c['q'],(config.reference_frame,),c['sign'],config)[0].cpu().tolist()) for c in candidates],
        photometry='Separate nonnegative scalar gain/background on every frame, including unseen depths',
        confidence='Multi-start spread is optimizer stability, not independent localization uncertainty',
        label_status='conditional pseudo-label; no independent volume GT')
    return result,dict(measured=case['measured'][:,0].cpu().numpy(),fitted=fitted[:,0].cpu().numpy(),frames=np.asarray(frames))


def fit_shared_sampling(cases,poses,projector,config,sampling,*,iterations=80,prior_weight=.002,progress=None):
    """Alternate per-bead pose and shared sampling updates on calibration beads.

    Only fit depths enter the objective. No held-out bead or depth is passed.
    Physical kernels are fixed. A small bounded-coordinate prior favors the
    nominal sampling and is printed in the audit, not selected on held-out data.
    """
    qs=[p['q'].detach().clone().requires_grad_(True) for p in poses]
    pose_optimizers=[torch.optim.Adam([q],lr=.015) for q in qs]
    sampling.requires_grad_(True);optimizer=torch.optim.Adam(sampling.parameters(),lr=.025)
    best=float('inf');best_state=None;best_poses=None;history=[]
    for step in range(iterations+1):
        sampling.requires_grad_(False)
        if step<iterations:
            for case,p,q,opt in zip(cases,poses,qs,pose_optimizers):
                opt.zero_grad(set_to_none=True)
                loss=fit_positive_photometry(render(case,q,p['sign'],projector,config,sampling),
                                            case['measured'][:len(config.fit_frames)])[0].mean()
                loss.backward();opt.step()
        sampling.requires_grad_(True);optimizer.zero_grad(set_to_none=True);value=0.
        for case,p,q in zip(cases,poses,qs):
            loss=fit_positive_photometry(render(case,q.detach(),p['sign'],projector,config,sampling),
                                        case['measured'][:len(config.fit_frames)])[0].mean()/len(cases)
            value+=float(loss.detach());loss.backward()
        prior=prior_weight*sampling.q.tanh().square().mean();prior.backward()
        total=value+float(prior.detach());history.append(dict(step=step,fit_loss=value,prior=float(prior.detach())))
        if not np.isfinite(total) or not torch.isfinite(sampling.q.grad).all():raise FloatingPointError('Invalid shared fit')
        if total<best:
            best=total;best_state=sampling.q.detach().clone()
            best_poses=[q.detach().clone() for q in qs]
        if progress and step%20==0:progress(f'Shared sampling {step}/{iterations}: {value:.5f}, {sampling.contract()}')
        if step<iterations:optimizer.step()
    with torch.no_grad():sampling.q.copy_(best_state)
    sampling.requires_grad_(False)
    return history,best_poses
