"""Finite-bead, held-out-depth diagnostic of the existing detection operator.

Only bead pose/depth mapping and scalar per-frame photometry are fitted. The LF
kernel and network are fixed. A uniformly fluorescent 2 um sphere is a stated
starting assumption, not a measured bead emission distribution.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
from .data import load_lf_tensor
from .physics_rearrange import space_to_view_depth, view_depth_to_space


@dataclass(frozen=True)
class BeadAuditConfig:
    diameter_um: float = 2.
    spacing_um: tuple = (.5, .21306818181818185, .21306818181818185)
    fit_frames: tuple = (16, 28, 40, 52, 64)
    held_out_frames: tuple = (20, 32, 44, 56, 60)
    reference_frame: int = 40
    stage_step_um: float = .5
    fit_iterations: int = 100
    learning_rate: float = .03
    voxel_supersampling: int = 8


class LocalBeadProjector:
    """Exact full convolution for sources confined to three central lenslets.

    Transposed convolution here is a computational rearrangement of *forward*
    projection, not an inverse or the LF adjoint. It avoids convolving zeros
    across a full field. Crops must start on the same lenslet grid as training.
    """
    def __init__(self, operator, local_lenslets=3):
        self.config=operator.config
        self.local_lenslets=local_lenslets
        self.local_size=local_lenslets*self.config.nnum
        self.output_size=(local_lenslets+self.config.kernel_size-1)*self.config.nnum
        self.offset=(self.config.kernel_size//2)*self.config.nnum
        self.gain=operator.gain
        self.weight=operator.projector.weight.transpose(0,1).flip(-2,-1).contiguous()
        self.device=self.weight.device

    def __call__(self, volume):
        if volume.shape[1:]!=(self.config.depth,self.local_size,self.local_size):
            raise ValueError('Source must fit the local volume')
        views=space_to_view_depth(volume,self.config.nnum).to(self.weight.dtype)
        projected=F.conv_transpose2d(views,self.weight)
        return view_depth_to_space(projected,channels=1,block_size=self.config.nnum).float()*self.gain


def sphere_template(shape,spacing_um,diameter_um,*,supersampling=8):
    """Voxel-averaged occupancy of a full 3D sphere (not a 2D disk)."""
    if min(shape)<1 or min(spacing_um)<=0 or diameter_um<=0 or supersampling<1:
        raise ValueError('Invalid bead sampling')
    axes=[(np.arange(n,dtype=np.float64)-(n-1)/2)*s for n,s in zip(shape,spacing_um)]
    z,y,x=np.meshgrid(*axes,indexing='ij',sparse=True)
    result=np.zeros(shape,dtype=np.float32)
    offsets=(np.arange(supersampling)+.5)/supersampling-.5
    for oz in offsets*spacing_um[0]:
        for oy in offsets*spacing_um[1]:
            for ox in offsets*spacing_um[2]:
                result+=((z+oz)**2+(y+oy)**2+(x+ox)**2<=(diameter_um/2)**2)
    return torch.from_numpy(result/supersampling**3)


class MovingSphere:
    def __init__(self, projector, config):
        self.projector=projector;self.config=config
        shape=(projector.config.depth,projector.local_size,projector.local_size)
        self.template=sphere_template(shape,config.spacing_um,config.diameter_um,
            supersampling=config.voxel_supersampling).to(projector.device)[None,None]
        axes=[torch.linspace(-1,1,n,device=projector.device) for n in shape]
        z,y,x=torch.meshgrid(*axes,indexing='ij')
        self.grid=torch.stack((x,y,z),-1)
        self.extent=torch.tensor([(n-1)*s/2 for n,s in zip(shape,config.spacing_um)],device=projector.device)

    def __call__(self, positions_zyx_um):
        shift=(positions_zyx_um/self.extent).flip(-1)
        grid=self.grid[None]-shift[:,None,None,None,:]
        return F.grid_sample(self.template.expand(len(shift),-1,-1,-1,-1),grid,
            mode='bilinear',padding_mode='zeros',align_corners=True)[:,0]


def fit_positive_photometry(pattern, measured):
    """Profile nonnegative gain/background independently in each frame.

    These two nuisance values may also be profiled on held-out frames: held-out
    scoring tests spatial shape, not absolute brightness or excitation profile.
    """
    dims=(-3,-2,-1)
    pm=pattern.mean(dims,keepdim=True);ym=measured.mean(dims,keepdim=True)
    pc=pattern-pm;yc=measured-ym
    gain=((pc*yc).sum(dims,keepdim=True)/pc.square().sum(dims,keepdim=True).clamp_min(1e-20)).clamp_min(0)
    background=ym-gain*pm
    # If the unconstrained intercept is negative, the optimum lies at b=0.
    boundary_gain=((pattern*measured).sum(dims,keepdim=True)/pattern.square().sum(dims,keepdim=True).clamp_min(1e-20)).clamp_min(0)
    gain=torch.where(background<0,boundary_gain,gain)
    background=background.clamp_min(0)
    fitted=gain*pattern+background
    relative_mse=(fitted-measured).square().sum(dims)/yc.square().sum(dims).clamp_min(1e-20)
    correlation=(pc*yc).sum(dims)/(pc.square().sum(dims)*yc.square().sum(dims)).sqrt().clamp_min(1e-20)
    return relative_mse,correlation,gain.flatten(),background.flatten(),fitted


def load_bead_case(folder,projector,config):
    folder=Path(folder)
    if set(config.fit_frames)&set(config.held_out_frames):raise ValueError('Depth splits overlap')
    if config.reference_frame not in config.fit_frames:raise ValueError('ROI reference must be a fit frame')
    reference=load_lf_tensor(folder/f'{config.reference_frame:06d}.mat')[0]
    n=projector.config.nnum;s=projector.output_size
    # Locate the highest-energy lenslet using only a fit frame. Keep phase origin.
    lenslet=reference.reshape(reference.shape[0]//n,n,reference.shape[1]//n,n).sum((1,3))
    peak=np.unravel_index(int(lenslet.argmax()),lenslet.shape)
    starts=[int((p-s//n//2)*n) for p in peak]
    if any(start<0 or start+s>bound for start,bound in zip(starts,reference.shape)):
        raise ValueError('Bead too near edge for the fixed complete-response ROI')
    arrays=[];records=[];frames=config.fit_frames+config.held_out_frames
    for index in frames:
        path=folder/f'{index:06d}.mat'
        a=load_lf_tensor(path)[0]
        crop=a[starts[0]:starts[0]+s,starts[1]:starts[1]+s]
        if not torch.isfinite(crop).all() or crop.min()<0:raise ValueError('Invalid raw counts')
        arrays.append(crop)
        records.append(dict(frame=index,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            raw_min=float(crop.min()),raw_median=float(crop.median()),raw_max=float(crop.max())))
    return torch.stack(arrays)[:,None].to(projector.device),dict(folder=str(folder),
        roi_start_yx=starts,roi_shape=[s,s],source_center_full_yx=[a+(s-1)/2 for a in starts],
        loading='load_lf_tensor: transpose MATLAB v7.3 once, same as network input',files=records)


def physical_parameters(q):
    # Lateral support remains inside the 3-lenslet emitter patch.
    return dict(dy_um=2*torch.tanh(q[0]),dx_um=2*torch.tanh(q[1]),
        z0_um=6*torch.tanh(q[2]),z_scale=1+.2*torch.tanh(q[3]))


def positions(q,frames,sign,config):
    p=physical_parameters(q)
    t=torch.as_tensor(frames,device=q.device,dtype=q.dtype)
    z=sign*(t-config.reference_frame)*config.stage_step_um*p['z_scale']+p['z0_um']
    if float(z.detach().abs().max())+config.diameter_um/2>22.5:
        raise ValueError('Finite sphere exceeds model depth')
    return torch.stack((z,p['dy_um'].expand_as(z),p['dx_um'].expand_as(z)),-1)


def audit_bead_case(folder,projector,config=BeadAuditConfig(),progress=print):
    measured,record=load_bead_case(folder,projector,config)
    sphere=MovingSphere(projector,config);count=len(config.fit_frames)
    target=measured[:count];frames=config.fit_frames
    started=time.perf_counter();candidates=[]
    # Sign is chosen on fit depths only; held-out images never select a pose.
    for sign in (1,-1):
        initial=[]
        with torch.no_grad():
            for z0 in (-4.,0.,4.):
                q=torch.zeros(4,device=projector.device);q[2]=np.arctanh(z0/6)
                pattern=projector(sphere(positions(q,frames,sign,config)))
                initial.append((float(fit_positive_photometry(pattern,target)[0].mean()),q))
        initial_loss,q=min(initial,key=lambda a:a[0]);q=q.detach().requires_grad_(True)
        optimizer=torch.optim.Adam([q],lr=config.learning_rate)
        best_loss=initial_loss;best_q=q.detach().clone();history=[]
        for iteration in range(config.fit_iterations):
            optimizer.zero_grad(set_to_none=True)
            pattern=projector(sphere(positions(q,frames,sign,config)))
            loss=fit_positive_photometry(pattern,target)[0].mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite bead loss')
            value=float(loss.detach());history.append(value)
            if value<best_loss:best_loss=value;best_q=q.detach().clone()
            loss.backward()
            if q.grad is None or not torch.isfinite(q.grad).all():raise FloatingPointError('Nonfinite bead gradient')
            optimizer.step()
            if progress and (iteration+1)%25==0:
                progress(f'{Path(folder).parent.parent.name}/{Path(folder).name} sign={sign:+d}: {iteration+1}/{config.fit_iterations}, fit residual={value:.4f}')
        with torch.no_grad():
            final_loss=float(fit_positive_photometry(projector(sphere(positions(q,frames,sign,config))),target)[0].mean())
            if final_loss<best_loss:best_loss=final_loss;best_q=q.detach().clone()
        candidates.append(dict(sign=sign,fit_relative_mse=best_loss,q=best_q,history=history))
    best=min(candidates,key=lambda a:a['fit_relative_mse']);sign=best['sign'];q=best['q']
    all_frames=config.fit_frames+config.held_out_frames
    with torch.no_grad():
        prediction=projector(sphere(positions(q,all_frames,sign,config)))
        mse,corr,gains,backgrounds,fitted=fit_positive_photometry(prediction,measured)
        baseline=projector(sphere(positions(torch.zeros_like(q),all_frames,1,config)))
        bmse,bcorr,_,_,bfitted=fit_positive_photometry(baseline,measured)
    rows=[]
    for i,index in enumerate(all_frames):
        rows.append(dict(frame=index,split='fit' if i<count else 'held_out',
            fitted_z_um=float(positions(q,(index,),sign,config)[0,0]),
            baseline_relative_mse=float(bmse[i]),relative_mse=float(mse[i]),
            baseline_correlation=float(bcorr[i]),correlation=float(corr[i]),
            nuisance_gain=float(gains[i]),nuisance_background_adu=float(backgrounds[i])))
    report=dict(data=record,config=asdict(config),parameters={k:float(v.detach()) for k,v in physical_parameters(q).items()},
        depth_sign=sign,fit_relative_mse=float(mse[:count].mean()),held_out_relative_mse=float(mse[count:].mean()),
        baseline_fit_relative_mse=float(bmse[:count].mean()),baseline_held_out_relative_mse=float(bmse[count:].mean()),
        sign_candidates=[{k:v for k,v in c.items() if k!='q'} for c in candidates],frames=rows,
        elapsed_seconds=time.perf_counter()-started,network_updates=0,kernel_updates=0,
        assumptions=['Uniformly fluorescent finite 2 um sphere; trilinear shifts of voxel-integrated occupancy',
            '0.5 um stage increment and reference index 40 are inherited calibration conventions',
            'Independent positive gain/background per frame absorb intensity; this cannot identify absolute gain or axial illumination',
            'No extra image rotation, transpose or resampling; native exported MATLAB orientation matches network loader',
            'No fitted aberrations, MLA distance, lateral drift, spatial background or correlated noise likelihood'])
    images=dict(measured=measured[:,0].cpu().numpy(),baseline=bfitted[:,0].cpu().numpy(),
        fitted=fitted[:,0].cpu().numpy(),frames=np.asarray(all_frames))
    return report,images


def plot_bead_audit(report,images):
    import matplotlib.pyplot as plt
    rows=report['frames'];selected=[i for i,r in enumerate(rows) if r['split']=='held_out'][::2]
    fig,axes=plt.subplots(len(selected),4,figsize=(12,3*len(selected)),layout='constrained',squeeze=False)
    for axes_row,i in zip(axes,selected):
        measured=images['measured'][i];fitted=images['fitted'][i];baseline=images['baseline'][i]
        lo=float(np.quantile(measured,.1));hi=float(np.quantile(measured,.999))
        residual=measured-fitted;bound=max(abs(residual).max(),1e-8)
        for ax,a,label in zip(axes_row[:3],[measured,baseline,fitted],['Measured LF','Nominal pose + photometry','Fitted pose + photometry']):
            ax.imshow(a,cmap='magma',origin='lower',vmin=lo,vmax=hi);ax.set_title(label);ax.axis('off')
        axes_row[3].imshow(residual,cmap='coolwarm',origin='lower',vmin=-bound,vmax=bound)
        axes_row[3].set_title(f"Residual | frame {rows[i]['frame']}\nr={rows[i]['correlation']:.3f}");axes_row[3].axis('off')
    fig.suptitle('Held-out depths · finite 2 µm bead · fixed training LF kernel\n'+report['data']['folder'])
    return fig
