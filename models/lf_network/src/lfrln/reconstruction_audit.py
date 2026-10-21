"""Physical LF controls and a port of the local Lu et al. phase-space RL.

The phase-space port follows getPhaseSpaceLFPSF.m and deconv_phase_space.m.
Linear FFT convolutions replace conv2; numerical denominator floors are explicit.
This module does not train a network or equate PS images with raw emitters.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import time
import h5py
import numpy as np
from scipy.fft import next_fast_len
import torch
from torch.nn import functional as F
from .emccd import _resize_weights
from .forward_model import load_lfm_forward_binary, LFMForwardConfig
from .physics_rearrange import space_to_view_depth, view_depth_to_space


def file_record(path):
    p=Path(path).resolve(); s=p.stat()
    return {'path':str(p),'size':s.st_size,'mtime_ns':s.st_mtime_ns}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()


class FixedLFOperator:
    """Current generator with a matching convolution-transpose adjoint."""
    def __init__(self,path,gain,*,device='cuda',dtype=torch.float16,config=None):
        self.projector=load_lfm_forward_binary(path,config=config or LFMForwardConfig(),
            device=device,dtype=dtype).to(device).eval()
        self.gain=float(gain)
        if not np.isfinite(self.gain) or self.gain<=0:raise ValueError('Positive gain required')
        self.config=self.projector.config
        if self.config.input_domain!='raw' or self.config.apply_axial_profile:
            raise ValueError('Audit requires the detection-only raw-emitter operator')
        self.device=torch.device(device)

    def __call__(self,x):
        return self.projector(x).float()*self.gain

    def adjoint(self,y):
        c=self.config
        a=space_to_view_depth(y,c.nnum).to(self.projector.weight.dtype)
        a=F.conv_transpose2d(a,self.projector.weight,padding=c.kernel_size//2)
        return view_depth_to_space(a,channels=c.depth,block_size=c.nnum).float()*self.gain

    def dot_test(self,height=517,seed=19):
        gen=torch.Generator(device=self.device).manual_seed(seed)
        x=torch.rand((1,self.config.depth,height,height),generator=gen,device=self.device)-.5
        y=torch.rand((1,1,height,height),generator=gen,device=self.device)-.5
        with torch.no_grad():
            lhs=(self(x).double()*y.double()).sum()
            rhs=(x.double()*self.adjoint(y).double()).sum()
        return float((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs()).clamp_min(1e-12))


@torch.no_grad()
def reconstruct_rl(lf,operator,*,iterations=50,update='measurement',callback=None):
    """measurement: A^T(y/Ax)/A^T1; volume: (A^Ty)/(A^TAx).

    No repeated emitter/sensor sum equality is imposed. Fixed stopping and
    positivity are the only regularization here. Returns [Z,Y,X] on CPU.
    """
    if update not in ('measurement','volume') or iterations<1:raise ValueError('Invalid RL settings')
    y=lf.to(operator.device).float().reshape(1,1,*lf.shape[-2:])
    if torch.any(y<0) or not torch.isfinite(y).all():raise ValueError('Invalid LF')
    shape=(1,operator.config.depth,*y.shape[-2:])
    x=torch.ones(shape,device=operator.device)
    x*=y.sum()/operator(x).sum().clamp_min(1e-12)
    sensitivity=operator.adjoint(torch.ones_like(y)).clamp_min(1e-12)
    numerator=operator.adjoint(y) if update=='volume' else None
    eps=max(float(y.max())*1e-7,1e-10)
    history=[];t0=time.perf_counter();ax=operator(x)
    for i in range(1,iterations+1):
        if update=='measurement':factor=operator.adjoint(y/ax.clamp_min(eps))/sensitivity
        else:factor=numerator/operator.adjoint(ax).clamp_min(1e-12)
        x=(x*factor).clamp_min(0);ax=operator(x)
        if not torch.isfinite(x).all():raise FloatingPointError('RL became nonfinite')
        row={'iteration':i,'lf_relative_l2':float(torch.linalg.vector_norm(ax-y)/torch.linalg.vector_norm(y).clamp_min(1e-12))}
        history.append(row)
        if callback and (i in (1,5,10,20) or i==iterations):callback(row)
    torch.cuda.synchronize(operator.device) if operator.device.type=='cuda' else None
    return x[0].cpu(),{'update':update,'history':history,'seconds':time.perf_counter()-t0}


def phase_psf_plane(h):
    """MATLAB getPhaseSpaceLFPSF.m on one H[Y,X,U,V] depth plane."""
    h=np.asarray(h,dtype=np.float32)
    size,_,n,_=h.shape
    length=size-((size-n)%(2*n))
    if h.shape!=(size,size,n,n) or n%2!=1:raise ValueError('Square odd view grid required')
    start=(size-length)//2
    h=h[start:start+length,start:start+length]
    shifted=np.zeros_like(h)
    center=n//2
    for u in range(n):
        for v in range(n):
            dy,dx=u-center,v-center
            sy=slice(max(-dy,0),length-max(dy,0));sx=slice(max(-dx,0),length-max(dx,0))
            ty=slice(max(dy,0),length-max(-dy,0));tx=slice(max(dx,0),length-max(-dx,0))
            shifted[ty,tx,u,v]=h[sy,sx,u,v]
    k=length//n
    # [lenslet-y, detector-u, lenslet-x, detector-v, source-u, source-v]
    a=shifted.reshape(k,n,k,n,n,n).transpose(0,4,2,5,1,3)
    return a[:,::-1,:,::-1,:,:].reshape(length,length,n,n).copy()


def build_phase_psf_cache(source,destination,*,depth=91,progress=None):
    """Stream native H[Y,X,U,V,Z] into [U,V,Z,Y,X], retaining actual zero-shifts."""
    source=Path(source);destination=Path(destination)
    contract={'version':1,'source':file_record(source),'depth':depth,'conversion':'local_getPhaseSpaceLFPSF_v1'}
    token=fingerprint(contract)
    if destination.exists():
        with h5py.File(destination) as f:
            if f.attrs.get('fingerprint')!=token:raise ValueError('Phase cache contract differs; use a new path')
        return destination
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_suffix('.building.h5')
    with h5py.File(source) as inp,h5py.File(temporary,'w') as out:
        d=inp['H'];size,_,n,_,nz=d.shape
        if depth>nz or (nz-depth)%2:raise ValueError('Invalid centered depth selection')
        start=(nz-depth)//2;length=size-((size-n)%(2*n))
        ds=out.create_dataset('phase_psf',shape=(n,n,depth,length,length),dtype='f4',chunks=(1,1,1,length,length))
        energy=np.zeros((n,n),dtype=np.float64)
        for z in range(depth):
            a=phase_psf_plane(d[:,:,:,:,start+z])
            ds[:,:,z]=a.transpose(2,3,0,1)
            energy+=a.sum((0,1),dtype=np.float64)
            if progress and (z%15==0 or z+1==depth):progress(f'Phase PSF {z+1}/{depth}')
        span=float(energy.max()-energy.min())
        weight=(energy-energy.min())/span if span>0 else np.ones_like(energy)
        out.create_dataset('weights',data=weight.astype(np.float32))
        out.attrs['fingerprint']=token;out.attrs['contract']=json.dumps(contract)
        out.attrs['depth_start']=start
    temporary.replace(destination)
    return destination


def ptychographic_order(n):
    order=[]
    for lo in range((n+1)//2):
        hi=n-1-lo
        if lo==hi:order.append((lo,lo));continue
        order.extend((lo,j) for j in range(lo,hi))
        order.extend((i,hi) for i in range(lo,hi))
        order.extend((hi,j) for j in range(hi,lo,-1))
        order.extend((i,lo) for i in range(hi,lo,-1))
    return order


def matlab_bicubic_views(lf,n):
    """Upsample angular views using MATLAB's cubic kernel/symmetric extension."""
    h,w=lf.shape[-2:]
    views=F.pixel_unshuffle(lf.reshape(1,1,h,w),n)[0]
    iy,wy=_resize_weights(h//n,h);ix,wx=_resize_weights(w//n,w)
    iy,wy,ix,wx=[a.to(lf.device) for a in (iy,wy,ix,wx)]
    views=(views[:,iy,:]*wy[None,:,:,None]).sum(2)
    views=(views[:,:,ix]*wx[None,None,:,:]).sum(-1)
    return views.reshape(n,n,h,w).clamp_min(0)


class PhaseViewConvolution:
    """Batched-depth linear conv2('same') and its exact adjoint via FFT."""
    def __init__(self,kernel,image_shape):
        self.kernel=kernel
        self.h,self.w=image_shape
        self.kh,self.kw=kernel.shape[-2:]
        if self.kh%2!=1 or self.kw%2!=1:raise ValueError('Odd PSF support required')
        self.cy,self.cx=self.kh//2,self.kw//2
        self.fft_shape=(next_fast_len(self.h+self.kh-1),next_fast_len(self.w+self.kw-1))
        self.spectrum=torch.fft.rfft2(kernel,s=self.fft_shape)

    def forward(self,x):
        a=torch.fft.rfft2(x,s=self.fft_shape)
        full=torch.fft.irfft2((a*self.spectrum).sum(0),s=self.fft_shape)
        return full[self.cy:self.cy+self.h,self.cx:self.cx+self.w]

    def adjoint(self,y):
        padded=F.pad(y,(self.cx,self.fft_shape[1]-self.w-self.cx,self.cy,self.fft_shape[0]-self.h-self.cy))
        spectrum=torch.fft.rfft2(padded)
        result=torch.fft.irfft2(spectrum[None]*self.spectrum.conj(),s=self.fft_shape)
        return result[:,:self.h,:self.w]

    def sensitivity(self):
        # Exact rectangular-overlap sums, avoiding a depth-batched inverse FFT.
        k=self.kernel
        prefix=F.pad(k.cumsum(-2).cumsum(-1),(1,0,1,0))
        y=torch.arange(self.h,device=k.device);x=torch.arange(self.w,device=k.device)
        ly=(self.cy-y).clamp(0,self.kh);hy=(self.h+self.cy-y).clamp(0,self.kh)
        lx=(self.cx-x).clamp(0,self.kw);hx=(self.w+self.cx-x).clamp(0,self.kw)
        return (prefix[:,hy[:,None],hx[None,:]]-prefix[:,ly[:,None],hx[None,:]]
                -prefix[:,hy[:,None],lx[None,:]]+prefix[:,ly[:,None],lx[None,:]])


@torch.no_grad()
def reconstruct_phase_space(lf,cache,*,gain=1.,sweeps=2,device='cuda',callback=None):
    """Port of local non-FFT phase-space RL, evaluated with exact linear FFTs.

    A sweep visits all angular views. This is not the existing FFT variant's
    abs(ifft) convention. Gain changes the kernel scale, not the relative weights.
    """
    if sweeps<1 or gain<=0:raise ValueError('Positive sweeps/gain required')
    device=torch.device(device);y=lf.to(device).float().squeeze()
    if y.ndim!=2 or torch.any(y<0) or not torch.isfinite(y).all():raise ValueError('Invalid LF')
    t0=time.perf_counter();history=[]
    with h5py.File(cache) as f:
        d=f['phase_psf'];n,_,z,_,_=d.shape
        views=matlab_bicubic_views(y,n);weights=f['weights'][:]
        x=torch.ones((z,*y.shape),device=device)*y.sum()/(z*y.numel())
        order=ptychographic_order(n)
        for sweep in range(1,sweeps+1):
            used=0
            for ordinal,(u,v) in enumerate(order):
                alpha=float(weights[u,v])
                if alpha<=0:continue
                kernel=torch.from_numpy(d[u,v][:]).to(device)*gain
                op=PhaseViewConvolution(kernel,tuple(y.shape))
                prediction=op.forward(x)
                view=views[u,v]
                eps=max(float(view.max())*1e-7,1e-10)
                ratio=view/prediction.clamp_min(eps)
                correction=op.adjoint(ratio).clamp_min(0)/op.sensitivity().clamp_min(1e-12)
                x=(x*(1-alpha+alpha*correction)).clamp_min(0)
                if not torch.isfinite(x).all():raise FloatingPointError('Phase-space RL became nonfinite')
                used+=1
                if callback and (ordinal%30==0 or ordinal==len(order)-1):
                    callback({'sweep':sweep,'view':ordinal+1,'views':len(order)})
                del kernel,op,prediction,ratio,correction
            history.append({'sweep':sweep,'updated_views':used})
    if device.type=='cuda':torch.cuda.synchronize(device)
    return x.cpu(),{'sweeps':sweeps,'history':history,'seconds':time.perf_counter()-t0,
        'implementation':'local deconv_phase_space.m port; linear FFT; MATLAB cubic; minmax PSF-energy weights',
        'interpolation':'internal angular-view upsampling only; no repeat raw LF registration'}


def final_projection_relative_mse(prediction,clean_lf,operator,epsilon=1e-12):
    """Optional synthetic loss on physical projection of FINAL raw-domain output.

    Differentiable through the fixed operator. No fitted gain, normalization of
    the prediction, measured-PS target, or H2 learned forward surrogate is used.
    """
    expected=operator(prediction)
    if expected.shape!=clean_lf.shape:raise ValueError('LF shapes differ')
    error=(expected-clean_lf.float()).square().flatten(1).mean(1)
    energy=clean_lf.float().square().flatten(1).mean(1).clamp_min(epsilon)
    return (error/energy).mean()
