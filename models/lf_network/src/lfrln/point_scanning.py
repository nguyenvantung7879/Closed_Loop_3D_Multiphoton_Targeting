"""Fixed PS imaging response and prediction-side paired supervision.

The supplied MATLAB array is already a two-photon intensity response. It is
never squared again. The paired output is defined as latent fluorophore
structure; synthetic replay uses uniform excitation. Measured LF feedback
requires its separate calibrated excitation profile and is not enabled here.
"""
from pathlib import Path
import hashlib
import h5py
import numpy as np
from scipy.fft import next_fast_len
import torch


def read_ps_response(path):
    path=Path(path)
    with h5py.File(path,'r') as f:
        # MATLAB logical [Y,X,Z] is stored as HDF5 [Z,X,Y].
        h=np.asarray(f['I_emPSF'],dtype=np.float64).transpose(0,2,1).copy()
        dx=float(f['dx_Sample'][()].item())*1e6
        dz=float(f['dz_Sample'][()].item())*1e6
    if h.ndim!=3 or any(n%2!=1 for n in h.shape):raise ValueError('Expected odd 3D PSF support')
    if not np.isfinite(h).all() or h.min()<0 or h.sum()<=0:raise ValueError('Invalid intensity PSF')
    if not np.isfinite([dx,dz]).all() or min(dx,dz)<=0:raise ValueError('Invalid PSF spacing')
    original_sum=float(h.sum());h/=original_sum
    record=dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        matlab_variable='I_emPSF',matlab_axes='Y,X,Z',torch_axes='Z,Y,X',shape=list(h.shape),
        spacing_um=[dz,dx,dx],original_sum=original_sum,response='already two-photon intensity; no extra squaring',
        support='full supplied support; no truncation')
    return torch.from_numpy(h.astype('float32')),record


class _FixedConvolution(torch.autograd.Function):
    @staticmethod
    def forward(ctx,volume,operator):
        ctx.operator=operator
        return operator._apply_fft(volume,adjoint=False)

    @staticmethod
    def backward(ctx,gradient):
        return ctx.operator._apply_fft(gradient,adjoint=True),None


class PSForwardOperator:
    """Zero-extended linear 3D convolution with an exact fixed adjoint.

    The custom backward saves no full padded input spectrum, keeping a
    full-support PSF practical for 91x517x517 training volumes.
    """
    def __init__(self,kernel,volume_shape,*,device='cpu',dtype=torch.float32,
                 spacing_um=(.5,.21306818181818185,.21306818181818185),minimum_coverage=.99):
        kernel=torch.as_tensor(kernel,dtype=dtype,device=device)
        if kernel.ndim!=3 or any(n%2!=1 for n in kernel.shape):raise ValueError('Expected odd 3D kernel')
        if not torch.isfinite(kernel).all() or (kernel<0).any() or kernel.sum()<=0:raise ValueError('Invalid PSF')
        if len(volume_shape)!=3 or min(volume_shape)<1:raise ValueError('Invalid volume shape')
        if not 0<minimum_coverage<=1:raise ValueError('Invalid coverage threshold')
        self.shape=tuple(volume_shape);self.device=kernel.device;self.dtype=dtype
        self.spacing_um=tuple(spacing_um);self.minimum_coverage=float(minimum_coverage)
        self.kernel_shape=tuple(kernel.shape)
        self.fft_shape=tuple(next_fast_len(a+b-1) for a,b in zip(self.shape,kernel.shape))
        padded=torch.zeros(self.fft_shape,device=self.device,dtype=dtype)
        padded[tuple(slice(0,n) for n in kernel.shape)]=kernel/kernel.sum()
        # Put zero lag at FFT origin; negative lags wrap into padding only.
        padded=torch.roll(padded,tuple(-(n//2) for n in kernel.shape),dims=(0,1,2))
        self.spectrum=torch.fft.rfftn(padded)
        del padded,kernel
        with torch.no_grad():
            coverage=self._apply_fft(torch.ones(self.shape,device=self.device,dtype=dtype),adjoint=False)
            self.valid_mask=coverage>=minimum_coverage
        if not self.valid_mask.any():raise ValueError('No valid region at this coverage threshold')

    @classmethod
    def from_mat(cls,path,volume_shape,*,expected_spacing_um=None,**kwargs):
        kernel,record=read_ps_response(path)
        if expected_spacing_um is not None and not np.allclose(record['spacing_um'],expected_spacing_um,rtol=0,atol=1e-8):
            raise ValueError('PSF spacing differs from the explicitly chosen registered grid')
        obj=cls(kernel,volume_shape,spacing_um=record['spacing_um'],**kwargs)
        obj.source_record=record
        return obj

    def _apply_fft(self,volume,*,adjoint):
        if tuple(volume.shape[-3:])!=self.shape or volume.device!=self.device or volume.dtype!=self.dtype:
            raise ValueError('Volume shape/device/dtype differs from the PS operator')
        with torch.autocast(device_type=self.device.type,enabled=False):
            spectrum=torch.fft.rfftn(volume,s=self.fft_shape,dim=(-3,-2,-1))
            spectrum=spectrum*(self.spectrum.conj() if adjoint else self.spectrum)
            result=torch.fft.irfftn(spectrum,s=self.fft_shape,dim=(-3,-2,-1))
            return result[(...,*tuple(slice(0,n) for n in self.shape))].contiguous()

    def __call__(self,volume):return _FixedConvolution.apply(volume,self)

    def contract(self):
        return dict(psf=getattr(self,'source_record',None),volume_shape=list(self.shape),fft_shape=list(self.fft_shape),
            spacing_um=list(self.spacing_um),boundary='zero extension; linear FFT, no circular wrap',
            minimum_coverage=self.minimum_coverage,valid_fraction=float(self.valid_mask.float().mean()),
            paired_output_domain='latent fluorophore structure',synthetic_excitation='uniform, W=1',
            measured_lf_consistency_enabled=False)


def ps_forward_relative_mse(prediction,target,operator,*,epsilon=1e-8,return_prediction=False):
    """RMS-normalized MSE of P(prediction) vs observed PS; target is unfiltered.

    Both RMS estimates and MSE use the same >=99% sensitivity region, avoiding
    loss pressure to compensate for unknown emitters beyond the volume edge.
    """
    if prediction.shape!=target.shape or prediction.ndim!=4:raise ValueError('Expected matching [B,Z,Y,X]')
    if epsilon<=0:raise ValueError('epsilon must be positive')
    predicted_ps=operator(prediction.float())
    target=target.float();mask=operator.valid_mask.to(target.dtype)
    dims=(-3,-2,-1);count=mask.sum()
    rms=lambda v:((v.square()*mask).sum(dims,keepdim=True)/count).clamp_min(epsilon**2).sqrt()
    error=(predicted_ps/rms(predicted_ps)-target/rms(target)).square()
    loss=((error*mask).sum(dims)/count).mean()
    return (loss,predicted_ps) if return_prediction else loss
