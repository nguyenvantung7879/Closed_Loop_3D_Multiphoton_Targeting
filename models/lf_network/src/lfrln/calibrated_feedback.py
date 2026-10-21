"""Restricted forward adaptation and observation/geometry losses.

The two sampling parameters below are a diagnostic post-render correction,
not a replacement for native camera integration or a fit of the MLA optics.
No discriminator or unrestricted learned image-to-image forward network is used.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class SensorSamplingCorrection(nn.Module):
    """Shared residual rotation/pitch around a fixed full-image coordinate origin.

    Translation is deliberately absent: it is confounded with bead position.
    Bilinear interpolation preserves positivity. Photometry handles amplitude;
    this module must not be interpreted as a flux-calibrated camera response.
    """
    def __init__(self, rotation_bound_deg=.4, relative_pitch_bound=.005):
        super().__init__()
        if rotation_bound_deg<=0 or not 0<relative_pitch_bound<1:
            raise ValueError('Invalid physical parameter bounds')
        self.rotation_bound_deg=float(rotation_bound_deg)
        self.relative_pitch_bound=float(relative_pitch_bound)
        self.q=nn.Parameter(torch.zeros(2))

    def parameters_physical(self):
        return self.rotation_bound_deg*self.q[0].tanh(), 1+self.relative_pitch_bound*self.q[1].tanh()

    def forward(self,image,*,roi_start_yx=(0,0),full_shape_yx=None):
        if image.ndim!=4 or image.shape[1]!=1:
            raise ValueError('Expected B,1,Y,X LF intensity')
        h,w=image.shape[-2:]
        full_shape_yx=full_shape_yx or (h,w)
        angle,pitch=self.parameters_physical();angle=angle*math.pi/180
        yy,xx=torch.meshgrid(torch.arange(h,device=image.device,dtype=image.dtype),
                            torch.arange(w,device=image.device,dtype=image.dtype),indexing='ij')
        cy,cx=[(n-1)/2 for n in full_shape_yx]
        y=yy+roi_start_yx[0]-cy;x=xx+roi_start_yx[1]-cx
        # Inverse mapping: image rotated counterclockwise by angle and enlarged
        # by pitch in the registered sensor coordinate convention.
        source_x=(angle.cos()*x+angle.sin()*y)/pitch+cx-roi_start_yx[1]
        source_y=(-angle.sin()*x+angle.cos()*y)/pitch+cy-roi_start_yx[0]
        grid=torch.stack((2*source_x/(w-1)-1,2*source_y/(h-1)-1),-1)
        return F.grid_sample(image,grid[None].expand(image.shape[0],-1,-1,-1),
                             align_corners=True,padding_mode='zeros',mode='bilinear')

    def contract(self):
        angle,pitch=self.parameters_physical()
        return dict(rotation_deg=float(angle.detach()),pitch_multiplier=float(pitch.detach()),
            rotation_bound_deg=self.rotation_bound_deg,relative_pitch_bound=self.relative_pitch_bound,
            translation_fitted=False,kind='post-render sampling diagnostic; not native-export calibration')


def variance_weighted_feedback(predicted,measured,variance,*,mask=None):
    """Diagonal Gaussian discrepancy in a shared count/intensity domain.

    Variance is detached; the network cannot reduce the loss by inflating it.
    Correlated resized-camera noise requires whitening for an exact likelihood.
    This implementation is explicitly a diagonal approximation.
    """
    if predicted.shape!=measured.shape or not torch.isfinite(variance).all() or (variance<=0).any():
        raise ValueError('Feedback requires matching observations and positive finite variance')
    if mask is None:mask=torch.ones_like(measured,dtype=torch.bool)
    mask=torch.broadcast_to(mask,measured.shape)
    if not mask.any():raise ValueError('Empty LF feedback mask')
    return ((predicted-measured).square()/variance.detach())[mask].mean()


class CalibratedLFObservation:
    """A fixed, documented A(W f) followed by sampling and camera photometry."""
    def __init__(self,detector,*,excitation,background,gain,sampling=None):
        if excitation is None or background is None or gain is None:
            raise ValueError('Explicit excitation, background and photometric gain are required')
        if not torch.isfinite(excitation).all() or (excitation<0).any():
            raise ValueError('Invalid excitation profile')
        if not math.isfinite(float(gain)) or float(gain)<=0:raise ValueError('Invalid gain')
        self.detector=detector;self.excitation=excitation.detach()
        self.background=background;self.gain=float(gain);self.sampling=sampling
        if sampling is not None:
            sampling.requires_grad_(False)

    def __call__(self,volume):
        prediction=self.detector(volume*self.excitation.to(volume))
        if self.sampling is not None:prediction=self.sampling(prediction.float())
        return prediction*self.gain+self.background


def weak_bead_geometry_loss(volume,center_zyx_um,*,spacing_um=(.5,.21306818181818185,.21306818181818185),
                            diameter_um=2.,position_sigma_um=(.5,.3,.3),diameter_tolerance_um=.5):
    """Scale-invariant moment supervision from a *conditional* finite-bead fit.

    A uniform solid sphere has per-axis variance r^2/5. Tolerances encode weak
    geometry, not independently measured localization confidence. No voxel GT
    or absolute brightness is claimed. Call only on isolated-bead volumes.
    """
    if volume.ndim!=4 or diameter_um<=0 or diameter_tolerance_um<=0 or min(position_sigma_um)<=0:
        raise ValueError('Invalid finite bead geometry')
    mass=volume.float().clamp_min(0)
    total=mass.sum((-3,-2,-1),keepdim=True)
    if (total<=0).any():raise ValueError('Empty reconstruction has undefined bead geometry')
    density=mass/total
    center=torch.as_tensor(center_zyx_um,device=volume.device,dtype=torch.float32)
    if center.shape!=(volume.shape[0],3):raise ValueError('Expected one center per volume')
    losses=[]
    for axis,(n,spacing,sigma) in enumerate(zip(volume.shape[-3:],spacing_um,position_sigma_um)):
        coord=(torch.arange(n,device=volume.device)-(n-1)/2)*spacing
        shape=[1,1,1,1];shape[axis+1]=n;coord=coord.reshape(shape)
        mean=(density*coord).sum((-3,-2,-1))
        variance=(density*(coord-mean[:,None,None,None]).square()).sum((-3,-2,-1))
        desired=(diameter_um/2)**2/5
        tolerance=((diameter_um+diameter_tolerance_um)/2)**2/5-desired
        losses.append(((mean-center[:,axis])/sigma).square()+((variance-desired)/tolerance).square())
    return torch.stack(losses).mean()
