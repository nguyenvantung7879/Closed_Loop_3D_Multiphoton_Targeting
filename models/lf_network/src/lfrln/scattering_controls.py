"""Positive effective LF scattering controls, independent of training.

This is a low-parameter intensity-transport closure on registered LF phase
space. It preserves the existing clear diffraction model but is NOT a
coherent propagation model or a microscopic estimate of mu_s, mu_a or g.
Angular redistribution is conditional on remaining in the represented
angular grid; throughput is a separate parameter. Spatial boundary loss is
retained, with zero extension and no wraparound or post-crop normalization.
"""
from dataclasses import asdict, dataclass
import math
import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class ScatteringConfig:
    lenslet_pixels: int = 11
    object_pixel_um: float = .21306818181818185
    spatial_fwhm_um: float = 31.875
    angular_sigma_fraction: float = .4
    diffuse_fraction: float = .5
    throughput: float = 1.

    def __post_init__(self):
        if self.lenslet_pixels<1 or self.lenslet_pixels%2!=1:
            raise ValueError('Require an odd positive lenslet grid')
        values=(self.object_pixel_um,self.spatial_fwhm_um,
                self.angular_sigma_fraction,self.diffuse_fraction,self.throughput)
        if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite configuration')
        if self.object_pixel_um<=0 or min(self.spatial_fwhm_um,self.angular_sigma_fraction)<0:
            raise ValueError('Invalid scattering scale')
        if not 0<=self.diffuse_fraction<=1 or not 0<=self.throughput<=1:
            raise ValueError('Mixture fraction and throughput must lie in [0,1]')

    def record(self):return asdict(self)


def to_phase_space(image, pitch=11):
    image=np.asarray(image)
    if image.ndim!=2 or any(n%pitch for n in image.shape):
        raise ValueError('Expected a 2D registered LF with complete lenslets')
    h,w=image.shape
    return image.reshape(h//pitch,pitch,w//pitch,pitch).transpose(0,2,1,3)


def from_phase_space(phase):
    a=np.asarray(phase)
    if a.ndim!=4 or a.shape[2]!=a.shape[3]:raise ValueError('Expected [Y,X,U,V]')
    return a.transpose(0,2,1,3).reshape(a.shape[0]*a.shape[2],a.shape[1]*a.shape[3])


def angular_transition(pitch, sigma_fraction):
    """Column-stochastic redistribution on one finite angular axis.

    Fraction is sigma / half angular-index range. No conversion to radians
    is asserted. Zero is identity; very large sigma approaches uniform mixing.
    """
    if sigma_fraction<0 or not math.isfinite(sigma_fraction):raise ValueError('Invalid angular width')
    if sigma_fraction==0 or pitch==1:return np.eye(pitch,dtype=np.float64)
    sigma=sigma_fraction*(pitch-1)/2
    d=np.arange(pitch)[:,None]-np.arange(pitch)[None,:]
    matrix=np.exp(-.5*(d/sigma)**2)
    return matrix/matrix.sum(axis=0,keepdims=True)


def diffuse_response(image, config=ScatteringConfig()):
    """Linear redistribution; signed observed templates are allowed for fitting.

    A nonnegative expected LF produces a nonnegative expected output. Signed
    data are never clipped during measurement fitting, avoiding tail bias.
    """
    a=np.asarray(image,dtype=np.float64)
    if not np.isfinite(a).all():raise ValueError('Nonfinite LF')
    phase=to_phase_space(a,config.lenslet_pixels)
    sigma=config.spatial_fwhm_um/(math.sqrt(8*math.log(2))*config.object_pixel_um*config.lenslet_pixels)
    if sigma>0:
        phase=gaussian_filter(phase,(sigma,sigma,0,0),mode='constant',cval=0,truncate=4)
    if config.angular_sigma_fraction>0:
        q=angular_transition(config.lenslet_pixels,config.angular_sigma_fraction)
        phase=np.einsum('au,xyuv,bv->xyab',q,phase,q,optimize=True)
    return from_phase_space(phase)


def scatter_lf(image, config=ScatteringConfig(), *, return_components=False):
    a=np.asarray(image,dtype=np.float64)
    # Preserve the exact clear limit; no unnecessary blur or dtype rounding.
    to_phase_space(a,config.lenslet_pixels)
    if not np.isfinite(a).all():raise ValueError('Nonfinite LF')
    eta=config.diffuse_fraction;t=config.throughput
    ballistic=t*(1-eta)*a
    scattered=np.zeros_like(a) if eta==0 else t*eta*diffuse_response(a,config)
    total=ballistic+scattered
    if return_components:
        return total,dict(ballistic=ballistic,scattered=scattered,
            input_sum=float(a.sum()),output_sum=float(total.sum()),
            scattered_sum=float(scattered.sum()),ballistic_sum=float(ballistic.sum()))
    return total


def fit_gain_offset(template, measured):
    """Nonnegative signal gain and unconstrained additive count baseline.

    The intercept includes camera offset and estimated-background error;
    measurements may already be signed after offset subtraction.
    """
    x=np.asarray(template,dtype=np.float64).ravel();y=np.asarray(measured,dtype=np.float64).ravel()
    xc=x-x.mean();yc=y-y.mean();den=xc@xc
    gain=max(float(xc@yc)/max(float(den),1e-24),0.)
    offset=float(y.mean()-gain*x.mean())
    residual=gain*x+offset-y
    return dict(gain=gain,offset=offset,
                normalized_mse=float(residual@residual/max(float(yc@yc),1e-24)))
