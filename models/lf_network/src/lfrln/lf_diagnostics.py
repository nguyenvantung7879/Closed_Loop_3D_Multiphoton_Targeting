"""Display and perturb LF inputs without mixing spatial and angular axes."""
import numpy as np
from scipy.ndimage import gaussian_filter


def smooth_spatial_component(lf, nnum=11, sigma_lenslets=2.):
    a=np.asarray(lf,dtype=np.float32).squeeze()
    if a.ndim!=2 or any(v%nnum for v in a.shape):
        raise ValueError('Expected LF image divisible by its angular grid')
    # [lenslet_y, angular_y, lenslet_x, angular_x]: smooth only lenslet axes.
    grid=a.reshape(a.shape[0]//nnum,nnum,a.shape[1]//nnum,nnum)
    return gaussian_filter(grid,sigma=(sigma_lenslets,0,sigma_lenslets,0),mode='reflect').reshape(a.shape)


def remove_smooth_component(lf, strength, nnum=11, sigma_lenslets=2.):
    if not 0<=strength<1:raise ValueError('Strength must lie in [0,1)')
    a=np.asarray(lf,dtype=np.float32).squeeze()
    if (a<0).any() or not np.isfinite(a).all() or a.sum()<=0:
        raise ValueError('Expected finite positive LF signal')
    b=np.maximum(a-strength*smooth_spatial_component(a,nnum,sigma_lenslets),0)
    b*=a.sum(dtype=np.float64)/b.sum(dtype=np.float64)
    return b


def render_lf_context(lf,path,title,nnum=11):
    import matplotlib.pyplot as plt
    a=np.asarray(lf).squeeze()
    if a.ndim!=2 or any(v%nnum for v in a.shape):raise ValueError('Invalid LF angular grid')
    spatial=a.reshape(a.shape[0]//nnum,nnum,a.shape[1]//nnum,nnum).sum((1,3))
    fig,axes=plt.subplots(1,2,figsize=(9,4.5),layout='constrained')
    for ax,image,label,units in zip(axes,(a,spatial),('Actual network LF input','Lenslet-sum preview (display only)'),('Detector pixels','Lenslet indices')):
        ax.imshow(image,origin='lower',cmap='magma',vmin=0,vmax=max(float(image.max()),1e-12))
        ax.set(title=label,xlabel=units,ylabel=units)
    fig.suptitle(title+'\nIndependent display maxima; the detector image is not a PS object-space projection.',fontsize=9)
    fig.savefig(path,dpi=140);plt.close(fig)
