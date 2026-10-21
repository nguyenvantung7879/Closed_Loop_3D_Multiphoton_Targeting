"""Explicit effective fluorescence excitation, separate from LF detection optics."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from .forward_model import lf_axial_profile


def load_lf_excitation(path, *, depth=91, depth_step_um=.5, device='cpu'):
    path=Path(path)
    data=path.read_bytes(); record=json.loads(data)
    if record.get('schema')!='lfrln.lf_excitation.v1' or record.get('semantics')!='effective_fluorescence':
        raise ValueError('Require an explicit effective-fluorescence excitation contract')
    if record.get('kind')!='extended_tf_axial':raise ValueError('Unsupported LF excitation model')
    if record.get('focus_z_um',0)!=0 or record.get('depth_step_um',depth_step_um)!=depth_step_um:
        raise ValueError('Excitation metadata differ from the focal-plane reconstruction grid')
    profile=torch.tensor(record['profile'],device=device,dtype=torch.float32)
    z=np.asarray(record['z_um'],dtype=np.float64)
    expected=(np.arange(depth)-(depth-1)/2)*depth_step_um
    if profile.shape!=(depth,) or z.shape!=(depth,) or not np.allclose(z,expected,atol=1e-8,rtol=0):
        raise ValueError('Excitation depth coordinates differ from the reconstruction grid')
    if not torch.isfinite(profile).all() or profile.min()<=0 or profile.max()>1.000001:
        raise ValueError('Require finite positive peak-normalized fluorescence excitation')
    if record['kind']=='extended_tf_axial':
        fwhm=float(record['fwhm_um'])
        if not np.isfinite(fwhm) or fwhm<=0:raise ValueError('Invalid LF excitation FWHM')
        reference=lf_axial_profile(depth,depth_step_um=depth_step_um,fwhm_um=fwhm,device=device).flatten()
        if not torch.allclose(profile,reference,atol=1e-7,rtol=1e-6):raise ValueError('Excitation formula and stored values disagree')
    record=dict(record,path=str(path.resolve()),sha256=hashlib.sha256(data).hexdigest())
    return profile.reshape(1,depth,1,1),record


def axial_error_weights(excitation):
    """Legacy bounded edge emphasis; this is not an illumination inverse."""
    if excitation.ndim!=4 or excitation.shape[0]!=1 or excitation.shape[-2:]!=(1,1):
        raise ValueError('Expected axial excitation [1,Z,1,1]')
    if not torch.isfinite(excitation).all() or excitation.min()<=0 or excitation.max()>1.000001:
        raise ValueError('Invalid effective excitation')
    return 1-excitation+excitation.min()
