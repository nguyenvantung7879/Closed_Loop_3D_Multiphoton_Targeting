"""Authenticated calibrated detection kernel and optional linear LF response."""
from pathlib import Path
import hashlib,json
import torch
from .reconstruction_audit import FixedLFOperator
from .forward_model import LFMForwardConfig
LEGACY_AUGUST_SHA256='435d921546288e4c1cf63cca3e966a8993999de7d3b50408bedc1583d91a13c6'

def sha256(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def load_physical_operator(path,*,device='cuda',dtype=torch.float32,require_validated=False):
    path=Path(path);metadata=path.with_suffix('.json')
    digest=sha256(path)
    if not metadata.exists() and Path(str(path)+'.json').exists() and digest!=LEGACY_AUGUST_SHA256:
        raise ValueError('Raw twin-package metadata (.bin.json) requires explicit integration of physical spacing, registration warp, gain and numerical validation; it cannot be loaded as a legacy August kernel.')
    if metadata.exists():
        contract=json.loads(metadata.read_text())
        if contract.get('schema')!='lfrln.calibrated_volume_operator.v1':raise ValueError('Unknown calibrated kernel contract')
        if digest!=contract['kernel_sha256']:raise ValueError('Calibrated kernel changed')
        if require_validated and contract.get('validation_status')!='validated':raise ValueError('Calibrated kernel must pass validation before training')
        if require_validated and sha256(contract['validation'])!=contract['validation_sha256']:raise ValueError('Kernel validation record changed')
        config=LFMForwardConfig(**contract['config']);gain=contract['gain']
    else:
        if digest!=LEGACY_AUGUST_SHA256:raise ValueError('Unknown optical kernel requires an authenticated calibration contract; no implicit August gain')
        config=LFMForwardConfig();gain=.0005535736252921639
        contract=dict(schema='legacy_august_kernel',kernel=str(path),kernel_sha256=digest,gain=gain,validation_status='known_august_sha256')
    return FixedLFOperator(path,gain,config=config,device=device,dtype=dtype),contract

class LinearCalibratedOperator:
    def __init__(self,physical,correction=None,excitation=None):
        self.physical=physical;self.correction=correction;self.device=physical.device;self.config=physical.config
        if excitation is not None:
            if getattr(self.config,'apply_axial_profile',False):raise ValueError('Excitation would be applied twice')
            if excitation.ndim!=4 or excitation.shape[:2]!=(1,self.config.depth) or not torch.isfinite(excitation).all() or excitation.min()<0 or excitation.max()<=0:
                raise ValueError('Invalid emitter-domain excitation')
        self.excitation=excitation.detach() if excitation is not None else None
    def __call__(self,x):
        y=self.physical(x if self.excitation is None else x*self.excitation.to(x))
        return self.correction(y) if self.correction is not None else y
    def adjoint(self,y):
        x=self.physical.adjoint(self.correction.adjoint(y) if self.correction is not None else y)
        return x if self.excitation is None else x*self.excitation.to(x)
