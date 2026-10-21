"""Positive linear correction of a calibrated native-grid LF operator.

This is an effective sensor-response family, not an optical PSF identification.
No image normalization, biases, interpolation, or nonlinear image processing.
"""
from dataclasses import dataclass
import math
import json
import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .physics_rearrange import space_to_view_depth, view_depth_to_space


@dataclass(frozen=True)
class ForwardCorrectionConfig:
    nnum: int = 11
    gain_deviation: float = 0.2
    max_redistribution: float = 0.2

    def __post_init__(self):
        if self.nnum < 1 or not 0 < self.gain_deviation < 1:
            raise ValueError('Invalid angular response configuration')
        if not 0 < self.max_redistribution < 1:
            raise ValueError('Redistribution fraction must be between zero and one')


class PositiveLFResponse(nn.Module):
    """Shared symmetric 3x3 lenslet redistribution and bounded view sensitivity.

    Each view uses the same spatial kernel. Positive mean-one view gains remove
    a global gain degree of freedom, but do not conserve flux for every angular
    distribution. Spatial kernels sum to one on an infinite grid; zero padding
    permits physical field loss. Never renormalize the cropped output.
    """
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ForwardCorrectionConfig()
        self.view_logits = nn.Parameter(torch.zeros(self.config.nnum**2))
        # Starts near identity, with usable gradients. Exact fixed-A0 baseline
        # must be evaluated separately, not called this initialization.
        self.redistribution_logit = nn.Parameter(torch.tensor(-5.))
        self.neighbor_logits = nn.Parameter(torch.zeros(4))

    def coefficients(self):
        c = self.config
        bounded = self.view_logits.tanh()
        gains = 1 + .5*c.gain_deviation*(bounded-bounded.mean())
        # Four antipodal pairs: horizontal, vertical and two diagonal axes.
        pair = self.neighbor_logits.softmax(0) / 2
        kernel = torch.stack((pair[2], pair[1], pair[3], pair[0],
                              pair.new_zeros(()), pair[0], pair[3], pair[1], pair[2]))
        fraction = c.max_redistribution*self.redistribution_logit.sigmoid()
        center = F.one_hot(torch.tensor(4, device=pair.device), 9).to(pair)
        return gains, ((1-fraction)*center + fraction*kernel).reshape(1,1,3,3)

    def forward(self, lf):
        if lf.ndim != 4 or lf.shape[1] != 1:
            raise ValueError('Expected native LF [batch,1,height,width]')
        views = space_to_view_depth(lf, self.config.nnum)
        gains, kernel = self.coefficients()
        filtered = F.conv2d(views, kernel.expand(views.shape[1],1,3,3),
                            padding=1, groups=views.shape[1])
        return view_depth_to_space(filtered*gains[None,:,None,None],
                                   channels=1, block_size=self.config.nnum)

    def regularization(self):
        gains, kernel = self.coefficients()
        return (gains-1).square().mean() + (1-kernel[0,0,1,1]).square()

    def adjoint(self,lf):
        """Exact transpose including view gain and zero-padded field boundaries."""
        views=space_to_view_depth(lf,self.config.nnum)
        gains,kernel=self.coefficients()
        corrected=F.conv_transpose2d(views*gains[None,:,None,None],
            kernel.expand(views.shape[1],1,3,3),padding=1,groups=views.shape[1])
        return view_depth_to_space(corrected,channels=1,block_size=self.config.nnum)


class CalibratedLearnedForward(nn.Module):
    """Compose C_phi(A0(x)); supports gradients to x for later feedback loss.

    A0 must be the raw-emitter-domain projector, not H1's cutoff-domain FP.
    Freeze correction parameters during initial reconstruction fine-tuning;
    preserve gradients through this module to the reconstruction output.
    """
    def __init__(self, physical_forward, correction):
        super().__init__()
        self.physical_forward = physical_forward
        self.correction = correction

    def forward(self, volume):
        return self.correction(self.physical_forward(volume))


@dataclass(frozen=True)
class PhaseRedistributionConfig:
    nnum: int = 11
    radius: int = 2
    max_redistribution: float = .5
    initial_centre_logit: float = 8.

    def __post_init__(self):
        if self.nnum<1 or self.radius<1 or not 0<self.max_redistribution<1:
            raise ValueError('Invalid positive phase redistribution configuration')


class PositivePhaseRedistribution(nn.Module):
    """Source-phase-dependent photon redistribution, 121*(25+8)=3993 logits.

    Positive coefficients sum to one for each SOURCE phase. Unlike row-normalizing
    an output-phase filter this conserves total flux on an infinite grid, for any
    angular distribution. Zero field boundaries lose escaping light. Local 5x5
    offsets and eight adjacent lenslets may be asymmetric. No output clipping,
    sample normalization, additive bias or deconvolution is used.
    """
    def __init__(self,config=None):
        super().__init__();self.config=config or PhaseRedistributionConfig()
        c=self.config;w=2*c.radius+1
        logits=torch.zeros(c.nnum**2,w*w+8);logits[:,w*w//2]=c.initial_centre_logit
        self.logits=nn.Parameter(logits)

    def coefficients(self):
        c=self.config;w=2*c.radius+1;centre=w*w//2
        other=torch.cat((self.logits[:,:centre],self.logits[:,centre+1:]),1)
        fraction=c.max_redistribution*torch.sigmoid(torch.logsumexp(other,1)-self.logits[:,centre])
        spread=other.softmax(1)*fraction[:,None]
        all_weights=torch.cat((spread[:,:centre],(1-fraction)[:,None],spread[:,centre:]),1)
        return all_weights[:,:w*w],all_weights[:,w*w:]

    def _phase(self,S,dy=0,dx=0):
        c=self.config;h,w=S.shape[-2:]
        return (((torch.arange(h,device=S.device)+dy)%c.nnum)[:,None]*c.nnum+
                ((torch.arange(w,device=S.device)+dx)%c.nnum)[None,:]).flatten()

    def forward(self,S):
        if S.ndim!=4 or S.shape[1]!=1:raise ValueError('Expected LF [B,1,H,W]')
        c=self.config;w=2*c.radius+1;local,lens=self.coefficients()
        patches=F.unfold(S,w,padding=c.radius)
        # Input is at p+offset, so its SOURCE phase selects the transport weight.
        weights=torch.stack([local[self._phase(S,dy,dx),i] for i,(dy,dx) in
            enumerate((dy,dx) for dy in range(-c.radius,c.radius+1) for dx in range(-c.radius,c.radius+1))])
        out=(patches*weights[None].to(S.dtype)).sum(1)
        neighbours=F.unfold(S,3,padding=c.nnum,dilation=c.nnum)
        neighbours=torch.cat((neighbours[:,:4],neighbours[:,5:]),1)
        out=out+(neighbours*lens[self._phase(S)].T[None].to(S.dtype)).sum(1)
        return out.reshape_as(S)

    def adjoint(self,S):
        c=self.config;w=2*c.radius+1;local,lens=self.coefficients();ph=self._phase(S)
        patches=F.unfold(S,w,padding=c.radius).flip(1)
        neighbours=F.unfold(S,3,padding=c.nnum,dilation=c.nnum)
        neighbours=torch.cat((neighbours[:,:4],neighbours[:,5:]),1).flip(1)
        out=(patches*local[ph].T[None].to(S.dtype)).sum(1)+(neighbours*lens[ph].T[None].to(S.dtype)).sum(1)
        return out.reshape_as(S)

    def regularization(self):
        local,lens=self.coefficients();centre=local.shape[1]//2
        return (1-local[:,centre]).square().mean()


@dataclass(frozen=True)
class LFResidualCNNConfig:
    nnum: int = 11
    channels: int = 16
    dilations: tuple = (1,2,4,11,1)
    sqrt_floor: float = 1e-12


class HomogeneousLFResidualCNN(nn.Module):
    """Positive degree-one homogeneous CNN; nonlinear, never baked as a linear PSF."""
    def __init__(self,config=None):
        super().__init__();self.config=config or LFResidualCNNConfig();c=self.config
        if not math.isfinite(c.sqrt_floor) or c.sqrt_floor<=0:raise ValueError('Positive finite square-root feature floor required')
        self.sqrt_floor=c.sqrt_floor;cin=6;layers=[]
        for i,d in enumerate(c.dilations):
            cout=2 if i==len(c.dilations)-1 else c.channels
            layers.append(nn.Conv2d(cin,cout,3,padding=d,dilation=d));cin=cout
        self.convs=nn.ModuleList(layers)
        nn.init.zeros_(self.convs[-1].weight);nn.init.zeros_(self.convs[-1].bias)

    def phase(self,H,W,device):
        n=self.config.nnum
        py=2*math.pi*((torch.arange(H,device=device)%n)+.5)/n
        px=2*math.pi*((torch.arange(W,device=device)%n)+.5)/n
        return torch.cat((torch.stack((py.sin(),py.cos()))[:,:,None].expand(2,H,W),
                          torch.stack((px.sin(),px.cos()))[:,None,:].expand(2,H,W)))

    def forward(self,S):
        m=S.amax((2,3),keepdim=True).clamp_min(1e-30);s=S/m
        x=torch.cat((s,s.clamp_min(self.sqrt_floor).sqrt(),self.phase(*S.shape[-2:],S.device)[None].expand(S.shape[0],-1,-1,-1)),1)
        for i,conv in enumerate(self.convs):
            x=conv(x)
            if i<len(self.convs)-1:x=F.gelu(x)
        c,r=x[:,:1],x[:,1:];self.last=(c,r)
        return S*(1+.5*c.tanh())+.1*m*torch.sigmoid(r-5)

    def regularization(self):
        c,r=self.last
        return (.5*c.tanh()).square().mean()+(.1*torch.sigmoid(r-5)).square().mean()


def make_forward_response(kind='shared',config=None):
    constructors={'shared':(PositiveLFResponse,ForwardCorrectionConfig),
                  'positive_phase':(PositivePhaseRedistribution,PhaseRedistributionConfig),
                  'cnn':(HomogeneousLFResidualCNN,LFResidualCNNConfig)}
    if kind not in constructors:raise ValueError('Unknown forward response family')
    cls,cfg=constructors[kind]
    return cls(cfg(**config) if isinstance(config,dict) else config)


def load_forward_response(path, *, kernel_sha256, device='cpu', frozen=True,require_validated=False):
    """Load against the authenticated A0; freezing does not detach input gradients."""
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload['kernel_sha256'] != kernel_sha256:
        raise ValueError('Forward correction was calibrated against a different kernel')
    if require_validated and payload.get('kind','shared')!='shared':
        report_path=Path(path).with_suffix('.validation.json')
        if not report_path.exists():raise ValueError('New response must pass sparse/dense numerical validation before training')
        report=json.loads(report_path.read_text())
        with Path(path).open('rb') as f:checkpoint_sha=hashlib.file_digest(f,'sha256').hexdigest()
        with Path(__file__).open('rb') as f:model_code_sha=hashlib.file_digest(f,'sha256').hexdigest()
        if not report.get('passed') or report['checkpoint_sha256']!=checkpoint_sha or report['kernel_sha256']!=kernel_sha256 or report['kind']!=payload['kind'] or report.get('model_code_sha256')!=model_code_sha:
            raise ValueError('Response validation does not match this checkpoint/operator')
    model = make_forward_response(payload.get('kind','shared'),payload['config'])
    model.load_state_dict(payload['state_dict'], strict=True)
    model.to(device)
    model.requires_grad_(not frozen)
    return model, payload
