"""An explicit, low-resolution emitter halo for paired observation supervision.

The existing reconstruction remains a 91-plane core. This auxiliary decoder
estimates the surrounding emitters from LF, without seeing PS/TF references.
It is NOT the proposed 151-plane physical LF reconstruction, nor calibrated
ground truth outside the field. Only PS/TF observation operators use this halo.
"""
from dataclasses import asdict
import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .boundary_context import ContextGeometry


class AuxiliaryEmitterContext(nn.Module):
    def __init__(self, geometry=ContextGeometry(), pitch=11, hidden=16, axial_knots=31):
        super().__init__()
        self.geometry = geometry
        self.pitch = pitch
        self.hidden = hidden
        self.axial_knots = axial_knots
        if any(n % pitch for n in geometry.core_shape[1:] + geometry.margin_zyx[1:]):
            raise ValueError('Lateral core and context must span whole lenslets')
        self.features = nn.Sequential(nn.Conv2d(pitch**2, hidden, 3, padding=1),
                                      nn.GELU(), nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU())
        self.head = nn.Conv2d(hidden, axial_knots, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, -5.)
        outside = torch.ones(geometry.extended_shape, dtype=torch.bool)
        outside[geometry.core_slices] = False
        self.register_buffer('outside', outside, persistent=False)

    def forward(self, lf, core):
        # The empty-exterior constraint is small. Half-precision head gradients
        # can underflow before GradScaler unscales them. This small decoder is
        # cheap enough to keep entirely FP32; the existing core still uses AMP.
        with torch.autocast(device_type=lf.device.type, enabled=False):
            return self._forward_float32(lf.float(), core.float())

    def _forward_float32(self, lf, core):
        if tuple(core.shape[-3:]) != self.geometry.core_shape:
            raise ValueError('Unexpected reconstruction core shape')
        if tuple(lf.shape) != (core.shape[0], 1, *self.geometry.core_shape[1:]):
            raise ValueError('Expected the existing registered LF, without re-export or synthetic padding')
        views = F.pixel_unshuffle(lf, self.pitch)
        features = self.features(views)
        mz, my, mx = self.geometry.margin_zyx
        # Padding FEATURES is a decoder operation, not an assumed emitter halo
        # or fabricated camera observation. The emitter halo is predicted below.
        features = F.pad(features, (mx//self.pitch, mx//self.pitch,
                                    my//self.pitch, my//self.pitch))
        knots = F.softplus(self.head(features).float())
        field = F.interpolate(knots[:, None], size=self.geometry.extended_shape,
                              mode='trilinear', align_corners=True)[:, 0]
        # Units follow the core, but halo supervision cannot manipulate its RMS
        # through this scale factor. No observation-derived intensity is input.
        scale = core.detach().float().square().mean((-3, -2, -1), keepdim=True).clamp_min(1e-16).sqrt()
        halo = field * scale * self.outside
        extended = F.pad(core.float(), (mx, mx, my, my, mz, mz)) + halo
        return extended, halo

    def empty_synthetic_halo_loss(self, halo, target):
        """Existing simulated objects are defined on a finite, zero-extended grid.

        This supervises ONLY that known-empty synthetic exterior. It never
        penalizes a real reconstruction's boundary or assumes real empty space.
        """
        energy = halo.square().sum((-3, -2, -1)) / self.outside.sum()
        target_energy = target.float().square().mean((-3, -2, -1)).clamp_min(1e-12)
        return (energy / target_energy).mean()

    def contract(self):
        return dict(type=type(self).__name__, geometry=asdict(self.geometry), pitch=self.pitch,
                    hidden=self.hidden, axial_knots=self.axial_knots,
                    parameters=sum(p.numel() for p in self.parameters()),
                    precision='FP32 auxiliary decoder even when the core uses AMP',
                    core='Existing H1/H2/fusion output, preserved exactly by assembly',
                    input='Measured LF only; detached predicted core RMS sets units',
                    exterior='Learned positive, spatially/axially smooth emitter estimate',
                    role='Auxiliary PS/TF supervision context; not a physical LF inversion of the halo',
                    synthetic_exterior='Known zero outside the finite simulated emitter grid',
                    measured_exterior='Unknown and estimated; no zero/replicate emitter padding assumption',
                    limitation='Halo is not identifiable from these losses alone; inspect spill contribution and common-core scores')


def save_context(path, decoder, *, core_checkpoint, epoch):
    """Bind the auxiliary state to the exact standard reconstruction checkpoint."""
    path, core_checkpoint = Path(path), Path(core_checkpoint)
    torch.save(dict(epoch=int(epoch), contract=decoder.contract(), state=decoder.state_dict(),
                    core_sha256=hashlib.sha256(core_checkpoint.read_bytes()).hexdigest()), path)


def load_context(path, *, core_checkpoint, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload['core_sha256'] != hashlib.sha256(Path(core_checkpoint).read_bytes()).hexdigest():
        raise ValueError('Auxiliary context belongs to a different core checkpoint')
    c = payload['contract']
    g = c['geometry']
    decoder = AuxiliaryEmitterContext(ContextGeometry(tuple(g['core_shape']), tuple(g['margin_zyx'])),
                                      pitch=c['pitch'], hidden=c['hidden'], axial_knots=c['axial_knots'])
    decoder.load_state_dict(payload['state'])
    return decoder.to(device).eval(), payload
