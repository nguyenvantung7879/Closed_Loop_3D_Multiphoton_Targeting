"""Explicit object context for PS observations; never invent an emitter halo.

The input is a larger *latent emitter volume*, not a padded PS observation.
The operator returns the central observation after linear convolution. This is
independent of a reconstruction architecture and does not extend a 91-plane
checkpoint or LF kernel by interpolation.
"""
from dataclasses import dataclass, asdict
import torch
from .point_scanning import PSForwardOperator


@dataclass(frozen=True)
class ContextGeometry:
    core_shape: tuple = (91, 517, 517)
    margin_zyx: tuple = (30, 33, 33)

    def __post_init__(self):
        if len(self.core_shape) != 3 or len(self.margin_zyx) != 3:
            raise ValueError('Require Z,Y,X shapes')
        if any(int(n) != n or n <= 0 for n in self.core_shape):
            raise ValueError('Invalid core shape')
        if any(int(n) != n or n < 0 for n in self.margin_zyx):
            raise ValueError('Invalid context margin')

    @property
    def extended_shape(self):
        return tuple(n + 2*m for n,m in zip(self.core_shape,self.margin_zyx))

    @property
    def core_slices(self):
        return tuple(slice(m,m+n) for n,m in zip(self.core_shape,self.margin_zyx))

    def crop(self, volume):
        if tuple(volume.shape[-3:]) != self.extended_shape:
            raise ValueError('Expected the explicitly reconstructed extended volume')
        return volume[(...,*self.core_slices)]

    def contract(self):
        return {**asdict(self), 'extended_shape':self.extended_shape,
                'halo_domain':'latent fluorophore intensity; not the observed PS stack'}


class ContextPSOperator:
    def __init__(self, kernel, geometry=ContextGeometry(), **kwargs):
        self.geometry=geometry
        self.full=PSForwardOperator(kernel, geometry.extended_shape, **kwargs)
        with torch.no_grad():
            coverage=self.full(torch.ones(geometry.extended_shape,
                device=self.full.device,dtype=self.full.dtype))
            self.coverage=geometry.crop(coverage).clone()
        self.valid_mask=self.coverage>=self.full.minimum_coverage

    def __call__(self, extended_volume):
        return self.geometry.crop(self.full(extended_volume))

    def contract(self):
        return {**self.geometry.contract(), 'minimum_coverage':self.full.minimum_coverage,
                'core_minimum_coverage':float(self.coverage.min()),
                'core_valid_fraction':int(self.valid_mask.sum())/self.valid_mask.numel(),
                'coverage_interpretation':'unit-field PSF coverage, not an error bound for arbitrarily bright outside emitters',
                'boundary':'linear convolution on enlarged object domain, then observation crop'}


def weighted_shape_mse(predicted_observation, observed, weight, epsilon=1e-8):
    """Same weighted RMS on both sides; target is an unfiltered observation."""
    if predicted_observation.shape != observed.shape or predicted_observation.ndim != 4:
        raise ValueError('Expected matching B,Z,Y,X observations')
    weight=weight.to(observed)
    if tuple(weight.shape) != tuple(observed.shape[-3:]) or weight.min()<0 or weight.sum()<=0:
        raise ValueError('Invalid observation weights')
    dims=(-3,-2,-1)
    def rms(x):
        return ((x.square()*weight).sum(dims,keepdim=True)/weight.sum()).clamp_min(epsilon**2).sqrt()
    difference=predicted_observation/rms(predicted_observation)-observed/rms(observed)
    return ((difference.square()*weight).sum(dims)/weight.sum()).mean()


def context_ps_loss(extended_prediction, observed_core, operator):
    return weighted_shape_mse(operator(extended_prediction.float()),
        observed_core.float(), operator.valid_mask)


@torch.no_grad()
def boundary_diagnostics(raw, predicted_ps, observed_ps, mask):
    """Intensity location is a diagnostic, not an artifact fraction or penalty."""
    dims=(-3,-2,-1);outside=~mask
    energy=raw.float().square()
    fraction=(energy*outside).sum(dims)/energy.sum(dims).clamp_min(1e-20)
    full=weighted_shape_mse(predicted_ps,observed_ps,torch.ones_like(mask))
    edge=weighted_shape_mse(predicted_ps,observed_ps,outside) if outside.any() else full*0
    return dict(outside_scored_squared_intensity_fraction=float(fraction.mean()),
        whole_volume_ps_shape_mse_diagnostic=float(full),
        boundary_ps_shape_mse_diagnostic=float(edge),
        scored_fraction=float(mask.float().mean()))
