"""Auxiliary PS-domain geometry losses; apply the PS response before calling.

Predictions and measured observations share the same coverage mask. One mass
normalization per volume preserves the depth distribution; there is no per-plane
normalization, MIP, extra excitation envelope, or filtering of the target.
"""
import torch


def ps_projection_losses(predicted_ps, observed_ps, valid_mask, epsilon=1e-8):
    if predicted_ps.shape != observed_ps.shape or predicted_ps.ndim != 4:
        raise ValueError('Expected matching B,Z,Y,X PS observations')
    if tuple(valid_mask.shape) != tuple(observed_ps.shape[-3:]):
        raise ValueError('PS coverage mask must have shape Z,Y,X')
    if valid_mask.dtype != torch.bool or not valid_mask.any():
        raise ValueError('Require a nonempty boolean PS coverage mask')
    if epsilon <= 0:
        raise ValueError('epsilon must be positive')
    mask = valid_mask.to(device=predicted_ps.device)
    # PS convolution is nonnegative analytically; remove only numerical negative
    # intensities from the auxiliary probability masses. The voxel loss is unchanged.
    p = predicted_ps.clamp_min(0) * mask
    t = observed_ps.to(predicted_ps).clamp_min(0) * mask
    dims = (-3, -2, -1)
    p = p / p.sum(dims, keepdim=True).clamp_min(epsilon)
    t = t / t.sum(dims, keepdim=True).clamp_min(epsilon)
    difference = p - t
    xz = difference.sum(dim=-2).abs().sum(dim=(-2, -1)).mean()
    yz = difference.sum(dim=-1).abs().sum(dim=(-2, -1)).mean()
    axial = difference.sum(dim=(-2, -1)).abs().sum(dim=-1).mean()
    return dict(bio_ps_xz=xz, bio_ps_yz=yz,
                bio_ps_projection=(xz + yz) / 2,
                bio_ps_axial_profile=axial)
