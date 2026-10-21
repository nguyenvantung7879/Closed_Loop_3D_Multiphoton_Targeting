from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch


VolumeL2Reduction = Literal[
    "voxel_mean", "relative_per_sample", "energy_capped_per_sample"
]


@dataclass(frozen=True)
class H2LossConfig:
    """Raw-volume supervision shared by H2-only and H2+fusion training."""

    supervised_scale: float = 1.0
    l2_reduction: VolumeL2Reduction = "energy_capped_per_sample"
    relative_l2_epsilon: float = 1e-12
    relative_l2_energy_floor: float = 1e-6
    high_frequency_weight: float = 0.0
    gradient_reduction: VolumeL2Reduction = "energy_capped_per_sample"
    gradient_spacing_um: tuple[float, float, float] = (1.0, 1.0, 1.0)
    gradient_energy_floor: float = 1e-6
    view_attention_identity_weight: float = 0.0
    view_attention_smoothness_weight: float = 0.0
    fp_consistency_weight: float = 0.0
    fp_flux_weight: float = 0.0
    fp_relative_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.supervised_scale <= 0:
            raise ValueError("supervised_scale must be positive")
        valid_reductions = (
            "voxel_mean",
            "relative_per_sample",
            "energy_capped_per_sample",
        )
        if self.l2_reduction not in valid_reductions:
            raise ValueError(f"l2_reduction must be one of {valid_reductions}")
        if self.relative_l2_epsilon <= 0:
            raise ValueError("relative_l2_epsilon must be positive")
        if self.relative_l2_energy_floor <= 0:
            raise ValueError("relative_l2_energy_floor must be positive")
        if self.high_frequency_weight < 0:
            raise ValueError("high_frequency_weight must be nonnegative")
        if self.gradient_reduction not in valid_reductions:
            raise ValueError(f"gradient_reduction must be one of {valid_reductions}")
        _validate_gradient_spacing(self.gradient_spacing_um)
        if self.gradient_energy_floor <= 0:
            raise ValueError("gradient_energy_floor must be positive")
        if self.view_attention_identity_weight < 0:
            raise ValueError("view_attention_identity_weight must be nonnegative")
        if self.view_attention_smoothness_weight < 0:
            raise ValueError(
                "view_attention_smoothness_weight must be nonnegative"
            )
        if self.fp_consistency_weight < 0 or self.fp_flux_weight < 0:
            raise ValueError("FP auxiliary weights must be nonnegative")
        if self.fp_relative_epsilon <= 0:
            raise ValueError("fp_relative_epsilon must be positive")


@dataclass(frozen=True)
class H2FusionLossConfig:
    """Supervise the merged output and optionally H2 itself.

    The connected-network baseline uses only final-output supervision. H2 has
    no separately defined high-frequency target, so its auxiliary term is an
    explicit ablation rather than a default part of the objective.
    """

    final_weight: float = 1.0
    use_axial_l2_weighting: bool = False
    depth_step_um: float = .5
    axial_fwhm_um: float = 40.62
    h2_auxiliary_weight: float = 0.0
    supervised_scale: float = 1.0
    l2_reduction: VolumeL2Reduction = "energy_capped_per_sample"
    relative_l2_epsilon: float = 1e-12
    relative_l2_energy_floor: float = 1e-6
    high_frequency_weight: float = 0.0
    gradient_reduction: VolumeL2Reduction = "energy_capped_per_sample"
    gradient_spacing_um: tuple[float, float, float] = (1.0, 1.0, 1.0)
    gradient_energy_floor: float = 1e-6
    fusion_residual_weight: float = 0.0
    view_attention_identity_weight: float = 0.0
    view_attention_smoothness_weight: float = 0.0
    fp_consistency_weight: float = 0.0
    fp_flux_weight: float = 0.0
    fp_relative_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.final_weight <= 0:
            raise ValueError("final_weight must be positive")
        if any(not math.isfinite(v) or v<=0 for v in (self.depth_step_um,self.axial_fwhm_um)):
            raise ValueError('Axial sampling and FWHM must be finite and positive')
        if self.h2_auxiliary_weight < 0:
            raise ValueError("h2_auxiliary_weight must be nonnegative")
        if self.fusion_residual_weight < 0:
            raise ValueError("fusion_residual_weight must be nonnegative")
        if self.high_frequency_weight < 0:
            raise ValueError("high_frequency_weight must be nonnegative")
        _validate_gradient_spacing(self.gradient_spacing_um)
        if self.gradient_energy_floor <= 0:
            raise ValueError("gradient_energy_floor must be positive")
        if self.view_attention_identity_weight < 0:
            raise ValueError("view_attention_identity_weight must be nonnegative")
        if self.view_attention_smoothness_weight < 0:
            raise ValueError(
                "view_attention_smoothness_weight must be nonnegative"
            )
        if self.fp_consistency_weight < 0 or self.fp_flux_weight < 0:
            raise ValueError("FP auxiliary weights must be nonnegative")
        if self.fp_relative_epsilon <= 0:
            raise ValueError("fp_relative_epsilon must be positive")
        H2LossConfig(
            supervised_scale=self.supervised_scale,
            l2_reduction=self.l2_reduction,
            relative_l2_epsilon=self.relative_l2_epsilon,
            relative_l2_energy_floor=self.relative_l2_energy_floor,
            high_frequency_weight=self.high_frequency_weight,
            gradient_reduction=self.gradient_reduction,
            gradient_spacing_um=self.gradient_spacing_um,
            gradient_energy_floor=self.gradient_energy_floor,
            view_attention_identity_weight=self.view_attention_identity_weight,
            view_attention_smoothness_weight=(
                self.view_attention_smoothness_weight
            ),
            fp_consistency_weight=self.fp_consistency_weight,
            fp_flux_weight=self.fp_flux_weight,
            fp_relative_epsilon=self.fp_relative_epsilon,
        )


def _validate_gradient_spacing(spacing_um: tuple[float, float, float]) -> None:
    if len(spacing_um) != 3:
        raise ValueError("gradient_spacing_um must contain (z, y, x)")
    if any(not math.isfinite(value) or value <= 0 for value in spacing_um):
        raise ValueError("gradient_spacing_um values must be finite and positive")


def volume_gradient_energy(
    value: torch.Tensor,
    *,
    spacing_um: tuple[float, float, float],
) -> torch.Tensor:
    """Return per-sample mean squared physical-gradient energy.

    The last three dimensions are interpreted as ``(z, y, x)``. Averaging
    the three directional energies prevents the numerical scale of this
    diagnostic from changing merely because another spatial axis is added.
    """
    if value.ndim < 4:
        raise ValueError(
            "A batched volume with at least four dimensions is required, got "
            f"{tuple(value.shape)}"
        )
    _validate_gradient_spacing(spacing_um)
    if any(size < 2 for size in value.shape[-3:]):
        raise ValueError(
            "Every spatial volume dimension must contain at least two samples"
        )

    volume = value.float()
    directional_energies = []
    for axis, spacing in zip((-3, -2, -1), spacing_um):
        derivative = torch.diff(volume, dim=axis) / spacing
        directional_energies.append(
            derivative.square().flatten(1).mean(dim=1)
        )
    return torch.stack(directional_energies, dim=0).mean(dim=0)


def volume_gradient_l2_reduction(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    spacing_um: tuple[float, float, float],
    reduction: VolumeL2Reduction,
    relative_epsilon: float,
    energy_floor: float,
) -> torch.Tensor:
    """Compare physical volume gradients with the selected L2 reduction.

    Unlike a Fourier-magnitude loss, this retains spatial phase: a sharp line
    reconstructed at the wrong voxel still incurs an error. For relative
    reductions, normalization by target-gradient energy makes the term nearly
    invariant to a common intensity gain. The capped form prevents dim or
    nearly uniform samples from receiving an unbounded weight.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            "Prediction and target shapes differ: "
            f"{prediction.shape} vs {target.shape}"
        )
    error_energy = volume_gradient_energy(
        prediction - target,
        spacing_um=spacing_um,
    )
    if reduction == "voxel_mean":
        return error_energy.mean()
    target_energy = volume_gradient_energy(target, spacing_um=spacing_um)
    denominator = (
        energy_floor if reduction == "energy_capped_per_sample" else relative_epsilon
    )
    return (error_energy / target_energy.clamp_min(denominator)).mean()


def view_attention_regularization(
    attention_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return identity and adjacent-angular smoothness penalties.

    The view channels are interpreted as a square angular grid in the same
    row-major order used by ``LFToViews``.  These are soft penalties; the hard
    positivity, mean-one, and range constraints belong to the attention
    module itself.
    """
    if attention_weights.ndim != 4 or attention_weights.shape[-2:] != (1, 1):
        raise ValueError(
            "attention_weights must have shape [B, V, 1, 1], got "
            f"{tuple(attention_weights.shape)}"
        )
    num_views = attention_weights.shape[1]
    nnum = int(num_views**0.5)
    if nnum * nnum != num_views:
        raise ValueError(f"View count {num_views} is not a square angular grid")

    identity = (attention_weights - 1.0).square().mean()
    if nnum == 1:
        return identity, identity * 0.0
    grid = attention_weights.reshape(-1, nnum, nnum)
    vertical = (grid[:, 1:] - grid[:, :-1]).square().mean()
    horizontal = (grid[:, :, 1:] - grid[:, :, :-1]).square().mean()
    return identity, 0.5 * (vertical + horizontal)


def _view_attention_components(
    reference: torch.Tensor,
    attention_weights: torch.Tensor | None,
    *,
    identity_weight: float,
    smoothness_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if attention_weights is None:
        if identity_weight > 0 or smoothness_weight > 0:
            raise ValueError(
                "attention_weights are required when an attention penalty is active"
            )
        zero = reference.detach() * 0.0
        return zero, zero
    identity, smoothness = view_attention_regularization(attention_weights)
    return identity_weight * identity, smoothness_weight * smoothness


def forward_projection_regularization(
    predicted_views: torch.Tensor,
    observed_views: torch.Tensor,
    *,
    relative_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return relative view error and log-flux mismatch for the learned FP.

    This supervises H2's internal learned forward projector in its own view
    domain. It is deliberately separate from a physical raw-volume projector
    used for experimental measurement consistency.
    """
    if predicted_views.shape != observed_views.shape:
        raise ValueError(
            "Predicted and observed view shapes differ: "
            f"{predicted_views.shape} vs {observed_views.shape}"
        )
    predicted = predicted_views.float()
    observed = observed_views.float()
    error_energy = (predicted - observed).square().flatten(1).mean(dim=1)
    observed_energy = observed.square().flatten(1).mean(dim=1)
    relative_error = (
        error_energy / observed_energy.clamp_min(relative_epsilon)
    ).mean()

    flux_epsilon = relative_epsilon**0.5
    predicted_flux = predicted.flatten(1).mean(dim=1).clamp_min(0)
    observed_flux = observed.flatten(1).mean(dim=1).clamp_min(0)
    log_flux_error = torch.log(
        (predicted_flux + flux_epsilon)
        / (observed_flux + flux_epsilon)
    ).square().mean()
    return relative_error, log_flux_error


def _forward_projection_components(
    reference: torch.Tensor,
    predicted_views: torch.Tensor | None,
    observed_views: torch.Tensor | None,
    *,
    consistency_weight: float,
    flux_weight: float,
    relative_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if predicted_views is None or observed_views is None:
        if consistency_weight > 0 or flux_weight > 0:
            raise ValueError(
                "predicted_views and observed_views are required when an FP "
                "auxiliary is active"
            )
        zero = reference.detach() * 0.0
        return zero, zero
    consistency, flux = forward_projection_regularization(
        predicted_views,
        observed_views,
        relative_epsilon=relative_epsilon,
    )
    return consistency_weight * consistency, flux_weight * flux


def volume_l2_reduction(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: VolumeL2Reduction,
    relative_epsilon: float,
    energy_floor: float,
    axial_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply one of the volume-level L2 reductions used in H2 experiments."""
    if prediction.shape != target.shape:
        raise ValueError(
            "Prediction and target shapes differ: "
            f"{prediction.shape} vs {target.shape}"
        )
    squared_error = (prediction - target).square()
    if axial_weights is not None:
        if axial_weights.shape!=(1,prediction.shape[1],1,1) or not torch.isfinite(axial_weights).all() or axial_weights.min()<=0:
            raise ValueError('Invalid axial loss weights')
        weights=axial_weights.to(prediction).expand_as(prediction)
        denominator_weight=weights.flatten(1).sum(1)
        error_energy=(squared_error*weights).flatten(1).sum(1)/denominator_weight
        if reduction=='voxel_mean':return error_energy.mean()
        target_energy=(target.square()*weights).flatten(1).sum(1)/denominator_weight
        floor=energy_floor if reduction=='energy_capped_per_sample' else relative_epsilon
        return (error_energy/target_energy.clamp_min(floor)).mean()
    if reduction == "voxel_mean":
        return squared_error.mean()
    error_energy = squared_error.flatten(1).mean(dim=1)
    target_energy = target.square().flatten(1).mean(dim=1)
    denominator = (
        energy_floor if reduction == "energy_capped_per_sample" else relative_epsilon
    )
    return (error_energy / target_energy.clamp_min(denominator)).mean()


def normalized_volume_energy(
    value: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: VolumeL2Reduction,
    relative_epsilon: float,
    energy_floor: float,
) -> torch.Tensor:
    """Measure a prediction component using the target's energy normalization."""
    if value.shape != target.shape:
        raise ValueError(
            f"Value and target shapes differ: {value.shape} vs {target.shape}"
        )
    value_energy = value.square().flatten(1).mean(dim=1)
    if reduction == "voxel_mean":
        return value_energy.mean()
    target_energy = target.square().flatten(1).mean(dim=1)
    denominator = (
        energy_floor if reduction == "energy_capped_per_sample" else relative_epsilon
    )
    return (value_energy / target_energy.clamp_min(denominator)).mean()


def h2_loss(
    h2_prediction: torch.Tensor,
    raw_target: torch.Tensor,
    config: H2LossConfig | None = None,
    *,
    attention_weights: torch.Tensor | None = None,
    predicted_views: torch.Tensor | None = None,
    observed_views: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Directly supervise H2 while H1 and fusion remain frozen."""
    config = config or H2LossConfig()
    h2_raw = volume_l2_reduction(
        h2_prediction / config.supervised_scale,
        raw_target / config.supervised_scale,
        reduction=config.l2_reduction,
        relative_epsilon=config.relative_l2_epsilon,
        energy_floor=config.relative_l2_energy_floor,
    )
    if config.high_frequency_weight > 0:
        h2_high_frequency_raw = volume_gradient_l2_reduction(
            h2_prediction / config.supervised_scale,
            raw_target / config.supervised_scale,
            spacing_um=config.gradient_spacing_um,
            reduction=config.gradient_reduction,
            relative_epsilon=config.relative_l2_epsilon,
            energy_floor=config.gradient_energy_floor,
        )
    else:
        h2_high_frequency_raw = h2_raw.detach() * 0.0
    h2_high_frequency = (
        config.high_frequency_weight * h2_high_frequency_raw
    )
    attention_identity, attention_smoothness = _view_attention_components(
        h2_raw,
        attention_weights,
        identity_weight=config.view_attention_identity_weight,
        smoothness_weight=config.view_attention_smoothness_weight,
    )
    fp_consistency, fp_flux = _forward_projection_components(
        h2_raw,
        predicted_views,
        observed_views,
        consistency_weight=config.fp_consistency_weight,
        flux_weight=config.fp_flux_weight,
        relative_epsilon=config.fp_relative_epsilon,
    )
    return {
        "total": (
            h2_raw + h2_high_frequency
            + attention_identity + attention_smoothness
            + fp_consistency + fp_flux
        ),
        "h2_l2": h2_raw,
        "h2_high_frequency": h2_high_frequency,
        "view_attention_identity": attention_identity,
        "view_attention_smoothness": attention_smoothness,
        "fp_consistency": fp_consistency,
        "fp_flux": fp_flux,
    }


def h2_fusion_loss(
    final_prediction: torch.Tensor,
    h2_prediction: torch.Tensor,
    raw_target: torch.Tensor,
    config: H2FusionLossConfig | None = None,
    *,
    fusion_residual: torch.Tensor | None = None,
    attention_weights: torch.Tensor | None = None,
    predicted_views: torch.Tensor | None = None,
    observed_views: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Supervise the final merge and keep H2 responsible for raw detail."""
    config = config or H2FusionLossConfig()
    axial_weights=None
    if config.use_axial_l2_weighting:
        from .forward_model import lf_axial_profile
        from .axial_illumination import axial_error_weights
        axial_weights=axial_error_weights(lf_axial_profile(final_prediction.shape[1],depth_step_um=config.depth_step_um,
                                           fwhm_um=config.axial_fwhm_um,device=final_prediction.device,dtype=final_prediction.dtype))
    scaled_target = raw_target / config.supervised_scale
    final_raw = volume_l2_reduction(
        final_prediction / config.supervised_scale,
        scaled_target,
        reduction=config.l2_reduction,
        relative_epsilon=config.relative_l2_epsilon,
        energy_floor=config.relative_l2_energy_floor,
        axial_weights=axial_weights,
    )
    if config.h2_auxiliary_weight > 0:
        h2_raw = volume_l2_reduction(
            h2_prediction / config.supervised_scale,
            scaled_target,
            reduction=config.l2_reduction,
            relative_epsilon=config.relative_l2_epsilon,
            energy_floor=config.relative_l2_energy_floor,
            axial_weights=axial_weights,
        )
    else:
        h2_raw = final_raw.detach() * 0.0
    if config.high_frequency_weight > 0:
        final_high_frequency_raw = volume_gradient_l2_reduction(
            final_prediction / config.supervised_scale,
            scaled_target,
            spacing_um=config.gradient_spacing_um,
            reduction=config.gradient_reduction,
            relative_epsilon=config.relative_l2_epsilon,
            energy_floor=config.gradient_energy_floor,
        )
        if config.h2_auxiliary_weight > 0:
            h2_high_frequency_raw = volume_gradient_l2_reduction(
                h2_prediction / config.supervised_scale,
                scaled_target,
                spacing_um=config.gradient_spacing_um,
                reduction=config.gradient_reduction,
                relative_epsilon=config.relative_l2_epsilon,
                energy_floor=config.gradient_energy_floor,
            )
        else:
            h2_high_frequency_raw = final_high_frequency_raw.detach() * 0.0
    else:
        final_high_frequency_raw = final_raw.detach() * 0.0
        h2_high_frequency_raw = final_raw.detach() * 0.0
    if config.fusion_residual_weight > 0:
        if fusion_residual is None:
            raise ValueError(
                "fusion_residual is required when fusion_residual_weight is positive"
            )
        residual_raw = normalized_volume_energy(
            fusion_residual / config.supervised_scale,
            scaled_target,
            reduction=config.l2_reduction,
            relative_epsilon=config.relative_l2_epsilon,
            energy_floor=config.relative_l2_energy_floor,
        )
    else:
        residual_raw = final_raw.detach() * 0.0
    final_component = config.final_weight * final_raw
    h2_component = config.h2_auxiliary_weight * h2_raw
    final_high_frequency_component = (
        config.high_frequency_weight * final_high_frequency_raw
    )
    h2_high_frequency_component = (
        config.high_frequency_weight
        * config.h2_auxiliary_weight
        * h2_high_frequency_raw
    )
    residual_component = config.fusion_residual_weight * residual_raw
    attention_identity, attention_smoothness = _view_attention_components(
        final_raw,
        attention_weights,
        identity_weight=config.view_attention_identity_weight,
        smoothness_weight=config.view_attention_smoothness_weight,
    )
    fp_consistency, fp_flux = _forward_projection_components(
        final_raw,
        predicted_views,
        observed_views,
        consistency_weight=config.fp_consistency_weight,
        flux_weight=config.fp_flux_weight,
        relative_epsilon=config.fp_relative_epsilon,
    )
    return {
        "total": (
            final_component
            + h2_component
            + final_high_frequency_component
            + h2_high_frequency_component
            + residual_component
            + attention_identity
            + attention_smoothness
            + fp_consistency
            + fp_flux
        ),
        "final_l2": final_component,
        "h2_auxiliary_l2": h2_component,
        "final_high_frequency": final_high_frequency_component,
        "h2_auxiliary_high_frequency": h2_high_frequency_component,
        "fusion_residual": residual_component,
        "view_attention_identity": attention_identity,
        "view_attention_smoothness": attention_smoothness,
        "fp_consistency": fp_consistency,
        "fp_flux": fp_flux,
    }
