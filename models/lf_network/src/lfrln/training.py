from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader

from .engines import run_fusion_epoch, run_h2_epoch, run_h2_fusion_epoch
from .h1_model import H1Config, H1RLN
from .losses_h2 import (
    H2FusionLossConfig,
    H2LossConfig,
    h2_fusion_loss,
    h2_loss,
)
from .optimizers import (
    make_fusion_optimizer,
    make_h1_optimizer,
    make_h2_fusion_optimizer,
    make_h2_optimizer,
    make_joint_optimizer,
    make_time_decay_scheduler,
)


@dataclass(frozen=True)
class H1LossConfig:
    """Loss weights from the 2026-03-09 MATLAB H1 training run."""

    charbonnier_weight: float = 0.1
    l2_weight: float = 1.0
    orthogonal_weight: float = 0.5
    charbonnier_epsilon: float = 1e-3
    depth_step_um: float = 0.5
    axial_fwhm_um: float = 40.62


@dataclass(frozen=True)
class H1PhysicsLossConfig:
    """Sparse-volume, frequency-aware, and LF-feedback loss for H1 V2."""

    charbonnier_weight: float = 1.0
    l2_weight: float = 0.25
    sum_projection_weight: float = 0.25
    generalized_projection_weight: float = 0.25
    gradient_weight: float = 0.1
    feedback_weight: float = 0.05
    foreground_weight: float = 4.0
    supervised_scale: float = 1.0
    l2_reduction: Literal[
        "voxel_mean", "relative_per_sample", "energy_capped_per_sample"
    ] = "relative_per_sample"
    relative_l2_epsilon: float = 1e-12
    relative_l2_energy_floor: float = 1e-6
    charbonnier_epsilon: float = 1e-3
    generalized_projection_power: float = 6.0
    normalize_projection_sums: bool = True
    use_axial_l2_weighting: bool = True
    depth_step_um: float = 0.5
    axial_fwhm_um: float = 40.62
    feedback_domain: Literal["raw", "cutoff"] = "cutoff"
    feedback_scale_mode: Literal["none", "gain", "gain_background"] = (
        "gain_background"
    )
    feedback_charbonnier_weight: float = 1.0
    feedback_l2_weight: float = 0.0
    feedback_gradient_weight: float = 0.1
    feedback_warmup_epochs: int = 5
    feedback_ramp_epochs: int = 20
    feedback_gain_min: float = 0.0
    feedback_gain_max: float = 1e4

    def __post_init__(self) -> None:
        nonnegative = (
            "charbonnier_weight",
            "l2_weight",
            "sum_projection_weight",
            "generalized_projection_weight",
            "gradient_weight",
            "feedback_weight",
            "foreground_weight",
            "feedback_charbonnier_weight",
            "feedback_l2_weight",
            "feedback_gradient_weight",
        )
        if any(getattr(self, name) < 0 for name in nonnegative):
            raise ValueError("Loss weights must be nonnegative")
        if self.charbonnier_epsilon <= 0:
            raise ValueError("charbonnier_epsilon must be positive")
        if self.supervised_scale <= 0:
            raise ValueError("supervised_scale must be positive")
        valid_l2_reductions = (
            "voxel_mean",
            "relative_per_sample",
            "energy_capped_per_sample",
        )
        if self.l2_reduction not in valid_l2_reductions:
            raise ValueError(
                f"l2_reduction must be one of {valid_l2_reductions}"
            )
        if self.relative_l2_epsilon <= 0:
            raise ValueError("relative_l2_epsilon must be positive")
        if self.relative_l2_energy_floor <= 0:
            raise ValueError("relative_l2_energy_floor must be positive")
        if self.generalized_projection_power <= 1:
            raise ValueError("generalized_projection_power must be greater than one")
        if self.feedback_warmup_epochs < 0 or self.feedback_ramp_epochs < 0:
            raise ValueError("Feedback warmup/ramp epochs must be nonnegative")
        if self.feedback_gain_min < 0:
            raise ValueError("feedback_gain_min must be nonnegative")
        if self.feedback_gain_max <= self.feedback_gain_min:
            raise ValueError("feedback_gain_max must exceed feedback_gain_min")


@dataclass(frozen=True)
class H1DualDomainLossConfig(H1PhysicsLossConfig):
    """Supervise raw emitters and ``B(raw)`` while feeding raw into ``A_H``."""

    feedback_domain: Literal["raw", "cutoff"] = "raw"
    cutoff_target_mode: Literal["provided", "from_raw"] = "provided"
    normalize_supervised_by_target_peak: bool = False
    cutoff_supervised_scale: float = 1.0
    raw_supervised_scale: float = 1.0
    raw_weight_mode: Literal["intensity", "balanced_support"] = "intensity"
    raw_charbonnier_weight: float = 0.25
    raw_l2_weight: float = 0.25
    raw_sum_projection_weight: float = 0.0
    raw_gradient_weight: float = 0.05
    raw_foreground_weight: float = 8.0
    raw_support_threshold_relative: float = 1e-3
    raw_balance_cap: float = 1e3

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.feedback_domain != "raw":
            raise ValueError("Dual-domain feedback must use the raw latent volume")
        if self.cutoff_target_mode not in ("provided", "from_raw"):
            raise ValueError("Unsupported cutoff_target_mode")
        if self.raw_weight_mode not in ("intensity", "balanced_support"):
            raise ValueError("Unsupported raw_weight_mode")
        raw_weights = (
            self.raw_charbonnier_weight,
            self.raw_l2_weight,
            self.raw_sum_projection_weight,
            self.raw_gradient_weight,
            self.raw_foreground_weight,
        )
        if any(weight < 0 for weight in raw_weights):
            raise ValueError("Raw-domain loss weights must be nonnegative")
        if self.cutoff_supervised_scale <= 0 or self.raw_supervised_scale <= 0:
            raise ValueError("Fixed supervised scales must be positive")
        if not 0 <= self.raw_support_threshold_relative < 1:
            raise ValueError("raw_support_threshold_relative must be in [0, 1)")
        if self.raw_balance_cap < 1:
            raise ValueError("raw_balance_cap must be at least one")


def axial_l2_weights(
    depth: int,
    *,
    depth_step_um: float = 0.5,
    fwhm_um: float = 40.62,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reproduce ``par.zWeightingDL`` from the MATLAB H1 script."""
    center = (depth - 1) / 2.0
    dz = (torch.arange(depth, device=device, dtype=dtype) - center) * depth_step_um
    distribution = 1.0 / torch.sqrt(
        1.0
        + (
            2.0
            * torch.sqrt(torch.tensor(3.0, device=device, dtype=dtype))
            * dz
            / fwhm_um
        )
        ** 2
    )
    weights = 1.0 - distribution + distribution.min()
    return weights.reshape(1, depth, 1, 1)


def orthogonal_mean_projection_loss(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Sum MSE over XY, XZ, and YZ mean projections."""
    loss_xy = torch.mean((prediction.mean(dim=1) - target.mean(dim=1)) ** 2)
    loss_xz = torch.mean((prediction.mean(dim=3) - target.mean(dim=3)) ** 2)
    loss_yz = torch.mean((prediction.mean(dim=2) - target.mean(dim=2)) ** 2)
    return loss_xy + loss_xz + loss_yz


def orthogonal_sum_projection_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float = 1e-3,
    normalize: bool = True,
) -> torch.Tensor:
    """Compare total signal along XY, XZ, and YZ projection rays."""
    loss = prediction.new_zeros(())
    for dimension in (1, 2, 3):
        predicted_projection = prediction.sum(dim=dimension)
        target_projection = target.sum(dim=dimension)
        if normalize:
            scale = float(prediction.shape[dimension])
            predicted_projection = predicted_projection / scale
            target_projection = target_projection / scale
        error = predicted_projection - target_projection
        loss = loss + torch.mean(torch.sqrt(error.square() + epsilon**2))
    return loss


def orthogonal_generalized_projection_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    power: float = 6.0,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Soft peak-projection loss that retains more support than a hard MIP."""
    loss = prediction.new_zeros(())
    for dimension in (1, 2, 3):
        predicted_projection = (
            prediction.clamp_min(0).pow(power).mean(dim=dimension) + epsilon
        ).pow(1.0 / power)
        target_projection = (
            target.clamp_min(0).pow(power).mean(dim=dimension) + epsilon
        ).pow(1.0 / power)
        loss = loss + torch.mean(torch.abs(predicted_projection - target_projection))
    return loss


def volume_gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float = 1e-3,
) -> torch.Tensor:
    """Charbonnier loss on axial and lateral finite differences."""
    loss = prediction.new_zeros(())
    for dimension in (1, 2, 3):
        error = torch.diff(prediction, dim=dimension) - torch.diff(
            target, dim=dimension
        )
        loss = loss + torch.mean(torch.sqrt(error.square() + epsilon**2))
    return loss


def _foreground_weights(target: torch.Tensor, strength: float) -> torch.Tensor:
    if strength == 0:
        return torch.ones_like(target)
    scale = target.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    return 1.0 + strength * (target.clamp_min(0) / scale)


def _scale_pair_by_target_peak(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    enabled: bool,
    fixed_scale: float,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Express a pair in per-target or fixed, dataset-wide loss units."""
    if enabled:
        peak = target.amax(dim=(1, 2, 3), keepdim=True).detach()
        scale = torch.where(peak > epsilon, peak, torch.ones_like(peak))
    else:
        scale = prediction.new_tensor(fixed_scale)
    return prediction / scale, target / scale


def _balanced_support_weights(
    target: torch.Tensor,
    *,
    threshold_relative: float,
    balance_cap: float,
    epsilon: float,
) -> torch.Tensor:
    """Balance sparse foreground/background mass independently per sample."""
    peak = target.amax(dim=(1, 2, 3), keepdim=True).detach()
    support = target > threshold_relative * peak
    support_count = support.sum(dim=(1, 2, 3), keepdim=True).to(target.dtype)
    total_count = float(target[0].numel())
    background_count = total_count - support_count
    positive_weight = (background_count / support_count.clamp_min(epsilon)).clamp(
        min=1.0, max=balance_cap
    )
    positive_weight = torch.where(
        support_count > 0, positive_weight, torch.ones_like(positive_weight)
    )
    return torch.where(support, positive_weight, torch.ones_like(target))


def _align_feedback_intensity(
    prediction: torch.Tensor,
    target: torch.Tensor,
    config: H1PhysicsLossConfig,
) -> torch.Tensor:
    """Fit global gain/DC without the nonlinear extrema of min-max scaling."""
    if config.feedback_scale_mode == "none":
        return prediction

    batch = prediction.shape[0]
    predicted_flat = prediction.reshape(batch, -1)
    target_flat = target.reshape(batch, -1)
    sum_pp = predicted_flat.square().sum(dim=1, keepdim=True)
    sum_py = (predicted_flat * target_flat).sum(dim=1, keepdim=True)

    if config.feedback_scale_mode == "gain":
        gain = sum_py / sum_pp.clamp_min(1e-12)
        background = torch.zeros_like(gain)
    else:
        count = float(predicted_flat.shape[1])
        sum_p = predicted_flat.sum(dim=1, keepdim=True)
        sum_y = target_flat.sum(dim=1, keepdim=True)
        determinant = (count * sum_pp - sum_p.square()).clamp_min(1e-12)
        gain = (count * sum_py - sum_p * sum_y) / determinant
        background = (sum_y - gain * sum_p) / count

    # Treat calibration as acquisition metadata, not an optimization shortcut.
    gain = gain.detach().clamp(
        min=config.feedback_gain_min, max=config.feedback_gain_max
    )
    background = background.detach()
    aligned = gain * predicted_flat + background
    return aligned.reshape_as(prediction)


def _feedback_multiplier(config: H1PhysicsLossConfig, epoch: int | None) -> float:
    if config.feedback_weight == 0:
        return 0.0
    if epoch is None:
        return 1.0
    completed_after_warmup = epoch - config.feedback_warmup_epochs
    if completed_after_warmup <= 0:
        return 0.0
    if config.feedback_ramp_epochs == 0:
        return 1.0
    return min(1.0, completed_after_warmup / config.feedback_ramp_epochs)


def h1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    config: H1LossConfig | None = None,
) -> dict[str, torch.Tensor]:
    """MATLAB-compatible H1 Charbonnier + weighted-L2 + projection loss."""
    config = config or H1LossConfig()
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {prediction.shape} vs {target.shape}"
        )

    error = prediction - target
    charbonnier = torch.mean(torch.sqrt(error.square() + config.charbonnier_epsilon**2))
    weights = axial_l2_weights(
        prediction.shape[1],
        depth_step_um=config.depth_step_um,
        fwhm_um=config.axial_fwhm_um,
        device=prediction.device,
        dtype=prediction.dtype,
    )
    l2 = torch.mean(weights * error.square())
    orthogonal = orthogonal_mean_projection_loss(prediction, target)
    total = (
        config.charbonnier_weight * charbonnier
        + config.l2_weight * l2
        + config.orthogonal_weight * orthogonal
    )
    return {
        "total": total,
        "charbonnier": config.charbonnier_weight * charbonnier,
        "l2": config.l2_weight * l2,
        "orthogonal": config.orthogonal_weight * orthogonal,
    }


def h1_physics_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    observed_lf: torch.Tensor,
    *,
    forward_model: nn.Module | None,
    config: H1PhysicsLossConfig | None = None,
    epoch: int | None = None,
    feedback_prediction: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """H1 V2 supervised loss plus a fixed, domain-checked LF feedback term.

    ``feedback_prediction`` permits a separate latent volume to be passed to
    the forward model while ``prediction`` remains the supervised output.
    """
    config = config or H1PhysicsLossConfig()
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {prediction.shape} vs {target.shape}"
        )
    if observed_lf.ndim != 4 or observed_lf.shape[1] != 1:
        raise ValueError(f"Expected observed LF [B, 1, H, W], got {observed_lf.shape}")

    supervised_prediction = prediction / config.supervised_scale
    supervised_target = target / config.supervised_scale
    error = supervised_prediction - supervised_target
    zero = error.reshape(-1)[0] * 0.0
    foreground = None
    if config.foreground_weight > 0 and (
        config.charbonnier_weight > 0 or config.l2_weight > 0
    ):
        foreground = _foreground_weights(
            supervised_target, config.foreground_weight
        )

    if config.charbonnier_weight > 0:
        charbonnier_error = torch.sqrt(
            error.square() + config.charbonnier_epsilon**2
        )
        if foreground is None:
            charbonnier_raw = charbonnier_error.mean()
        else:
            charbonnier_raw = (
                (foreground * charbonnier_error).sum()
                / foreground.sum().clamp_min(1.0)
            )
    else:
        charbonnier_raw = zero

    if config.l2_weight > 0:
        squared_error = error.square()
        target_square = supervised_target.square()
        if foreground is None and not config.use_axial_l2_weighting:
            if config.l2_reduction == "voxel_mean":
                l2_raw = squared_error.mean()
            else:
                error_energy = squared_error.flatten(1).mean(dim=1)
                target_energy = target_square.flatten(1).mean(dim=1)
                denominator_floor = (
                    config.relative_l2_energy_floor
                    if config.l2_reduction == "energy_capped_per_sample"
                    else config.relative_l2_epsilon
                )
                l2_raw = (
                    error_energy
                    / target_energy.clamp_min(denominator_floor)
                ).mean()
        else:
            if config.use_axial_l2_weighting:
                axial = axial_l2_weights(
                    supervised_prediction.shape[1],
                    depth_step_um=config.depth_step_um,
                    fwhm_um=config.axial_fwhm_um,
                    device=supervised_prediction.device,
                    dtype=supervised_prediction.dtype,
                )
            else:
                axial = None
            if foreground is None:
                if axial is None:  # pragma: no cover - handled by the fast path
                    l2_raw = squared_error.mean()
                else:
                    l2_weights = axial.expand_as(squared_error)
            else:
                l2_weights = foreground if axial is None else foreground * axial
            if config.l2_reduction == "voxel_mean":
                l2_raw = (l2_weights * squared_error).sum() / (
                    l2_weights.sum().clamp_min(1.0)
                )
            else:
                weight_sum = l2_weights.flatten(1).sum(dim=1).clamp_min(1.0)
                error_energy = (
                    l2_weights * squared_error
                ).flatten(1).sum(dim=1) / weight_sum
                target_energy = (
                    l2_weights * target_square
                ).flatten(1).sum(dim=1) / weight_sum
                denominator_floor = (
                    config.relative_l2_energy_floor
                    if config.l2_reduction == "energy_capped_per_sample"
                    else config.relative_l2_epsilon
                )
                l2_raw = (
                    error_energy
                    / target_energy.clamp_min(denominator_floor)
                ).mean()
    else:
        l2_raw = zero

    if config.sum_projection_weight > 0:
        sum_projection_raw = orthogonal_sum_projection_loss(
            supervised_prediction,
            supervised_target,
            epsilon=config.charbonnier_epsilon,
            normalize=config.normalize_projection_sums,
        )
    else:
        sum_projection_raw = zero
    if config.generalized_projection_weight > 0:
        generalized_projection_raw = orthogonal_generalized_projection_loss(
            supervised_prediction,
            supervised_target,
            power=config.generalized_projection_power,
        )
    else:
        generalized_projection_raw = zero
    if config.gradient_weight > 0:
        gradient_raw = volume_gradient_loss(
            supervised_prediction,
            supervised_target,
            epsilon=config.charbonnier_epsilon,
        )
    else:
        gradient_raw = zero

    components = {
        "charbonnier": config.charbonnier_weight * charbonnier_raw,
        "l2": config.l2_weight * l2_raw,
        "sum_projection": config.sum_projection_weight * sum_projection_raw,
        "generalized_projection": (
            config.generalized_projection_weight * generalized_projection_raw
        ),
        "gradient": config.gradient_weight * gradient_raw,
    }

    ramp = _feedback_multiplier(config, epoch)
    if ramp > 0:
        if forward_model is None:
            raise ValueError("feedback_weight > 0 requires a fixed forward_model")
        projector_domain = getattr(forward_model, "input_domain", None)
        if projector_domain != config.feedback_domain:
            raise ValueError(
                "Feedback-domain mismatch: loss expects "
                f"{config.feedback_domain!r}, projector declares "
                f"{projector_domain!r}"
            )
        feedback_input = (
            prediction if feedback_prediction is None else feedback_prediction
        )
        predicted_lf = forward_model(feedback_input)
        if predicted_lf.shape != observed_lf.shape:
            raise ValueError(
                f"Projected LF shape {predicted_lf.shape} differs from observed "
                f"LF shape {observed_lf.shape}"
            )
        aligned_lf = _align_feedback_intensity(predicted_lf, observed_lf, config)
        lf_error = aligned_lf - observed_lf
        feedback_raw = torch.mean(
            torch.sqrt(lf_error.square() + config.charbonnier_epsilon**2)
        )
        feedback_l2 = torch.mean(lf_error.square())
        feedback_gradient = prediction.new_zeros(())
        for dimension in (2, 3):
            gradient_error = torch.diff(aligned_lf, dim=dimension) - torch.diff(
                observed_lf, dim=dimension
            )
            feedback_gradient = feedback_gradient + torch.mean(
                torch.sqrt(
                    gradient_error.square() + config.charbonnier_epsilon**2
                )
            )
        feedback = config.feedback_weight * ramp * (
            config.feedback_charbonnier_weight * feedback_raw
            + config.feedback_l2_weight * feedback_l2
            + config.feedback_gradient_weight * feedback_gradient
        )
    else:
        feedback = prediction.sum() * 0.0
    components["feedback"] = feedback
    components["total"] = sum(components.values())
    return {"total": components.pop("total"), **components}


def h1_dual_domain_loss(
    raw_prediction: torch.Tensor,
    cutoff_prediction: torch.Tensor,
    raw_target: torch.Tensor,
    cutoff_target: torch.Tensor,
    observed_lf: torch.Tensor,
    *,
    forward_model: nn.Module | None,
    config: H1DualDomainLossConfig | None = None,
    epoch: int | None = None,
) -> dict[str, torch.Tensor]:
    """Joint raw-emitter, cutoff-volume, and raw-to-LF consistency loss."""
    config = config or H1DualDomainLossConfig()
    if raw_prediction.shape != raw_target.shape:
        raise ValueError(
            "Raw prediction and target shapes differ: "
            f"{raw_prediction.shape} vs {raw_target.shape}"
        )
    if cutoff_prediction.shape != cutoff_target.shape:
        raise ValueError(
            "Cutoff prediction and target shapes differ: "
            f"{cutoff_prediction.shape} vs {cutoff_target.shape}"
        )

    cutoff_loss_prediction, cutoff_loss_target = _scale_pair_by_target_peak(
        cutoff_prediction,
        cutoff_target,
        enabled=config.normalize_supervised_by_target_peak,
        fixed_scale=config.cutoff_supervised_scale,
        epsilon=config.charbonnier_epsilon,
    )
    cutoff_losses = h1_physics_loss(
        cutoff_loss_prediction,
        cutoff_loss_target,
        observed_lf,
        forward_model=forward_model,
        config=config,
        epoch=epoch,
        feedback_prediction=raw_prediction,
    )

    raw_loss_prediction, raw_loss_target = _scale_pair_by_target_peak(
        raw_prediction,
        raw_target,
        enabled=config.normalize_supervised_by_target_peak,
        fixed_scale=config.raw_supervised_scale,
        epsilon=config.charbonnier_epsilon,
    )
    raw_error = raw_loss_prediction - raw_loss_target
    if config.raw_weight_mode == "balanced_support":
        raw_foreground = _balanced_support_weights(
            raw_loss_target,
            threshold_relative=config.raw_support_threshold_relative,
            balance_cap=config.raw_balance_cap,
            epsilon=config.charbonnier_epsilon,
        )
    else:
        raw_foreground = _foreground_weights(
            raw_loss_target, config.raw_foreground_weight
        )
    raw_normalizer = raw_foreground.sum().clamp_min(1.0)
    raw_charbonnier = (
        raw_foreground
        * torch.sqrt(raw_error.square() + config.charbonnier_epsilon**2)
    ).sum() / raw_normalizer
    raw_l2 = (raw_foreground * raw_error.square()).sum() / raw_normalizer
    raw_sum_projection = orthogonal_mean_projection_loss(
        raw_loss_prediction, raw_loss_target
    )
    raw_gradient = volume_gradient_loss(
        raw_loss_prediction,
        raw_loss_target,
        epsilon=config.charbonnier_epsilon,
    )

    components: dict[str, torch.Tensor] = {
        f"cutoff_{name}": value
        for name, value in cutoff_losses.items()
        if name not in ("total", "feedback")
    }
    components["raw_charbonnier"] = (
        config.raw_charbonnier_weight * raw_charbonnier
    )
    components["raw_l2"] = config.raw_l2_weight * raw_l2
    components["raw_sum_projection"] = (
        config.raw_sum_projection_weight * raw_sum_projection
    )
    components["raw_gradient"] = config.raw_gradient_weight * raw_gradient
    components["feedback"] = cutoff_losses["feedback"]
    total = sum(components.values())
    return {"total": total, **components}


def run_h1_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    loss_config: H1LossConfig | H1PhysicsLossConfig,
    forward_model: nn.Module | None = None,
    epoch: int | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    gradient_clip_norm: float | None = None,
    max_batches: int | None = None,
    progress: bool = False,
    batch_history: list[dict[str, float | int]] | None = None,
) -> dict[str, float]:
    """Run one epoch and optionally retain detached metrics for every batch."""
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    observations = 0
    dual_domain = isinstance(loss_config, H1DualDomainLossConfig)

    if dual_domain:
        if not callable(getattr(model, "predict_domains", None)):
            raise TypeError(
                "H1DualDomainLossConfig requires a model with predict_domains()"
            )
        supervised_domain = getattr(model, "supervised_output_domain", None)
        feedback_domain = getattr(model, "feedback_output_domain", None)
        if supervised_domain != "cutoff" or feedback_domain != "raw":
            raise ValueError(
                "Dual-domain H1 must expose supervised='cutoff' and "
                "feedback='raw' outputs"
            )

    if forward_model is not None:
        forward_model.eval()
        model_domain = getattr(model, "output_domain", None)
        projector_domain = getattr(forward_model, "input_domain", None)
        if dual_domain:
            if projector_domain != loss_config.feedback_domain:
                raise ValueError(
                    f"Projector input domain {projector_domain!r} does not match "
                    f"dual feedback domain {loss_config.feedback_domain!r}"
                )
        elif isinstance(loss_config, H1PhysicsLossConfig) and model_domain is not None:
            if model_domain != loss_config.feedback_domain:
                raise ValueError(
                    f"Model output domain {model_domain!r} does not match feedback "
                    f"domain {loss_config.feedback_domain!r}"
                )
            if projector_domain != model_domain:
                raise ValueError(
                    f"Projector input domain {projector_domain!r} does not match "
                    f"model output domain {model_domain!r}"
                )
        trainable_forward_parameters = [
            parameter for parameter in forward_model.parameters() if parameter.requires_grad
        ]
        if trainable_forward_parameters:
            raise ValueError("The feedback forward_model must be frozen")

    batches: Any = loader
    if progress:
        from tqdm.auto import tqdm

        batches = tqdm(loader, desc="train" if training else "validation")

    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for batch_index, batch in enumerate(batches):
            if max_batches is not None and batch_index >= max_batches:
                break
            lf = batch["lf"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)
            raw_target = None
            cutoff_target = None
            if dual_domain:
                if "raw" not in batch:
                    raise ValueError(
                        "Dual-domain training requires a 'raw' target in each batch"
                    )
                raw_target = batch["raw"].to(device, non_blocking=True)
                if loss_config.cutoff_target_mode == "from_raw":
                    objective_filter = getattr(model, "objective_filter", None)
                    if objective_filter is None:
                        raise TypeError(
                            "cutoff_target_mode='from_raw' requires the model's "
                            "fixed objective_filter"
                        )
                    with torch.no_grad():
                        cutoff_target = objective_filter(raw_target)
                else:
                    cutoff_target = batch.get("cutoff", batch["gt"]).to(
                        device, non_blocking=True
                    )
            observed_lf = batch.get("lf_feedback", batch["lf"]).to(
                device, non_blocking=True
            )
            if (
                isinstance(loss_config, H1PhysicsLossConfig)
                and _feedback_multiplier(loss_config, epoch) > 0
            ):
                compatibility_key = (
                    "raw_feedback_compatible"
                    if dual_domain
                    else "feedback_compatible"
                )
                if compatibility_key in batch:
                    compatibility = torch.as_tensor(batch[compatibility_key])
                    if not bool(compatibility.all()):
                        raise ValueError(
                            f"This synthetic pair is marked {compatibility_key}=False "
                            "and cannot be used for the configured feedback path"
                        )
            batch_size = lf.shape[0]

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp and device.type == "cuda",
            ):
                if dual_domain:
                    domain_predictions = model.predict_domains(lf)
                    if raw_target is None or cutoff_target is None:  # pragma: no cover
                        raise RuntimeError("Dual-domain targets were not loaded")
                    losses = h1_dual_domain_loss(
                        domain_predictions["raw"],
                        domain_predictions["cutoff"],
                        raw_target,
                        cutoff_target,
                        observed_lf,
                        forward_model=forward_model,
                        config=loss_config,
                        epoch=epoch,
                    )
                else:
                    prediction = model(lf)
                if isinstance(loss_config, H1PhysicsLossConfig) and not dual_domain:
                    losses = h1_physics_loss(
                        prediction,
                        gt,
                        observed_lf,
                        forward_model=forward_model,
                        config=loss_config,
                        epoch=epoch,
                    )
                elif not dual_domain:
                    losses = h1_loss(prediction, gt, loss_config)

            if not torch.isfinite(losses["total"]).all():
                raise FloatingPointError(
                    f"Nonfinite H1 loss at batch {batch_index + 1}; optimizer update refused. "
                    "Restart from a finite source checkpoint; inspect input/ratio ranges and precision.")
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(losses["total"]).backward()
                    if gradient_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm, error_if_nonfinite=True)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses["total"].backward()
                    if gradient_clip_norm is not None:
                        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm, error_if_nonfinite=True)
                    optimizer.step()

            for name, value in losses.items():
                totals.setdefault(name, 0.0)
                totals[name] += float(value.detach()) * batch_size
            if batch_history is not None:
                batch_history.append(
                    {
                        "batch": batch_index + 1,
                        "batch_size": batch_size,
                        **{
                            name: float(value.detach())
                            for name, value in losses.items()
                        },
                    }
                )
            observations += batch_size
            if progress:
                batches.set_postfix(total=f"{float(losses['total'].detach()):.5g}")

    if observations == 0:
        raise ValueError("The DataLoader produced no batches")
    return {name: value / observations for name, value in totals.items()}


def save_h1_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    epoch: int,
    history: list[dict[str, Any]],
    loss_config: H1LossConfig | H1PhysicsLossConfig,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    if any(not value.isfinite().all() for value in model.state_dict().values()):
        raise FloatingPointError("Refusing to save an H1 checkpoint with nonfinite model state")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "model_type": type(model).__name__,
        "model_config": asdict(model.config),
        "loss_config": asdict(loss_config),
        "history": history,
    }
    objective_filter = getattr(model, "objective_filter", None)
    if objective_filter is not None and hasattr(objective_filter, "config"):
        payload["objective_filter_config"] = asdict(objective_filter.config)
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    torch.save(payload, path)


def load_h1_training_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str | None = None,
    map_location: Any | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Restore an H1 training checkpoint.

    ``device`` is the original project argument. ``map_location`` is accepted
    as a PyTorch-style alias so notebook and library callers can use either
    spelling. The reconstructed model remains on CPU until the caller moves it
    explicitly with ``model.to(...)``.
    """
    if device is not None and map_location is not None:
        raise ValueError("Pass only one of 'device' or 'map_location'")
    checkpoint_location = map_location if map_location is not None else device
    if checkpoint_location is None:
        checkpoint_location = "cpu"
    payload = torch.load(
        path,
        map_location=checkpoint_location,
        weights_only=False,
    )
    model_type = payload.get("model_type", "H1RLN")
    if any(not value.isfinite().all() for value in payload["model_state"].values()):
        raise FloatingPointError("H1 checkpoint contains nonfinite weights; restart from a finite source checkpoint")
    if model_type == "H1RLN":
        model: nn.Module = H1RLN(H1Config(**payload["model_config"]))
    elif model_type == "H1Lite":
        from .h1_lite_model import H1Lite, H1LiteConfig

        model = H1Lite(H1LiteConfig(**payload["model_config"]))
    elif model_type == "H1LiteV2":
        from .h1_lite_v2_model import H1LiteV2, H1LiteV2Config

        model = H1LiteV2(H1LiteV2Config(**payload["model_config"]))
    elif model_type == "H1RLLite":
        from .h1_rl_lite_model import H1RLLite, H1RLLiteConfig

        model_config = dict(payload["model_config"])
        # H1RLLite checkpoints created before the strict RL revision always
        # contained the direct full-resolution LF branch.
        model_config.setdefault("use_full_resolution_branch", True)
        # Older H1RLLite checkpoints used the normalized/leaky FP head. Keep
        # that architecture when loading them, while new models default to the
        # positive physically scaled projection head.
        model_config.setdefault("use_physical_fp_head", False)
        # Checkpoints predating the ratio-preserving H1 ablation used only the
        # normalized ratio. Preserve their exact executable architecture.
        model_config.setdefault("use_raw_ratio_bp_path", False)
        # Historical H1RLLite checkpoints used the learned 2x PixelShuffle.
        # The revised phase-mixing decoder has a different state dictionary,
        # so missing metadata must select the historical architecture.
        model_config.setdefault("decoder_mode", "learned_subpixel")
        model = H1RLLite(H1RLLiteConfig(**model_config))
    elif model_type == "H1DualDomainV2":
        from .h1_dual_domain_model import H1DualDomainV2, H1DualDomainV2Config
        from .synthetic import IncoherentObjectiveFilter, ObjectivePSFConfig

        psf_key = "objective_filter.psf"
        if psf_key not in payload["model_state"]:
            raise ValueError("Dual-domain checkpoint is missing its objective PSF")
        objective_config = ObjectivePSFConfig(
            **payload.get("objective_filter_config", {})
        )
        objective_filter = IncoherentObjectiveFilter(
            payload["model_state"][psf_key], objective_config
        )
        model = H1DualDomainV2(
            objective_filter,
            H1DualDomainV2Config(**payload["model_config"]),
        )
    else:
        raise ValueError(f"Unsupported checkpoint model_type: {model_type}")
    model.load_state_dict(payload["model_state"])
    return model, payload


def save_h1_h2_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    epoch: int,
    history: list[dict[str, Any]],
    loss_config: H2LossConfig | H2FusionLossConfig,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    """Save the connected H1, H2, and fusion modules as one checkpoint."""
    for name in ("h1", "h2", "fusion"):
        module = getattr(model, name, None)
        if module is None or not hasattr(module, "config"):
            raise TypeError(f"Connected model is missing configured {name!r}")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": int(epoch),
        "model_type": type(model).__name__,
        "model_state": model.state_dict(),
        "h1_config": asdict(model.h1.config),
        "h2_config": asdict(model.h2.config),
        "fusion_config": asdict(model.fusion.config),
        "active_stage": getattr(model, "active_stage", "h2_fusion"),
        "loss_config": asdict(loss_config),
        "history": history,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    torch.save(payload, destination)


def load_h1_h2_training_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str | None = None,
    map_location: Any | None = None,
    h2_config_overrides: Mapping[str, Any] | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Restore a complete connected H1 → H2 → fusion checkpoint.

    ``h2_config_overrides`` is intentionally limited to compatible optional
    paths: the additive detail head and the scale of the PSF-attention
    log-residual. These changes do not alter established FP/BP tensor shapes.
    A new detail head and a zero-scaled attention residual both preserve the
    source prediction exactly.
    """
    if device is not None and map_location is not None:
        raise ValueError("Pass only one of 'device' or 'map_location'")
    checkpoint_location = map_location if map_location is not None else device
    if checkpoint_location is None:
        checkpoint_location = "cpu"
    payload = torch.load(
        Path(path),
        map_location=checkpoint_location,
        weights_only=False,
    )
    if payload.get("model_type") != "H1H2Layerwise":
        raise ValueError(
            "Expected an H1H2Layerwise checkpoint, got "
            f"{payload.get('model_type')!r}"
        )

    from .h1_h2_layerwise_model import (
        H1H2FusionMerge,
        H1H2FusionMergeConfig,
        H1H2Layerwise,
    )
    from .h1_rl_lite_model import H1RLLite, H1RLLiteConfig
    from .h2_rl_lite_model import H2RLLite, H2RLLiteConfig

    h1_config = dict(payload["h1_config"])
    h1_config.setdefault("use_full_resolution_branch", True)
    h1_config.setdefault("use_physical_fp_head", False)
    h1_config.setdefault("decoder_mode", "learned_subpixel")
    h2_config = dict(payload["h2_config"])
    if h2_config_overrides:
        allowed_overrides = {
            "use_additive_detail_path",
            "additive_detail_scale",
            "additive_detail_softplus_beta",
            "view_attention_residual_scale",
        }
        unsupported = set(h2_config_overrides) - allowed_overrides
        if unsupported:
            raise ValueError(
                "H2 checkpoint overrides are limited to the additive detail "
                f"path; unsupported keys: {sorted(unsupported)}"
            )
        h2_config.update(h2_config_overrides)
    fusion_config = dict(payload["fusion_config"])
    # Checkpoints created before the interpretable H1/H2-only fusion revision
    # always used both the H2-correction input and the additive residual.
    fusion_config.setdefault("include_h2_correction_input", True)
    fusion_config.setdefault("use_additive_residual", True)
    model = H1H2Layerwise(
        h1=H1RLLite(H1RLLiteConfig(**h1_config)),
        h2=H2RLLite(H2RLLiteConfig(**h2_config)),
        fusion=H1H2FusionMerge(
            H1H2FusionMergeConfig(**fusion_config)
        ),
    )
    if h2_config_overrides:
        incompatible = model.load_state_dict(
            payload["model_state"], strict=False
        )
        allowed_missing = {
            "h2.additive_detail_head.weight",
            "h2.additive_detail_head.bias",
        }
        unexpected_missing = set(incompatible.missing_keys) - allowed_missing
        if unexpected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Checkpoint is not compatible with the requested H2 detail "
                "extension; missing="
                f"{sorted(unexpected_missing)}, unexpected="
                f"{sorted(incompatible.unexpected_keys)}"
            )
    else:
        model.load_state_dict(payload["model_state"])
    model.configure_stage(payload.get("active_stage", "h2_fusion"))
    return model, payload
