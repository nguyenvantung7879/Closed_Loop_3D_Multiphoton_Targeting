from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


TensorMetrics = dict[str, torch.Tensor]


def _batch_flatten(value: torch.Tensor) -> torch.Tensor:
    if value.ndim < 2:
        raise ValueError(
            f"Expected a batched tensor with at least two dimensions, got {value.shape}"
        )
    return value.reshape(value.shape[0], -1)


def _positive_denominator(value: torch.Tensor) -> torch.Tensor:
    if not value.is_floating_point():
        raise TypeError("Diagnostic tensors must use a floating-point dtype")
    return value.clamp_min(torch.finfo(value.dtype).tiny)


def support_recovery_diagnostics(
    candidate: torch.Tensor,
    target: torch.Tensor,
    *,
    support_mask: torch.Tensor | None = None,
    target_threshold_relative: float = 1e-2,
    candidate_threshold_relative: float = 5e-2,
) -> TensorMetrics:
    """Measure whether a reconstruction is present on known target support.

    Inputs may be ``[D,H,W]`` or ``[B,D,H,W]``. ``support_mask`` can restrict
    the audit to one known object family or line diameter. The candidate
    threshold is relative to its full-volume peak, so an almost-zero response
    inside the selected geometry does not count as recovered merely because it
    is the largest value in that small region. ``scale_fitted_relative_l2``
    removes one non-negative scalar gain and is therefore useful for comparing
    cutoff-domain H1 with raw-domain H2/GT.
    """

    if not 0 <= target_threshold_relative < 1:
        raise ValueError("target_threshold_relative must be in [0, 1)")
    if not 0 <= candidate_threshold_relative < 1:
        raise ValueError("candidate_threshold_relative must be in [0, 1)")
    if candidate.ndim == 3:
        candidate = candidate.unsqueeze(0)
    if target.ndim == 3:
        target = target.unsqueeze(0)
    if candidate.ndim != 4 or candidate.shape != target.shape:
        raise ValueError(
            "Candidate and target must have matching [D,H,W] or [B,D,H,W] "
            f"shapes; got {tuple(candidate.shape)} and {tuple(target.shape)}"
        )
    if not candidate.is_floating_point() or not target.is_floating_point():
        raise TypeError("Candidate and target must be floating-point tensors")

    if support_mask is None:
        analysis_mask = torch.ones_like(target, dtype=torch.bool)
    else:
        if support_mask.ndim == 3:
            support_mask = support_mask.unsqueeze(0)
        try:
            analysis_mask = torch.broadcast_to(
                support_mask.to(device=target.device, dtype=torch.bool),
                target.shape,
            )
        except RuntimeError as error:
            raise ValueError(
                "support_mask must be broadcastable to the volume shape"
            ) from error

    candidate_flat = _batch_flatten(candidate.clamp_min(0))
    target_flat = _batch_flatten(target.clamp_min(0))
    analysis_flat = _batch_flatten(analysis_mask)
    masked_target = torch.where(analysis_mask, target.clamp_min(0), 0)
    target_peak = _batch_flatten(masked_target).amax(dim=1, keepdim=True)
    candidate_peak = candidate_flat.amax(dim=1, keepdim=True)
    target_support = analysis_flat & (
        target_flat > target_threshold_relative * target_peak
    )
    candidate_support = (
        candidate_flat > candidate_threshold_relative * candidate_peak
    )
    recovered_support = target_support & candidate_support

    target_support_count = target_support.sum(dim=1)
    support_recall = (
        recovered_support.sum(dim=1).to(target.dtype)
        / target_support_count.clamp_min(1).to(target.dtype)
    )
    target_energy_on_support = torch.where(
        target_support, target_flat.square(), 0
    )
    recovered_target_energy = torch.where(
        recovered_support, target_flat.square(), 0
    )
    energy_recall = (
        recovered_target_energy.sum(dim=1)
        / _positive_denominator(target_energy_on_support.sum(dim=1))
    )

    analysis_weight = analysis_flat.to(target.dtype)
    target_energy = (analysis_weight * target_flat.square()).sum(dim=1)
    error_energy = (
        analysis_weight * (candidate_flat - target_flat).square()
    ).sum(dim=1)
    candidate_energy = (
        analysis_weight * candidate_flat.square()
    ).sum(dim=1)
    fitted_gain = (
        (analysis_weight * candidate_flat * target_flat).sum(dim=1)
        / _positive_denominator(candidate_energy)
    ).clamp_min(0)
    fitted_error = (
        analysis_weight
        * (fitted_gain[:, None] * candidate_flat - target_flat).square()
    ).sum(dim=1)
    target_denominator = _positive_denominator(target_energy)
    return {
        "target_support_voxels": target_support_count.to(target.dtype),
        "support_voxel_recall": support_recall,
        "target_energy_recall": energy_recall,
        "masked_relative_l2": (error_energy / target_denominator).sqrt(),
        "scale_fitted_relative_l2": (
            fitted_error / target_denominator
        ).sqrt(),
        "fitted_nonnegative_gain": fitted_gain,
    }


def lenslet_phase_artifact_diagnostics(
    volume: torch.Tensor,
    *,
    pitch: int,
) -> dict[str, torch.Tensor]:
    """Measure image structure locked to a repeated lenslet phase.

    The volume is sum-projected over depth, a local mean is removed, and the
    residual is folded modulo ``pitch``. Averaging the returned normalized
    phase maps over unrelated samples suppresses specimen structure while a
    detector/lenslet-locked pattern remains. Inputs may have shape ``[D,H,W]``
    or ``[B,D,H,W]``.
    """

    if pitch <= 1:
        raise ValueError("pitch must be greater than one")
    unbatched = volume.ndim == 3
    if unbatched:
        volume = volume.unsqueeze(0)
    if volume.ndim != 4:
        raise ValueError(
            "Expected volume [D,H,W] or [B,D,H,W], got "
            f"{tuple(volume.shape)}"
        )
    if not volume.is_floating_point():
        volume = volume.float()

    projection = volume.sum(dim=1)
    local_mean = F.avg_pool2d(
        projection.unsqueeze(1),
        kernel_size=2 * pitch + 1,
        stride=1,
        padding=pitch,
    )[:, 0]
    high_pass = projection - local_mean

    # Excluding one pitch prevents padded edge values from biasing the phase
    # average. Starting one complete pitch inward preserves the phase origin.
    if min(projection.shape[-2:]) <= 3 * pitch:
        raise ValueError(
            "Spatial dimensions must exceed three lenslet pitches; got "
            f"{tuple(projection.shape[-2:])} with pitch={pitch}"
        )
    interior = projection[:, pitch:-pitch, pitch:-pitch]
    high_pass = high_pass[:, pitch:-pitch, pitch:-pitch]
    height = (high_pass.shape[-2] // pitch) * pitch
    width = (high_pass.shape[-1] // pitch) * pitch
    high_pass = high_pass[:, :height, :width]
    interior = interior[:, :height, :width]

    batch = high_pass.shape[0]
    folded = high_pass.reshape(
        batch, height // pitch, pitch, width // pitch, pitch
    )
    phase_map = folded.mean(dim=(1, 3))
    phase_map = phase_map - phase_map.mean(dim=(1, 2), keepdim=True)
    reference_level = interior.abs().mean(dim=(1, 2)).clamp_min(
        torch.finfo(interior.dtype).tiny
    )
    normalized_map = phase_map / reference_level[:, None, None]
    score = normalized_map.square().mean(dim=(1, 2)).sqrt()

    result = {
        "phase_map": phase_map,
        "normalized_phase_map": normalized_map,
        "artifact_score": score,
        "projection_reference_level": reference_level,
    }
    if unbatched:
        return {name: value[0] for name, value in result.items()}
    return result


def h2_ratio_diagnostics(
    h2_intermediates: Mapping[str, torch.Tensor],
    *,
    ratio_epsilon: float | torch.Tensor,
    ratio_max: float | None,
) -> TensorMetrics:
    """Measure H2 forward-projection scale and RL-ratio health per sample.

    ``ratio_saturation_fraction`` is computed from the ratio *before* the
    model's hard clamp. This distinguishes a genuinely healthy ratio from a
    clamped tensor whose maximum merely happens to equal ``ratio_max``.
    Returned tensors have shape ``[batch]`` and remain on the input device.
    """

    missing = {"observed_views", "predicted_views"} - set(h2_intermediates)
    if missing:
        raise KeyError(f"H2 intermediates are missing {sorted(missing)}")
    observed = h2_intermediates["observed_views"]
    predicted = h2_intermediates["predicted_views"]
    if observed.shape != predicted.shape:
        raise ValueError(
            "Observed and predicted views must have identical shapes; got "
            f"{observed.shape} and {predicted.shape}"
        )
    observed_flat = _batch_flatten(observed)
    predicted_flat = _batch_flatten(predicted)
    observed_mean = observed_flat.mean(dim=1)
    predicted_mean = predicted_flat.mean(dim=1)

    epsilon = torch.as_tensor(
        ratio_epsilon,
        device=predicted.device,
        dtype=predicted.dtype,
    )
    if torch.any(epsilon <= 0):
        raise ValueError("ratio_epsilon must be positive")
    if epsilon.ndim == 1 and epsilon.shape[0] == predicted.shape[0]:
        epsilon = epsilon.reshape(
            predicted.shape[0], *((1,) * (predicted.ndim - 1))
        )
    try:
        epsilon_field = torch.broadcast_to(epsilon, predicted.shape)
    except RuntimeError as error:
        raise ValueError(
            "ratio_epsilon must be scalar, one value per batch item, or "
            "broadcastable to the view tensor"
        ) from error
    raw_ratio = (observed + epsilon) / (predicted + epsilon)
    raw_ratio_flat = _batch_flatten(raw_ratio)

    if ratio_max is None:
        saturation = torch.zeros_like(raw_ratio_flat, dtype=torch.bool)
        clamped_ratio = raw_ratio
    else:
        if ratio_max <= 0:
            raise ValueError("ratio_max must be positive when provided")
        saturation = raw_ratio_flat >= ratio_max
        clamped_ratio = raw_ratio.clamp(max=ratio_max)

    metrics: TensorMetrics = {
        "observed_view_mean": observed_mean,
        "predicted_view_mean": predicted_mean,
        "observed_to_predicted_flux_ratio": (
            observed_mean / _positive_denominator(predicted_mean)
        ),
        "epsilon_to_predicted_mean": (
            _batch_flatten(epsilon_field).mean(dim=1)
            / _positive_denominator(predicted_mean)
        ),
        "raw_ratio_mean": raw_ratio_flat.mean(dim=1),
        "raw_ratio_median": raw_ratio_flat.median(dim=1).values,
        "raw_ratio_max": raw_ratio_flat.max(dim=1).values,
        "ratio_saturation_fraction": saturation.float().mean(dim=1),
        "clamped_ratio_mean": _batch_flatten(clamped_ratio).mean(dim=1),
    }
    return metrics


def fusion_residual_diagnostics(
    scaled_residual: torch.Tensor,
    reference: torch.Tensor,
    *,
    residual_weight: float,
    energy_floor: float,
) -> TensorMetrics:
    """Report raw and loss-weighted additive-fusion residual energy.

    ``reference`` is normally the raw GT for synthetic data. For unlabeled
    experimental inference it may instead be the gated merge; callers should
    label that choice because the resulting number has a different meaning.
    """

    if scaled_residual.shape != reference.shape:
        raise ValueError(
            "Fusion residual and reference shapes differ: "
            f"{scaled_residual.shape} vs {reference.shape}"
        )
    if residual_weight < 0:
        raise ValueError("residual_weight must be nonnegative")
    if energy_floor <= 0:
        raise ValueError("energy_floor must be positive")

    residual_flat = _batch_flatten(scaled_residual)
    reference_flat = _batch_flatten(reference)
    residual_energy = residual_flat.square().mean(dim=1)
    reference_energy = reference_flat.square().mean(dim=1)
    relative_energy = residual_energy / reference_energy.clamp_min(energy_floor)
    return {
        "fusion_residual_energy": residual_energy,
        "fusion_reference_energy": reference_energy,
        "fusion_residual_relative_energy_raw": relative_energy,
        "fusion_residual_relative_rms_raw": relative_energy.sqrt(),
        "fusion_residual_relative_energy_weighted": (
            residual_weight * relative_energy
        ),
        "fusion_residual_mean_absolute": residual_flat.abs().mean(dim=1),
        "fusion_residual_max_absolute": residual_flat.abs().max(dim=1).values,
    }


def metrics_to_rows(
    metrics: Mapping[str, torch.Tensor],
    *,
    names: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, float | str]]:
    """Convert per-sample tensor metrics into notebook/CSV-friendly rows."""

    if not metrics:
        return []
    batch_sizes = {
        int(value.shape[0])
        for value in metrics.values()
        if value.ndim == 1
    }
    if len(batch_sizes) != 1 or any(value.ndim != 1 for value in metrics.values()):
        raise ValueError("Every metric must be a one-dimensional batch tensor")
    batch_size = batch_sizes.pop()
    if names is not None and len(names) != batch_size:
        raise ValueError(f"Expected {batch_size} names, got {len(names)}")

    cpu_metrics = {
        key: value.detach().float().cpu() for key, value in metrics.items()
    }
    rows: list[dict[str, float | str]] = []
    for index in range(batch_size):
        row: dict[str, float | str] = {
            key: float(value[index]) for key, value in cpu_metrics.items()
        }
        if names is not None:
            row = {"name": names[index], **row}
        rows.append(row)
    return rows


def compare_volume_regions(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    margin: int = 0,
) -> dict[str, float]:
    """Compare two matching volumes, optionally excluding an XY border."""

    if candidate.shape != reference.shape:
        raise ValueError(
            f"Candidate and reference shapes differ: {candidate.shape} vs "
            f"{reference.shape}"
        )
    if candidate.ndim < 3:
        raise ValueError("Volume tensors must include two spatial dimensions")
    height, width = candidate.shape[-2:]
    if margin < 0 or 2 * margin >= min(height, width):
        raise ValueError(
            f"margin={margin} is invalid for spatial shape {(height, width)}"
        )
    if margin:
        candidate = candidate[..., margin:-margin, margin:-margin]
        reference = reference[..., margin:-margin, margin:-margin]

    difference_norm = (candidate - reference).square().sum().sqrt()
    reference_norm = reference.square().sum().sqrt()
    reference_sum = reference.sum()
    tiny = torch.finfo(reference.dtype).tiny
    return {
        "relative_l2": float(
            (difference_norm / reference_norm.clamp_min(tiny)).detach().cpu()
        ),
        "sum_ratio": float(
            (candidate.sum() / reference_sum.clamp_min(tiny)).detach().cpu()
        ),
        "candidate_max": float(candidate.max().detach().cpu()),
        "reference_max": float(reference.max().detach().cpu()),
    }


def tiled_field_of_view_diagnostics(
    model: nn.Module,
    lf: torch.Tensor,
    *,
    repeats: tuple[int, int] = (2, 2),
    margin: int = 33,
    stage_names: tuple[str, ...] = ("h1", "h2", "merged"),
) -> dict[str, dict[str, float]]:
    """Compare a reconstruction alone with the same LF tiled into a larger FOV.

    The top-left tile is byte-identical to ``lf``. The diagnostic therefore
    exposes dependencies on spatial normalization, global pooling, padding,
    and receptive-field geometry. It is a consistency test, not an accuracy
    metric against ground truth.
    """

    if lf.ndim == 3:
        lf = lf.unsqueeze(0)
    if lf.ndim != 4 or lf.shape[0] != 1 or lf.shape[1] != 1:
        raise ValueError(f"Expected one LF [1, 1, H, W], got {tuple(lf.shape)}")
    repeat_y, repeat_x = repeats
    if repeat_y <= 0 or repeat_x <= 0:
        raise ValueError("repeats must contain positive integers")
    if not hasattr(model, "predict_stages"):
        raise TypeError("model must provide predict_stages(lf)")

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            reference_stages = model.predict_stages(lf)
            tiled_lf = lf.repeat(1, 1, repeat_y, repeat_x)
            tiled_stages = model.predict_stages(tiled_lf)
    finally:
        model.train(was_training)

    height, width = lf.shape[-2:]
    result: dict[str, dict[str, float]] = {}
    for stage in stage_names:
        if stage not in reference_stages or stage not in tiled_stages:
            raise KeyError(f"Stage {stage!r} is missing from model predictions")
        reference = reference_stages[stage]
        candidate = tiled_stages[stage][..., :height, :width]
        if not isinstance(reference, torch.Tensor) or not isinstance(
            candidate, torch.Tensor
        ):
            raise TypeError(f"Stage {stage!r} must be a tensor")
        result[stage] = compare_volume_regions(
            candidate,
            reference,
            margin=margin,
        )
    return result


def full_frame_crop_diagnostics(
    model: nn.Module,
    lf: torch.Tensor,
    *,
    crop_shape: tuple[int, int] = (231, 231),
    lenslet_pitch: int = 11,
    margin: int = 33,
    stage_names: tuple[str, ...] = ("h1", "h2", "merged"),
) -> dict[str, Any]:
    """Compare a full-frame reconstruction with its aligned standalone crop."""

    if lf.ndim == 3:
        lf = lf.unsqueeze(0)
    if lf.ndim != 4 or lf.shape[0] != 1 or lf.shape[1] != 1:
        raise ValueError(f"Expected one LF [1, 1, H, W], got {tuple(lf.shape)}")
    if lenslet_pitch <= 0:
        raise ValueError("lenslet_pitch must be positive")
    crop_height, crop_width = crop_shape
    height, width = lf.shape[-2:]
    if crop_height > height or crop_width > width:
        raise ValueError(
            f"Crop {crop_shape} exceeds full-frame shape {(height, width)}"
        )
    if crop_height % lenslet_pitch or crop_width % lenslet_pitch:
        raise ValueError(
            f"Crop {crop_shape} must be divisible by lenslet_pitch={lenslet_pitch}"
        )
    if not hasattr(model, "predict_stages"):
        raise TypeError("model must provide predict_stages(lf)")

    def aligned_start(size: int, crop: int) -> int:
        centered = (size - crop) / 2
        start = int(round(centered / lenslet_pitch)) * lenslet_pitch
        maximum = size - crop
        return min(max(start, 0), maximum - maximum % lenslet_pitch)

    top = aligned_start(height, crop_height)
    left = aligned_start(width, crop_width)
    cropped_lf = lf[..., top : top + crop_height, left : left + crop_width]

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            full_stages = model.predict_stages(lf)
            crop_stages = model.predict_stages(cropped_lf)
    finally:
        model.train(was_training)

    stage_metrics: dict[str, dict[str, float]] = {}
    for stage in stage_names:
        if stage not in full_stages or stage not in crop_stages:
            raise KeyError(f"Stage {stage!r} is missing from model predictions")
        full_volume = full_stages[stage]
        crop_volume = crop_stages[stage]
        if not isinstance(full_volume, torch.Tensor) or not isinstance(
            crop_volume, torch.Tensor
        ):
            raise TypeError(f"Stage {stage!r} must be a tensor")
        full_window = full_volume[
            ..., top : top + crop_height, left : left + crop_width
        ]
        stage_metrics[stage] = compare_volume_regions(
            full_window,
            crop_volume,
            margin=margin,
        )
    return {
        "crop_origin": (top, left),
        "crop_shape": crop_shape,
        "stages": stage_metrics,
    }


def connected_prediction_diagnostics(
    model: nn.Module,
    lf: torch.Tensor,
    *,
    raw_target: torch.Tensor | None = None,
    residual_reference: torch.Tensor | None = None,
    residual_weight: float = 0.0,
    energy_floor: float = 1e-6,
) -> tuple[dict[str, Any], TensorMetrics]:
    """Run one connected prediction and collect ratio and fusion diagnostics.

    Synthetic callers should pass ``raw_target``. Experimental callers can
    omit it; the fusion residual is then normalized by the pre-residual gated
    merge, making it an inference diagnostic rather than a supervised metric.
    """

    if not hasattr(model, "predict_stages") or not hasattr(model, "h2"):
        raise TypeError("Expected a connected H1H2 model")
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            stages = model.predict_stages(lf, return_intermediates=True)
    finally:
        model.train(was_training)

    h2_intermediates = stages.get("h2_intermediates")
    fusion_intermediates = stages.get("fusion_intermediates")
    if not isinstance(h2_intermediates, Mapping):
        raise TypeError("Connected prediction did not return H2 intermediates")
    if not isinstance(fusion_intermediates, Mapping):
        raise TypeError("Connected prediction did not return fusion intermediates")

    metrics = h2_ratio_diagnostics(
        h2_intermediates,
        ratio_epsilon=model.h2.config.ratio_epsilon,
        ratio_max=model.h2.config.ratio_max,
    )
    reference = raw_target if raw_target is not None else residual_reference
    if reference is None:
        reference = fusion_intermediates["gated_merge"]
    residual_metrics = fusion_residual_diagnostics(
        fusion_intermediates["scaled_residual"],
        reference,
        residual_weight=residual_weight,
        energy_floor=energy_floor,
    )
    metrics.update(residual_metrics)
    return stages, metrics
