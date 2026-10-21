"""Calibration and comparison helpers for simulated LF measurements."""

from __future__ import annotations

import torch


def fit_gain_background(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    nonnegative_gain: bool = True,
    epsilon: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Least-squares align ``prediction`` to ``target`` with one gain and offset.

    The fitted model is ``target ~= gain * prediction + background``.  The fit
    is global over all tensor elements and is computed from detached float64
    values for numerical stability.  This function is intended for diagnostic
    comparisons, not as a differentiable training operation.

    Returns:
        The aligned prediction (on the original device and with the original
        dtype) and a dictionary containing the fitted parameters and comparison
        metrics.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have identical shapes, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if prediction.device != target.device:
        raise ValueError(
            "prediction and target must be on the same device, got "
            f"{prediction.device} and {target.device}"
        )
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("prediction and target must be floating-point tensors")
    if prediction.numel() == 0:
        raise ValueError("prediction and target must not be empty")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")

    p = prediction.detach().reshape(-1).double()
    y = target.detach().reshape(-1).double()
    p_mean = p.mean()
    y_mean = y.mean()
    p_centered = p - p_mean
    y_centered = y - y_mean
    denominator = p_centered.square().sum()

    if denominator <= epsilon:
        gain = torch.zeros((), dtype=p.dtype, device=p.device)
    else:
        gain = torch.dot(p_centered, y_centered) / denominator
        if nonnegative_gain:
            gain = gain.clamp_min(0)
    background = y_mean - gain * p_mean

    aligned64 = gain * p + background
    error = aligned64 - y
    relative_l2 = torch.linalg.vector_norm(error) / torch.linalg.vector_norm(y).clamp_min(
        epsilon
    )
    mae = error.abs().mean()

    aligned_centered = aligned64 - aligned64.mean()
    correlation_denominator = (
        torch.linalg.vector_norm(aligned_centered)
        * torch.linalg.vector_norm(y_centered)
    )
    if correlation_denominator <= epsilon:
        correlation = torch.zeros((), dtype=p.dtype, device=p.device)
    else:
        correlation = torch.dot(aligned_centered, y_centered) / correlation_denominator

    aligned = aligned64.reshape_as(prediction).to(dtype=prediction.dtype)
    metrics = {
        "gain": float(gain),
        "background": float(background),
        "mae": float(mae),
        "relative_l2": float(relative_l2),
        "correlation": float(correlation),
    }
    return aligned, metrics
