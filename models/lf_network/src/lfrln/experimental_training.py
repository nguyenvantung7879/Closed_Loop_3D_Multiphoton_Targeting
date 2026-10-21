from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .paired_experimental import common_band_relative_l2


@dataclass(frozen=True)
class ExperimentalTrainingSources:
    """Independent switches for the three experimental-training signals."""

    synthetic_replay: bool = True
    paired_supervision: bool = True
    unpaired_feedback: bool = True

    def active_names(self) -> tuple[str, ...]:
        names = []
        if self.synthetic_replay:
            names.append("synthetic_replay")
        if self.paired_supervision:
            names.append("paired_supervision")
        if self.unpaired_feedback:
            names.append("unpaired_feedback")
        return tuple(names)

    def validate(self) -> None:
        if not self.active_names():
            raise ValueError("Enable at least one experimental-training source")


@dataclass(frozen=True)
class ExperimentalLossWeights:
    """Loss weights before source-specific update-frequency scaling."""

    synthetic: float = 1.0
    paired: float = 0.10
    paired_h2_auxiliary: float = 0.10
    unpaired_max: float = 0.01
    gain_anchor: float = 1e-3

    def validate(self) -> None:
        values = {
            "synthetic": self.synthetic,
            "paired": self.paired,
            "paired_h2_auxiliary": self.paired_h2_auxiliary,
            "unpaired_max": self.unpaired_max,
            "gain_anchor": self.gain_anchor,
        }
        invalid = {name: value for name, value in values.items() if value < 0}
        if invalid:
            raise ValueError(f"Loss weights must be nonnegative, got {invalid}")


@dataclass(frozen=True)
class ExperimentalUpdateSchedule:
    """How often paired/unpaired batches are mixed with synthetic replay."""

    paired_every_n_steps: int = 2
    unpaired_every_n_steps: int = 4
    paired_ramp_epochs: int = 10
    unpaired_ramp_epochs: int = 10

    def validate(self) -> None:
        values = {
            "paired_every_n_steps": self.paired_every_n_steps,
            "unpaired_every_n_steps": self.unpaired_every_n_steps,
            "paired_ramp_epochs": self.paired_ramp_epochs,
            "unpaired_ramp_epochs": self.unpaired_ramp_epochs,
        }
        invalid = {name: value for name, value in values.items() if value <= 0}
        if invalid:
            raise ValueError(f"Update intervals and ramps must be positive, got {invalid}")


def linear_ramp(maximum: float, epoch: int, ramp_epochs: int) -> float:
    """Linearly increase a nonnegative weight from zero to ``maximum``."""

    if maximum < 0:
        raise ValueError("maximum must be nonnegative")
    if epoch < 0:
        raise ValueError("epoch must be nonnegative")
    if ramp_epochs <= 0:
        raise ValueError("ramp_epochs must be positive")
    return maximum * min(1.0, epoch / ramp_epochs)


def paired_common_band_loss(
    stages: dict[str, torch.Tensor],
    target: torch.Tensor,
    *,
    h2_auxiliary_weight: float = 0.10,
    kernel_size: tuple[int, int, int] = (3, 5, 5),
) -> dict[str, torch.Tensor]:
    """Supervise registered LF/point-scanning pairs in their common band.

    The point-scanning target is not treated as raw-emitter ground truth. The
    same low-pass filter and independent RMS normalization are applied to the
    final prediction, H2 prediction, and target before measuring relative L2.
    """

    if h2_auxiliary_weight < 0:
        raise ValueError("h2_auxiliary_weight must be nonnegative")
    final = common_band_relative_l2(
        stages["merged"], target, kernel_size=kernel_size
    )
    h2 = common_band_relative_l2(stages["h2"], target, kernel_size=kernel_size)
    return {
        "total": final + h2_auxiliary_weight * h2,
        "final": final,
        "h2": h2,
    }


def anscombe_huber_feedback_loss(
    predicted_signal_counts: torch.Tensor,
    raw_observed_counts: torch.Tensor,
    *,
    background_counts: float,
    saturation_count: float = 65535.0,
    huber_beta: float = 0.25,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Count-domain feedback loss for an unpaired experimental LF frame."""

    if predicted_signal_counts.shape != raw_observed_counts.shape:
        raise ValueError(
            "predicted and observed counts must share shape, got "
            f"{tuple(predicted_signal_counts.shape)} and "
            f"{tuple(raw_observed_counts.shape)}"
        )
    if saturation_count <= 0 or huber_beta <= 0 or epsilon <= 0:
        raise ValueError("saturation_count, huber_beta, and epsilon must be positive")
    observed_signal = (raw_observed_counts - background_counts).clamp_min(0)
    mask = (raw_observed_counts < saturation_count).to(predicted_signal_counts.dtype)
    transformed_prediction = 2.0 * torch.sqrt(
        predicted_signal_counts.clamp_min(0) + 3.0 / 8.0
    )
    transformed_observation = 2.0 * torch.sqrt(observed_signal + 3.0 / 8.0)
    pixel_loss = F.smooth_l1_loss(
        transformed_prediction,
        transformed_observation,
        beta=huber_beta,
        reduction="none",
    )
    valid_count = mask.sum().clamp_min(1)
    data_loss = (pixel_loss * mask).sum() / valid_count
    normalization = (transformed_observation.square() * mask).sum() / valid_count
    return data_loss / normalization.clamp_min(epsilon)


def log_gain_anchor_loss(
    current: torch.nn.ParameterDict,
    initial: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Keep learned multiplicative gains near their calibrated initial values."""

    if set(current) != set(initial):
        raise ValueError("Current and initial log-gain keys do not match")
    if not current:
        raise ValueError("At least one log gain is required")
    return torch.stack(
        [(parameter - initial[key]).square() for key, parameter in current.items()]
    ).mean()


def normalized_metric(value: float, baseline: float, *, epsilon: float = 1e-12) -> float:
    """Return a dimensionless validation ratio for checkpoint selection."""

    return float(value) / max(float(baseline), epsilon)
