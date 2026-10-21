from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from .h1_lite_model import (
    AddEpsilon,
    DepthwiseSeparableResidualBlock,
    ElementwiseAdd,
    ElementwiseDivide,
    ElementwiseMultiply,
)
from .physics_rearrange import LFToViews, SpaceToViewDepth, ViewDepthToSpace


ViewAttentionMode = Literal[
    "legacy_sigmoid",
    "mean_preserving_bounded",
    "psf_energy_prior",
]
RatioClampMode = Literal["hard", "soft"]
H2FPMode = Literal["full_projection", "observed_residual"]


@dataclass(frozen=True)
class H2RLLiteConfig:
    """Configuration for the native-grid H2 Richardson--Lucy update."""

    depth: int = 91
    h2_nnum: int = 11
    fp_channels: int = 32
    fp_blocks: int = 3
    fp_mode: H2FPMode = "full_projection"
    fp_residual_scale: float = 1e-4
    # Optional radiometric ablation for the LF-anchored neural ratio only.
    # H1 and the external calibrated LF operator retain original intensities.
    observed_mean_reference: float | None = None
    view_attention_channels: int = 32
    view_attention_mode: ViewAttentionMode = "legacy_sigmoid"
    view_attention_max_deviation: float = 0.25
    # Equation-13-style PSF energy probabilities in LFToViews row-major
    # order. ``psf_energy_prior`` converts these sum-one probabilities to
    # mean-one multiplicative BP weights. A zero residual scale makes the
    # attention exactly fixed; a positive value enables a learned log-weight
    # residual without a tanh or hard output range.
    view_attention_psf_prior: tuple[float, ...] | None = None
    view_attention_residual_scale: float = 0.0
    view_attention_relative_bound: float | None = None
    view_attention_prior_epsilon: float = 1e-12
    latent_channels: int = 256
    bp_dilations: tuple[int, ...] = (1, 2, 4, 8, 4, 2)
    volume_blocks: int = 3
    ratio_epsilon: float = 1e-5
    ratio_max: float | None = 20.0
    ratio_clamp_mode: RatioClampMode = "hard"
    ratio_soft_clamp_beta: float = 1.0
    use_ratio_instance_norm: bool = True
    instance_norm_epsilon: float = 1e-5
    leaky_relu_slope: float = 0.01
    # Used only by the checkpoint-compatible ``full_projection`` mode.
    fp_output_scale_initial: float = 5e-4
    fp_output_softplus_beta: float = 5.0
    correction_softplus_beta: float = 5.0
    use_additive_detail_path: bool = False
    additive_detail_scale: float = 0.05
    additive_detail_softplus_beta: float = 5.0

    def __post_init__(self) -> None:
        integer_fields = (
            "depth",
            "h2_nnum",
            "fp_channels",
            "fp_blocks",
            "view_attention_channels",
            "latent_channels",
            "volume_blocks",
        )
        for field in integer_fields:
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} must be positive")
        if not self.bp_dilations or any(value <= 0 for value in self.bp_dilations):
            raise ValueError("bp_dilations must contain positive values")
        if self.fp_mode not in ("full_projection", "observed_residual"):
            raise ValueError(
                "fp_mode must be 'full_projection' or 'observed_residual'"
            )
        if self.fp_residual_scale <= 0:
            raise ValueError("fp_residual_scale must be positive")
        if self.observed_mean_reference is not None:
            if self.fp_mode != "observed_residual":
                raise ValueError("observed_mean_reference requires observed_residual FP")
            if not math.isfinite(self.observed_mean_reference) or self.observed_mean_reference <= 0:
                raise ValueError("observed_mean_reference must be finite and positive")
        if self.ratio_epsilon <= 0:
            raise ValueError("ratio_epsilon must be positive")
        if self.ratio_max is not None and self.ratio_max <= 0:
            raise ValueError("ratio_max must be positive when provided")
        if self.ratio_clamp_mode not in ("hard", "soft"):
            raise ValueError("ratio_clamp_mode must be 'hard' or 'soft'")
        if self.ratio_soft_clamp_beta <= 0:
            raise ValueError("ratio_soft_clamp_beta must be positive")
        if self.view_attention_mode not in (
            "legacy_sigmoid",
            "mean_preserving_bounded",
            "psf_energy_prior",
        ):
            raise ValueError(
                "view_attention_mode must be 'legacy_sigmoid' or "
                "'mean_preserving_bounded' or 'psf_energy_prior'"
            )
        if not 0 < self.view_attention_max_deviation < 1:
            raise ValueError(
                "view_attention_max_deviation must be between zero and one"
            )
        if self.view_attention_relative_bound is not None and (
            not math.isfinite(self.view_attention_relative_bound)
            or not 0 < self.view_attention_relative_bound < 1
        ):
            raise ValueError("view_attention_relative_bound must be between 0 and 1")
        if self.view_attention_residual_scale < 0:
            raise ValueError("view_attention_residual_scale must be non-negative")
        if self.view_attention_prior_epsilon <= 0:
            raise ValueError("view_attention_prior_epsilon must be positive")
        if self.view_attention_mode == "psf_energy_prior":
            prior = self.view_attention_psf_prior
            if prior is None or len(prior) != self.num_views:
                raise ValueError(
                    "psf_energy_prior attention requires one non-negative "
                    f"weight per view ({self.num_views} values)"
                )
            prior_tensor = torch.as_tensor(prior, dtype=torch.float64)
            if not torch.isfinite(prior_tensor).all():
                raise ValueError("view_attention_psf_prior must be finite")
            if torch.any(prior_tensor < 0) or not torch.any(prior_tensor > 0):
                raise ValueError(
                    "view_attention_psf_prior must be non-negative with "
                    "at least one positive value"
                )
        if self.fp_output_scale_initial <= 0:
            raise ValueError("fp_output_scale_initial must be positive")
        if self.fp_output_softplus_beta <= 0:
            raise ValueError("fp_output_softplus_beta must be positive")
        if self.correction_softplus_beta <= 0:
            raise ValueError("correction_softplus_beta must be positive")
        if self.additive_detail_scale < 0:
            raise ValueError("additive_detail_scale must be non-negative")
        if self.additive_detail_softplus_beta <= 0:
            raise ValueError("additive_detail_softplus_beta must be positive")

    @property
    def num_views(self) -> int:
        return self.h2_nnum**2

    @property
    def projected_channels(self) -> int:
        return self.depth * self.num_views


def _instance_norm(channels: int, config: H2RLLiteConfig) -> nn.InstanceNorm2d:
    return nn.InstanceNorm2d(
        channels,
        eps=config.instance_norm_epsilon,
        affine=True,
        track_running_stats=False,
    )


class DualPoolViewAttention(nn.Module):
    """View attention driven by both mean and maximum ratio responses.

    ``mean_preserving_bounded`` produces positive weights with an exact view
    mean of one and a configurable maximum deviation from one.  It therefore
    cannot alter the global RL ratio level or switch individual views almost
    completely off/on.  ``legacy_sigmoid`` retains the original [0, 2]
    parameterization for existing checkpoints. ``psf_energy_prior`` starts
    from a fixed Equation-13-style PSF-energy map. It optionally learns an
    unconstrained log-weight residual and converts the result to positive,
    exactly mean-one weights with a softmax; no tanh is used.
    """

    def __init__(self, config: H2RLLiteConfig) -> None:
        super().__init__()
        self.mode = config.view_attention_mode
        self.max_deviation = config.view_attention_max_deviation
        self.residual_scale = config.view_attention_residual_scale
        self.relative_bound = config.view_attention_relative_bound
        self.prior_epsilon = config.view_attention_prior_epsilon
        if self.mode == "psf_energy_prior":
            prior = torch.as_tensor(
                config.view_attention_psf_prior, dtype=torch.float32
            )
            prior = prior.clamp_min(self.prior_epsilon)
            prior = prior / prior.sum()
        else:
            prior = torch.full(
                (config.num_views,), 1.0 / config.num_views
            )
        self.register_buffer(
            "prior_probabilities",
            prior.reshape(1, config.num_views, 1, 1),
            persistent=False,
        )
        self.squeeze = nn.Conv2d(
            config.num_views,
            config.view_attention_channels,
            kernel_size=1,
        )
        self.activation = nn.ReLU()
        self.excite = nn.Conv2d(
            config.view_attention_channels,
            config.num_views,
            kernel_size=1,
        )

    def attention_components(
        self, ratio: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "psf_energy_prior" and self.residual_scale == 0:
            prior = self.prior_probabilities.to(ratio)
            weights = ratio.shape[1] * prior.expand(ratio.shape[0], -1, -1, -1)
            residual_logits = ratio.new_zeros(
                (ratio.shape[0], ratio.shape[1], 1, 1)
            )
            return weights, residual_logits

        average = F.adaptive_avg_pool2d(ratio, output_size=1)
        maximum = F.adaptive_max_pool2d(ratio, output_size=1)
        attention = self.squeeze(average) + self.squeeze(maximum)
        attention = self.activation(attention)
        logits = self.excite(attention)
        if self.mode == "legacy_sigmoid":
            return 2.0 * torch.sigmoid(logits), logits

        if self.mode == "psf_energy_prior":
            # The centered residual is easier to interpret, although adding a
            # common offset would cancel in softmax. Multiplication by zero
            # gives the strictly fixed-PSF experiment; a later ablation can
            # enable a small residual without changing the parameterization.
            residual_logits = logits - logits.mean(dim=1, keepdim=True)
            prior = self.prior_probabilities.to(logits)
            # Equivalent to normalize(prior * exp(scale * residual)), but
            # softmax subtracts the largest log-weight before exponentiation.
            # Explicit exp can overflow in FP32 for otherwise finite logits.
            if self.relative_bound is None:
                probabilities = torch.softmax(
                    prior.log() + self.residual_scale * residual_logits,
                    dim=1,
                )
            else:
                # PSF-weighted centering gives sum(prior * delta) == 0.
                # Since t and its weighted mean lie in [-1, 1], |delta| <= b.
                # Thus weights/prior_weights stays in [1-b, 1+b], without
                # a post-normalization operation that could break the bound.
                t = torch.tanh(self.residual_scale * residual_logits)
                centered = t - (prior * t).sum(dim=1, keepdim=True)
                probabilities = prior * (1.0 + 0.5 * self.relative_bound * centered)
            return ratio.shape[1] * probabilities, residual_logits

        # tanh(logits) - mean(tanh(logits)) lies in [-2, 2].  Dividing by
        # two keeps the final deviation within +/- max_deviation while the
        # subtraction makes the mean weight exactly one for every sample.
        bounded = torch.tanh(logits)
        centered = 0.5 * (bounded - bounded.mean(dim=1, keepdim=True))
        return 1.0 + self.max_deviation * centered, logits

    def attention_weights(self, ratio: torch.Tensor) -> torch.Tensor:
        weights, _ = self.attention_components(ratio)
        return weights

    def forward(
        self, ratio_feature: torch.Tensor, ratio: torch.Tensor
    ) -> torch.Tensor:
        return ratio_feature * self.attention_weights(ratio)


class H2RLLite(nn.Module):
    """A compact learned RL correction on the full 11 x 11 LF view grid.

    ``full_projection`` executes

    ``S11(x1) -> learned full FP -> y/(FP+eps) -> learned BP -> S11^-1``.

    ``observed_residual`` follows the residual-FP design used by RLN:

    ``views(y) -> learned positive delta -> y/(y+delta+eps) -> BP -> S11^-1``.

    Both modes retain the explicit update ``x2_RL = x1 * c2`` and use
    independently learned, unmatched forward and backward projectors. An
    optional centered-Softplus detail head can add a small signed residual,
    ``x2 = clamp_min(x2_RL + alpha * delta_HF, 0)``. This gives H2 a controlled
    way to create structure where H1 is nearly zero without replacing the RL
    path. Both heads are zero initialized, so a newly constructed H2 is neutral
    in either mode: ``c2 = 1``, ``delta_HF = 0``, and ``x2 = x1``.
    ``use_ratio_instance_norm=False`` preserves the absolute division level
    before BP; the legacy normalized mode remains available for loading and
    reproducing existing checkpoints.
    """

    input_domain = "cutoff"
    output_domain = "raw"

    def __init__(self, config: H2RLLiteConfig | None = None) -> None:
        super().__init__()
        self.config = config or H2RLLiteConfig()
        c = self.config

        self.forward_rearrange = SpaceToViewDepth(c.h2_nnum)
        self.observed_rearrange = LFToViews(c.h2_nnum)

        # The full-projection mode receives depth x view channels from H1.
        # The paper-style residual mode instead receives the observed views
        # and learns only the positive term added to that measurement anchor.
        fp_input_channels = (
            c.projected_channels
            if c.fp_mode == "full_projection"
            else c.num_views
        )
        self.fp_input = nn.Conv2d(
            fp_input_channels, c.fp_channels, kernel_size=1
        )
        self.fp_blocks = nn.ModuleList(
            DepthwiseSeparableResidualBlock(
                c.fp_channels,
                dilation=1 if index % 2 == 0 else 2,
                negative_slope=c.leaky_relu_slope,
            )
            for index in range(c.fp_blocks)
        )
        self.fp_to_views = nn.Conv2d(
            c.fp_channels, c.num_views, kernel_size=3, padding=1
        )
        self.fp_output_positive = nn.Softplus(beta=c.fp_output_softplus_beta)
        if c.fp_mode == "full_projection":
            self.log_fp_output_scale_parameter = nn.Parameter(
                torch.tensor(math.log(c.fp_output_scale_initial))
            )
            self.fp_output_scale_multiply = ElementwiseMultiply()
        else:
            self.register_buffer(
                "fp_residual_scale",
                torch.tensor(c.fp_residual_scale),
                persistent=False,
            )
            self.fp_residual_scale_multiply = ElementwiseMultiply()
            self.fp_residual_add = ElementwiseAdd()

        # Classical RL ratio: (y + eps) / (A(x) + eps).
        self.ratio_numerator_add = AddEpsilon(c.ratio_epsilon)
        self.ratio_denominator_add = AddEpsilon(c.ratio_epsilon)
        self.ratio_divide = ElementwiseDivide()
        self.ratio_norm: nn.Module = (
            _instance_norm(c.num_views, c)
            if c.use_ratio_instance_norm
            else nn.Identity()
        )
        self.view_attention = DualPoolViewAttention(c)

        self.bp_input = nn.Conv2d(c.num_views, c.latent_channels, kernel_size=1)
        self.bp_blocks = nn.ModuleList(
            DepthwiseSeparableResidualBlock(
                c.latent_channels,
                dilation=dilation,
                negative_slope=c.leaky_relu_slope,
            )
            for dilation in c.bp_dilations
        )
        self.bp_output = nn.Conv2d(
            c.latent_channels, c.projected_channels, kernel_size=1
        )
        self.backward_rearrange = ViewDepthToSpace(
            channels=c.depth, block_size=c.h2_nnum
        )

        self.volume_blocks = nn.ModuleList(
            DepthwiseSeparableResidualBlock(
                c.depth,
                dilation=1 if index % 2 == 0 else 2,
                negative_slope=c.leaky_relu_slope,
            )
            for index in range(c.volume_blocks)
        )
        self.correction_head = nn.Conv2d(c.depth, c.depth, kernel_size=1)
        self.correction_positive = nn.Softplus(
            beta=c.correction_softplus_beta
        )
        correction_at_zero = F.softplus(
            torch.tensor(0.0), beta=c.correction_softplus_beta
        )
        self.register_buffer(
            "correction_identity_scale",
            correction_at_zero.reciprocal(),
            persistent=False,
        )
        self.correction_scale_multiply = ElementwiseMultiply()
        self.rl_update_multiply = ElementwiseMultiply()
        if c.use_additive_detail_path:
            self.additive_detail_head = nn.Conv2d(
                c.depth, c.depth, kernel_size=1
            )
            self.additive_detail_positive = nn.Softplus(
                beta=c.additive_detail_softplus_beta
            )
            detail_at_zero = F.softplus(
                torch.tensor(0.0), beta=c.additive_detail_softplus_beta
            )
            self.register_buffer(
                "additive_detail_center",
                detail_at_zero,
                persistent=False,
            )
            self.additive_detail_scale_multiply = ElementwiseMultiply()
            self.additive_detail_add = ElementwiseAdd()

        self._initialize_layers()
        self.enforce_configured_parameter_freezes()

    def enforce_configured_parameter_freezes(self) -> None:
        """Keep the optional attention residual truly fixed when disabled."""
        if (
            self.config.view_attention_mode == "psf_energy_prior"
            and self.config.view_attention_residual_scale == 0
        ):
            self.view_attention.squeeze.requires_grad_(False)
            self.view_attention.excite.requires_grad_(False)

    def _initialize_layers(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.kaiming_normal_(
            self.view_attention.squeeze.weight,
            mode="fan_in",
            nonlinearity="relu",
        )
        nn.init.zeros_(self.view_attention.squeeze.bias)
        if self.config.view_attention_mode in (
            "mean_preserving_bounded",
            "psf_energy_prior",
        ):
            # Constrained attention starts exactly at its intended neutral or
            # physical prior. In PSF mode a later nonzero residual scale then
            # learns deviations from the prior rather than a new map.
            nn.init.zeros_(self.view_attention.excite.weight)
        else:
            nn.init.kaiming_normal_(
                self.view_attention.excite.weight,
                mode="fan_in",
                nonlinearity="relu",
            )
        nn.init.zeros_(self.view_attention.excite.bias)

        # Normalized Softplus(0) = 1: H2 cannot damage trained H1 at epoch zero.
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)
        if self.config.use_additive_detail_path:
            # Centered Softplus(0) = 0: enabling the detail path on an existing
            # checkpoint leaves its prediction unchanged before new training.
            nn.init.zeros_(self.additive_detail_head.weight)
            nn.init.zeros_(self.additive_detail_head.bias)

    def _validate_inputs(self, lf: torch.Tensor, h1: torch.Tensor) -> None:
        if lf.ndim != 4 or lf.shape[1] != 1:
            raise ValueError(f"Expected LF [B, 1, H, W], got {tuple(lf.shape)}")
        expected = (lf.shape[0], self.config.depth, *lf.shape[-2:])
        if tuple(h1.shape) != expected:
            raise ValueError(f"Expected H1 {expected}, got {tuple(h1.shape)}")
        if lf.shape[-2] % self.config.h2_nnum or lf.shape[-1] % self.config.h2_nnum:
            raise ValueError(
                f"LF dimensions {tuple(lf.shape[-2:])} must be divisible by "
                f"h2_nnum={self.config.h2_nnum}"
            )

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def forward(
        self,
        lf: torch.Tensor,
        h1: torch.Tensor,
        *,
        return_intermediates: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        self._validate_inputs(lf, h1)

        observed_views = self.observed_rearrange(lf.clamp_min(0))
        if self.config.observed_mean_reference is not None:
            mean = observed_views.mean((1, 2, 3), keepdim=True)
            # A nonzero frame has identical neural input under any positive
            # global LF scaling. Blank frames remain blank, without division
            # by zero. One scalar preserves angular and spatial contrasts.
            divisor = torch.where(mean > 0, mean, torch.ones_like(mean))
            observed_views = observed_views / divisor * self.config.observed_mean_reference
        if self.config.fp_mode == "full_projection":
            fp_source = self.forward_rearrange(h1)
        else:
            fp_source = observed_views

        forward_features = self.fp_input(fp_source)
        for block in self.fp_blocks:
            forward_features = block(forward_features)
        positive_fp_output = self.fp_output_positive(
            self.fp_to_views(forward_features)
        )
        fp_residual: torch.Tensor | None = None
        if self.config.fp_mode == "full_projection":
            predicted_views = self.fp_output_scale_multiply(
                positive_fp_output,
                self.log_fp_output_scale_parameter.exp(),
            )
        else:
            fp_residual = self.fp_residual_scale_multiply(
                positive_fp_output,
                self.fp_residual_scale.to(positive_fp_output),
            )
            predicted_views = self.fp_residual_add(
                observed_views,
                fp_residual,
            )
        ratio_numerator = self.ratio_numerator_add(observed_views)
        ratio_denominator = self.ratio_denominator_add(predicted_views)
        ratio = self.ratio_divide(ratio_numerator, ratio_denominator)
        if self.config.ratio_max is not None:
            if self.config.ratio_clamp_mode == "hard":
                ratio = ratio.clamp(max=self.config.ratio_max)
            else:
                # Smooth approximation to min(ratio, ratio_max). Unlike the
                # hard clamp, an over-scale ratio keeps a (small) gradient
                # that can correct the learned forward-projector scale.
                ratio = self.config.ratio_max - F.softplus(
                    self.config.ratio_max - ratio,
                    beta=self.config.ratio_soft_clamp_beta,
                )
                ratio = ratio.clamp_min(0)
        ratio_for_backprojection = self.ratio_norm(ratio)
        attention_weights, attention_residual_logits = (
            self.view_attention.attention_components(ratio)
        )
        attended_ratio = ratio_for_backprojection * attention_weights

        latent = self.bp_input(attended_ratio)
        for block in self.bp_blocks:
            latent = block(latent)
        depth_view_features = self.bp_output(latent)
        volume_features = self.backward_rearrange(depth_view_features)
        for block in self.volume_blocks:
            volume_features = block(volume_features)

        correction_logits = self.correction_head(volume_features)
        base_correction = self.correction_positive(correction_logits)
        rl_correction = self.correction_scale_multiply(
            base_correction,
            self.correction_identity_scale.to(base_correction),
        )
        multiplicative_reconstruction = self.rl_update_multiply(
            h1, rl_correction
        )
        additive_detail_logits: torch.Tensor | None = None
        additive_detail_residual: torch.Tensor | None = None
        preclamp_reconstruction = multiplicative_reconstruction
        if self.config.use_additive_detail_path:
            additive_detail_logits = self.additive_detail_head(volume_features)
            centered_detail = self.additive_detail_positive(
                additive_detail_logits
            ) - self.additive_detail_center.to(additive_detail_logits)
            additive_detail_residual = self.additive_detail_scale_multiply(
                centered_detail,
                self.config.additive_detail_scale,
            )
            preclamp_reconstruction = self.additive_detail_add(
                multiplicative_reconstruction,
                additive_detail_residual,
            )
        reconstruction = preclamp_reconstruction.clamp_min(0)

        if not return_intermediates:
            return reconstruction
        result = {
            "h1": h1,
            # ``forward_projected`` is retained as a legacy audit key. In
            # observed-residual mode it is the observed-view FP source.
            "forward_projected": fp_source,
            "fp_source": fp_source,
            "forward_features": forward_features,
            "positive_fp_output": positive_fp_output,
            "predicted_views": predicted_views,
            "observed_views": observed_views,
            "ratio_numerator": ratio_numerator,
            "ratio_denominator": ratio_denominator,
            "ratio": ratio,
            "ratio_for_backprojection": ratio_for_backprojection,
            "attention_weights": attention_weights,
            "attention_residual_logits": attention_residual_logits,
            "attended_ratio": attended_ratio,
            "latent": latent,
            "depth_view_features": depth_view_features,
            "volume_features": volume_features,
            "correction_logits": correction_logits,
            "base_correction": base_correction,
            "rl_correction": rl_correction,
            "multiplicative_reconstruction": multiplicative_reconstruction,
            "preclamp_reconstruction": preclamp_reconstruction,
            "reconstruction": reconstruction,
        }
        if fp_residual is not None:
            result["fp_residual"] = fp_residual
        if additive_detail_residual is not None:
            assert additive_detail_logits is not None
            result["additive_detail_logits"] = additive_detail_logits
            result["additive_detail_residual"] = additive_detail_residual
        return result


def build_h2_rl_lite(config: H2RLLiteConfig | None = None) -> H2RLLite:
    return H2RLLite(config=config)
