from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from .h1_lite_v2_model import H1LiteV2, H1LiteV2Config
from .h1_lite_model import ElementwiseAdd, ElementwiseMultiply, ResizeLike


H1DecoderMode = Literal["learned_subpixel", "phase_mixing_resize"]


class ResidualPhaseMixer(torch.nn.Module):
    """Mix neighbouring native-grid samples without creating subpixel phases.

    This is deliberately a depthwise, translation-equivariant operation.  It
    cannot assign separate filters to the four phases of a learned 2x
    PixelShuffle, and it does not normalize away the physical intensity scale.
    The residual convolution starts at zero, so the module initially acts as
    an identity before the final bilinear resize.
    """

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.residual_scale = residual_scale
        self.depthwise = torch.nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.residual_scale_multiply = ElementwiseMultiply()
        self.residual_add = ElementwiseAdd()
        torch.nn.init.zeros_(self.depthwise.weight)
        torch.nn.init.zeros_(self.depthwise.bias)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        residual = self.depthwise(volume)
        scaled_residual = self.residual_scale_multiply(
            residual,
            self.residual_scale,
        )
        return self.residual_add(volume, scaled_residual)


@dataclass(frozen=True)
class H1RLLiteConfig(H1LiteV2Config):
    """Lightweight H1 settings with an explicit multiplicative RL update."""

    use_full_resolution_branch: bool = False
    use_physical_fp_head: bool = True
    fp_output_scale_initial: float = 5e-4
    fp_output_softplus_beta: float = 5.0
    residual_scale: float = 1.0
    coarse_scale_initial: float = 0.1
    output_bias: float = -1.8
    correction_scale_initial: float | None = None
    use_raw_ratio_bp_path: bool = False
    normalized_ratio_residual_initial: float = 0.0
    normalized_ratio_residual_max: float = 0.25
    decoder_mode: H1DecoderMode = "phase_mixing_resize"
    phase_mixer_kernel_size: int = 5
    phase_mixer_residual_scale: float = 0.1
    ratio_denominator_mode: str = "observed_plus_prediction"
    instance_norm_mode: str = "legacy"
    view_attention_psf_prior: tuple[float, ...] | None = None
    view_attention_residual_scale: float = 0.1
    view_attention_relative_bound: float | None = None
    bp_output_initial_scale: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.view_attention_psf_prior is not None:
            prior = torch.as_tensor(self.view_attention_psf_prior)
            if (prior.numel() != self.num_views or not torch.isfinite(prior).all()
                    or (prior < 0).any() or prior.sum() <= 0):
                raise ValueError("H1 PSF prior must have one finite nonnegative entry per H1 view")
        if self.view_attention_relative_bound is not None and (
            not math.isfinite(self.view_attention_relative_bound)
            or not 0 < self.view_attention_relative_bound < 1
        ):
            raise ValueError("view_attention_relative_bound must be between 0 and 1")
        if self.view_attention_residual_scale < 0:
            raise ValueError("H1 view-attention residual scale must be nonnegative")
        if not math.isfinite(self.bp_output_initial_scale) or self.bp_output_initial_scale <= 0:
            raise ValueError("BP output initialization scale must be finite and positive")
        if self.ratio_denominator_mode not in ("observed_plus_prediction", "prediction"):
            raise ValueError("Unknown H1 ratio_denominator_mode")
        if self.ratio_denominator_mode == "prediction" and not self.use_physical_fp_head:
            raise ValueError("FP-only denominator requires a positive FP head")
        if self.instance_norm_mode not in ("legacy", "without_ratio", "none"):
            raise ValueError("Unknown H1 instance_norm_mode")
        if (
            self.correction_scale_initial is not None
            and self.correction_scale_initial <= 0
        ):
            raise ValueError("correction_scale_initial must be positive")
        if self.fp_output_scale_initial <= 0:
            raise ValueError("fp_output_scale_initial must be positive")
        if self.fp_output_softplus_beta <= 0:
            raise ValueError("fp_output_softplus_beta must be positive")
        if self.normalized_ratio_residual_max <= 0:
            raise ValueError("normalized_ratio_residual_max must be positive")
        if abs(self.normalized_ratio_residual_initial) >= (
            self.normalized_ratio_residual_max
        ):
            raise ValueError(
                "abs(normalized_ratio_residual_initial) must be smaller than "
                "normalized_ratio_residual_max"
            )
        if self.decoder_mode not in (
            "learned_subpixel",
            "phase_mixing_resize",
        ):
            raise ValueError(
                "decoder_mode must be 'learned_subpixel' or "
                "'phase_mixing_resize'"
            )
        if (
            self.phase_mixer_kernel_size <= 0
            or self.phase_mixer_kernel_size % 2 == 0
        ):
            raise ValueError("phase_mixer_kernel_size must be a positive odd integer")
        if self.phase_mixer_residual_scale <= 0:
            raise ValueError("phase_mixer_residual_scale must be positive")


class PredictionDenominator(torch.nn.Module):
    """Positive FP-only denominator; keep numerator/epsilon unchanged in ablations."""

    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, observed, predicted):
        return predicted + self.epsilon


def branch_h1_signal_handling(model, *, ratio_denominator_mode=None, instance_norm_mode=None):
    """Copy shared weights exactly; only discarded InstanceNorm affine terms may differ."""
    from dataclasses import replace
    config = replace(model.config,
        ratio_denominator_mode=ratio_denominator_mode or model.config.ratio_denominator_mode,
        instance_norm_mode=instance_norm_mode or model.config.instance_norm_mode)
    branched = H1RLLite(config)
    old, new = model.state_dict(), branched.state_dict()
    allowed = {f"{name}.{key}" for name, module in model.named_modules()
               if isinstance(module, torch.nn.InstanceNorm2d) for key in module.state_dict()}
    removed = set(old) - set(new)
    if removed - allowed or set(new) - set(old):
        raise ValueError("Signal-handling branch must preserve every non-normalization parameter")
    branched.load_state_dict({key: old[key] for key in new}, strict=True)
    return branched, sorted(removed)


class H1RLLite(H1LiteV2):
    """H1 V2 capacity expressed as a positive multiplicative RL correction.

    The native H1 path remains

    ``S5(x0) -> learned FP -> division -> learned BP -> S5^-1``.

    By default, the correction factor is computed only from the learned
    FP/division/BP path.  The former direct full-resolution LF branch is kept
    behind ``use_full_resolution_branch=True`` only for checkpoint-compatible
    ablations.  The final update is explicitly ``x1 = x0 * correction``.

    ``correction`` is a network output, not an independent parameter named
    c-theta.  A learned per-depth scale is initialized to cancel the very small
    constant MATLAB initialization and keep gradients well scaled without
    per-sample output normalization.

    ``decoder_mode='phase_mixing_resize'`` retains the physical inverse
    rearrangement ``S5^-1`` but replaces the later learned 2x PixelShuffle
    with a native-grid depthwise residual mixer and bilinear resize.  This is
    the recommended H1 cutoff decoder because it cannot learn four independent
    high-resolution phases that appear as a lenslet/checkerboard texture.
    """

    output_domain = "cutoff"

    def __init__(self, config: H1RLLiteConfig | None = None) -> None:
        config = config or H1RLLiteConfig()
        super().__init__(config=config)
        self.config = config

        correction_scale = config.correction_scale_initial
        if correction_scale is None:
            correction_scale = 1.0 / config.initial_value
        self.log_correction_scale_parameter = torch.nn.Parameter(
            torch.full((1, config.depth, 1, 1), math.log(correction_scale))
        )
        if config.use_physical_fp_head:
            # The ratio compares predicted and observed views directly, so the
            # last FP layer must preserve a positive physical intensity scale.
            # InstanceNorm/LeakyReLU is intentionally removed from this head.
            self.fp_to_views = torch.nn.Conv2d(
                config.fp_channels,
                config.num_views,
                kernel_size=3,
                padding=1,
            )
            torch.nn.init.xavier_uniform_(self.fp_to_views.weight)
            torch.nn.init.zeros_(self.fp_to_views.bias)
            self.fp_output_positive = torch.nn.Softplus(
                beta=config.fp_output_softplus_beta
            )
            self.log_fp_output_scale_parameter = torch.nn.Parameter(
                torch.tensor(math.log(config.fp_output_scale_initial))
            )
            self.fp_output_scale_multiply = ElementwiseMultiply()
        self.correction_resize = ResizeLike()
        self.coarse_scale_multiply = ElementwiseMultiply()
        self.correction_bias_add = ElementwiseAdd()
        self.correction_positive = torch.nn.Softplus(
            beta=config.output_softplus_beta
        )
        self.correction_scale_multiply = ElementwiseMultiply()
        self.rl_update_multiply = ElementwiseMultiply()
        self.highres_scale_multiply = ElementwiseMultiply()
        self.highres_add = ElementwiseAdd()

        if config.decoder_mode == "phase_mixing_resize":
            # These inherited H1LiteV2 modules create four separately learned
            # 2x subpixel phases.  Remove them entirely so parameter counts and
            # executable graph inspection match the revised H1 architecture.
            del self.subpixel_expand
            del self.subpixel_shuffle
            self.phase_mixer = ResidualPhaseMixer(
                config.depth,
                kernel_size=config.phase_mixer_kernel_size,
                residual_scale=config.phase_mixer_residual_scale,
            )

        if config.use_raw_ratio_bp_path:
            initial_fraction = (
                config.normalized_ratio_residual_initial
                / config.normalized_ratio_residual_max
            )
            initial_logit = math.atanh(initial_fraction)
            self.normalized_ratio_residual_logit = torch.nn.Parameter(
                torch.tensor(initial_logit)
            )
            self.raw_ratio_attention_multiply = ElementwiseMultiply()
            self.normalized_ratio_residual_multiply = ElementwiseMultiply()
            self.ratio_path_add = ElementwiseAdd()

            # The raw-ratio path starts as an exact identity. Attention can
            # subsequently learn view reliability without removing the
            # positive measurement-consistency signal at initialization.
            with torch.no_grad():
                self.view_attention.excite.weight.zero_()
                self.view_attention.excite.bias.zero_()

        if not config.use_full_resolution_branch:
            # H1 is the strict RL-structured cutoff stage.  Removing these
            # registered modules prevents raw LF features from bypassing the
            # FP -> division -> BP path and also removes their parameters.
            for module_name in (
                "lf_stem",
                "lf_blocks",
                "fuse",
                "fuse_activation",
                "refine_blocks",
                "to_depth_residual",
            ):
                delattr(self, module_name)

        if config.ratio_denominator_mode == "prediction":
            self.ratio_denominator_add = PredictionDenominator(config.ratio_epsilon)
        if config.view_attention_psf_prior is not None:
            from .h2_rl_lite_model import DualPoolViewAttention, H2RLLiteConfig
            self.view_attention = DualPoolViewAttention(H2RLLiteConfig(
                depth=config.depth, h2_nnum=config.h1_nnum,
                view_attention_channels=config.view_attention_channels,
                view_attention_mode="psf_energy_prior",
                view_attention_psf_prior=config.view_attention_psf_prior,
                view_attention_residual_scale=config.view_attention_residual_scale,
                view_attention_relative_bound=config.view_attention_relative_bound))
            torch.nn.init.zeros_(self.view_attention.excite.weight)
            torch.nn.init.zeros_(self.view_attention.excite.bias)
            if config.view_attention_residual_scale == 0:
                self.view_attention.requires_grad_(False)
        if config.instance_norm_mode == "without_ratio":
            self.ratio_norm = torch.nn.Identity()
        elif config.instance_norm_mode == "none":
            def remove_norms(module):
                for name, child in list(module.named_children()):
                    if isinstance(child, torch.nn.InstanceNorm2d):
                        setattr(module, name, torch.nn.Identity())
                    else:
                        remove_norms(child)
            remove_norms(self)

        # Initialization only, never a runtime normalization. Loaded checkpoint
        # weights overwrite this initialization. All BP weights remain trainable.
        with torch.no_grad():
            self.bp_output.weight.mul_(config.bp_output_initial_scale)
            if self.bp_output.bias is not None:
                self.bp_output.bias.mul_(config.bp_output_initial_scale)

    def _predict_views(self, forward_features: torch.Tensor) -> torch.Tensor:
        predicted_views = super()._predict_views(forward_features)
        if not self.config.use_physical_fp_head:
            return predicted_views
        positive_views = self.fp_output_positive(predicted_views)
        fp_output_scale = self.log_fp_output_scale_parameter.exp()
        return self.fp_output_scale_multiply(
            positive_views, fp_output_scale
        )

    def _backprojection_ratio_features(
        self,
        normalized_ratio: torch.Tensor,
        ratio: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not self.config.use_raw_ratio_bp_path:
            return super()._backprojection_ratio_features(
                normalized_ratio, ratio
            )

        attention_weights = self.view_attention.attention_weights(ratio)
        raw_attended_ratio = self.raw_ratio_attention_multiply(
            ratio, attention_weights
        )
        normalized_attended_ratio = self.view_attention(
            normalized_ratio, ratio
        )
        residual_scale = (
            self.config.normalized_ratio_residual_max
            * self.normalized_ratio_residual_logit.tanh()
        )
        normalized_ratio_residual = self.normalized_ratio_residual_multiply(
            normalized_attended_ratio, residual_scale
        )
        correction = self.ratio_path_add(
            raw_attended_ratio, normalized_ratio_residual
        )
        return correction, {
            "ratio_attention_weights": attention_weights,
            "raw_attended_ratio": raw_attended_ratio,
            "normalized_attended_ratio": normalized_attended_ratio,
            "normalized_ratio_residual_scale": residual_scale,
            "normalized_ratio_residual": normalized_ratio_residual,
        }

    def forward(
        self, lf: torch.Tensor, *, return_intermediates: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        # Removing the observation anchor/normalization can produce valid ratios
        # above FP16's 65504 limit. Protect the entire BP/decoder and its backward
        # graph; casting only the division output does not protect autocast convs.
        if (self.config.ratio_denominator_mode == "prediction"
                or self.config.instance_norm_mode != "legacy"):
            with torch.autocast(device_type=lf.device.type, enabled=False):
                return self._forward_impl(lf.float(), return_intermediates=return_intermediates)
        return self._forward_impl(lf, return_intermediates=return_intermediates)

    def _forward_impl(
        self, lf: torch.Tensor, *, return_intermediates: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        intermediates = self._forward_to_volume_small(lf)
        initial = intermediates["initial"]
        volume_small = intermediates["volume_small"]

        decoder_intermediates: dict[str, torch.Tensor]
        if self.config.decoder_mode == "learned_subpixel":
            decoded_volume = self.subpixel_shuffle(
                self.subpixel_expand(volume_small)
            )
            decoded_volume = self.correction_resize(decoded_volume, lf)
            decoder_intermediates = {"subpixel_volume": decoded_volume}
        else:
            phase_mixed_volume = self.phase_mixer(volume_small)
            decoded_volume = self.correction_resize(phase_mixed_volume, lf)
            decoder_intermediates = {
                "phase_mixed_volume": phase_mixed_volume,
            }

        coarse_scale = self.log_coarse_scale_parameter.exp()
        correction_logits = self.coarse_scale_multiply(
            coarse_scale, decoded_volume
        )
        optional_intermediates: dict[str, torch.Tensor] = {}
        if self.config.use_full_resolution_branch:
            lf_features = self.lf_stem(lf)
            for block in self.lf_blocks:
                lf_features = block(lf_features)

            fused = self.fuse_activation(
                self.fuse(torch.cat((decoded_volume, lf_features), dim=1))
            )
            for block in self.refine_blocks:
                fused = block(fused)
            highres_residual = self.to_depth_residual(fused)
            scaled_highres_residual = self.highres_scale_multiply(
                highres_residual,
                self.config.residual_scale,
            )
            correction_logits = self.highres_add(
                correction_logits, scaled_highres_residual
            )
            optional_intermediates = {
                "lf_features": lf_features,
                "highres_residual": highres_residual,
            }
        correction_logits = self.correction_bias_add(
            correction_logits, self.output_bias_parameter
        )
        base_correction = self.correction_positive(correction_logits)
        correction_scale = self.log_correction_scale_parameter.exp()
        rl_correction = self.correction_scale_multiply(
            correction_scale, base_correction
        )
        reconstruction = self.rl_update_multiply(initial, rl_correction)

        if not return_intermediates:
            return reconstruction
        return {
            **intermediates,
            **decoder_intermediates,
            "decoded_volume": decoded_volume,
            "coarse_scale": coarse_scale,
            **optional_intermediates,
            "correction_logits": correction_logits,
            "base_correction": base_correction,
            "correction_scale": correction_scale,
            "rl_correction": rl_correction,
            "reconstruction": reconstruction,
        }


def build_h1_rl_lite(config: H1RLLiteConfig | None = None) -> H1RLLite:
    return H1RLLite(config=config)
