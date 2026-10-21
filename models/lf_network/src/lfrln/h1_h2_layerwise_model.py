from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from .h1_lite_model import DepthwiseSeparableResidualBlock
from .h1_rl_lite_model import H1RLLite, H1RLLiteConfig
from .h2_rl_lite_model import H2RLLite, H2RLLiteConfig


TrainingStage = Literal["h1", "h2", "fusion", "h2_fusion", "joint"]


@dataclass(frozen=True)
class H1H2FusionMergeConfig:
    """Configuration for the small learned H1/H2 fusion head."""

    depth: int = 91
    channels: int = 64
    blocks: int = 3
    residual_scale: float = 0.1
    residual_softplus_beta: float = 5.0
    leaky_relu_slope: float = 0.01
    include_h2_correction_input: bool = False
    use_additive_residual: bool = False
    fixed_gate: float | None = None

    def __post_init__(self) -> None:
        if self.depth <= 0 or self.channels <= 0 or self.blocks <= 0:
            raise ValueError("depth, channels, and blocks must be positive")
        if self.residual_scale < 0:
            raise ValueError("residual_scale must be non-negative")
        if self.residual_softplus_beta <= 0:
            raise ValueError("residual_softplus_beta must be positive")
        if self.fixed_gate is not None and (
            not math.isfinite(self.fixed_gate) or not 0 <= self.fixed_gate <= 1
        ):
            raise ValueError("fixed_gate must lie in [0, 1]")


class H1H2FusionMerge(nn.Module):
    """Fuse H1 and H2 with a voxel-wise gate.

    The default architecture consumes only the two reconstructions and returns
    their learned convex interpolation. Optional correction input and additive
    residual flags exist solely for loading and auditing earlier experiments.
    Fine-detail recovery belongs to H2's explicit detail path rather than the
    final fusion block.
    """

    input_domain = "cutoff_and_raw"
    output_domain = "raw"

    def __init__(self, config: H1H2FusionMergeConfig | None = None) -> None:
        super().__init__()
        self.config = config or H1H2FusionMergeConfig()
        c = self.config

        input_volume_count = 3 if c.include_h2_correction_input else 2
        self.input_projection = nn.Conv2d(
            input_volume_count * c.depth, c.channels, kernel_size=1
        )
        self.activation = nn.LeakyReLU(c.leaky_relu_slope)
        self.blocks = nn.ModuleList(
            DepthwiseSeparableResidualBlock(
                c.channels,
                dilation=1 if index % 2 == 0 else 2,
                negative_slope=c.leaky_relu_slope,
            )
            for index in range(c.blocks)
        )
        self.gate_head = nn.Conv2d(c.channels, c.depth, kernel_size=1)
        self.residual_head = (
            nn.Conv2d(c.channels, c.depth, kernel_size=1)
            if c.use_additive_residual
            else None
        )
        self.register_buffer(
            "residual_center",
            F.softplus(
                torch.tensor(0.0), beta=c.residual_softplus_beta
            ),
            persistent=False,
        )

        self._initialize_layers()

    def _initialize_layers(self) -> None:
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        for block in self.blocks:
            nn.init.xavier_uniform_(block.depthwise.weight)
            nn.init.zeros_(block.depthwise.bias)
            nn.init.xavier_uniform_(block.pointwise.weight)
            nn.init.zeros_(block.pointwise.bias)

        # At construction: gate=0.5, residual=0.  Since H2 also starts as H1,
        # the complete H1->H2->fusion model is initially exactly H1.
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)
        if self.residual_head is not None:
            nn.init.zeros_(self.residual_head.weight)
            nn.init.zeros_(self.residual_head.bias)

    @torch.no_grad()
    def reset_as_passthrough(
        self,
        source: Literal["h1", "balanced", "h2"] = "balanced",
        *,
        endpoint_probability: float = 0.999,
    ) -> H1H2FusionMerge:
        """Reset learned fusion and choose its initial H1/H2 passthrough.

        Sigmoid gates cannot equal zero or one at a finite bias, so the H1 and
        H2 endpoints use ``endpoint_probability`` as a close approximation.
        The additive residual is reset to exactly zero.
        """
        if source not in ("h1", "balanced", "h2"):
            raise ValueError(f"Unsupported passthrough source: {source}")
        if not 0.5 < endpoint_probability < 1.0:
            raise ValueError("endpoint_probability must be between 0.5 and 1")
        self._initialize_layers()
        if source == "balanced":
            gate_probability = 0.5
        elif source == "h2":
            gate_probability = endpoint_probability
        else:
            gate_probability = 1.0 - endpoint_probability
        gate_bias = torch.logit(
            torch.tensor(
                gate_probability,
                device=self.gate_head.bias.device,
                dtype=self.gate_head.bias.dtype,
            )
        )
        self.gate_head.bias.fill_(gate_bias)
        return self

    def forward(
        self,
        h1: torch.Tensor,
        h2: torch.Tensor,
        h2_correction: torch.Tensor | None = None,
        *,
        return_intermediates: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if h1.shape != h2.shape:
            raise ValueError(
                "H1 and H2 must have identical shapes; got "
                f"{tuple(h1.shape)} and {tuple(h2.shape)}"
            )
        if h1.ndim != 4 or h1.shape[1] != self.config.depth:
            raise ValueError(
                f"Expected volumes [B, {self.config.depth}, H, W], "
                f"got {tuple(h1.shape)}"
            )

        fusion_inputs = [h1, h2]
        if self.config.include_h2_correction_input:
            if h2_correction is None or h2_correction.shape != h1.shape:
                correction_shape = (
                    None if h2_correction is None else tuple(h2_correction.shape)
                )
                raise ValueError(
                    "Legacy fusion requires an H2 correction matching H1/H2; "
                    f"got {correction_shape}"
                )
            fusion_inputs.append(h2_correction)
        features = self.activation(
            self.input_projection(torch.cat(fusion_inputs, dim=1))
        )
        for block in self.blocks:
            features = block(features)

        gate = torch.sigmoid(self.gate_head(features))
        if self.config.fixed_gate is not None:
            gate = torch.full_like(gate, self.config.fixed_gate)
        gated_merge = h1 + gate * (h2 - h1)
        if self.residual_head is None:
            residual_logits = torch.zeros_like(gated_merge)
            scaled_residual = torch.zeros_like(gated_merge)
        else:
            residual_logits = self.residual_head(features)
            residual = F.softplus(
                residual_logits, beta=self.config.residual_softplus_beta
            ) - self.residual_center.to(residual_logits)
            scaled_residual = self.config.residual_scale * residual
        reconstruction = (gated_merge + scaled_residual).clamp_min(0)

        if not return_intermediates:
            return reconstruction
        return {
            "features": features,
            "gate": gate,
            "gated_merge": gated_merge,
            "residual_logits": residual_logits,
            "scaled_residual": scaled_residual,
            "reconstruction": reconstruction,
        }


class H1H2Layerwise(nn.Module):
    """H1, H2, and final fusion assembled as one connected network.

    The recommended second training stage is ``h2_fusion``: H1 still executes
    but is frozen, while H2 and fusion are optimized together from the final
    merged output. A later ``joint`` stage can fine-tune every learned
    H1/H2/fusion parameter end to end while preserving H1's fixed physical
    initialization. The individual ``h2`` and ``fusion`` stages remain
    available for diagnostics and ablation experiments.
    """

    output_domain = "raw"

    def __init__(
        self,
        h1: H1RLLite | None = None,
        h2: H2RLLite | None = None,
        fusion: H1H2FusionMerge | None = None,
    ) -> None:
        super().__init__()
        self.h1 = h1 or H1RLLite()
        depth = self.h1.config.depth
        self.h2 = h2 or H2RLLite(H2RLLiteConfig(depth=depth))
        self.fusion = fusion or H1H2FusionMerge(
            H1H2FusionMergeConfig(depth=depth)
        )
        if self.h2.config.depth != depth or self.fusion.config.depth != depth:
            raise ValueError("H1, H2, and fusion depth must match")
        self.active_stage: TrainingStage = "joint"

    @staticmethod
    def _is_trainable(module: nn.Module) -> bool:
        return any(parameter.requires_grad for parameter in module.parameters())

    @staticmethod
    def _set_trainable(module: nn.Module, value: bool) -> None:
        module.requires_grad_(value)

    def freeze_h1(self) -> H1H2Layerwise:
        self._set_trainable(self.h1, False)
        return self

    def unfreeze_h1(self) -> H1H2Layerwise:
        self._set_trainable(self.h1, True)
        # The H1 initial estimate is a fixed physical initialization.
        self.h1.initial.requires_grad_(False)
        return self

    def freeze_h2(self) -> H1H2Layerwise:
        self._set_trainable(self.h2, False)
        return self

    def unfreeze_h2(self) -> H1H2Layerwise:
        self._set_trainable(self.h2, True)
        return self

    def freeze_fusion(self) -> H1H2Layerwise:
        self._set_trainable(self.fusion, False)
        return self

    def unfreeze_fusion(self) -> H1H2Layerwise:
        self._set_trainable(self.fusion, True)
        return self

    def reset_fusion(
        self,
        source: Literal["h1", "balanced", "h2"] = "balanced",
        *,
        endpoint_probability: float = 0.999,
    ) -> H1H2Layerwise:
        """Discard learned fusion weights and restore a chosen passthrough."""
        self.fusion.reset_as_passthrough(
            source, endpoint_probability=endpoint_probability
        )
        return self

    def configure_stage(self, stage: TrainingStage) -> H1H2Layerwise:
        if stage not in ("h1", "h2", "fusion", "h2_fusion", "joint"):
            raise ValueError(f"Unsupported training stage: {stage}")
        self.active_stage = stage
        self._set_trainable(self.h1, stage in ("h1", "joint"))
        self._set_trainable(self.h2, stage in ("h2", "h2_fusion", "joint"))
        if stage in ("h2", "h2_fusion", "joint"):
            self.h2.enforce_configured_parameter_freezes()
        self._set_trainable(
            self.fusion,
            stage in ("fusion", "h2_fusion", "joint"),
        )
        if self.fusion.config.fixed_gate is not None and not self.fusion.config.use_additive_residual:
            self.freeze_fusion()
        if stage in ("h1", "joint"):
            self.h1.initial.requires_grad_(False)
        return self

    def _call_h1(
        self, lf: torch.Tensor, *, return_intermediates: bool
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if self._is_trainable(self.h1):
            return self.h1(lf, return_intermediates=return_intermediates)
        with torch.no_grad():
            return self.h1(lf, return_intermediates=return_intermediates)

    def _call_h2(
        self,
        lf: torch.Tensor,
        h1: torch.Tensor,
        *,
        return_intermediates: bool,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if self._is_trainable(self.h2):
            return self.h2(lf, h1, return_intermediates=return_intermediates)
        with torch.no_grad():
            return self.h2(lf, h1, return_intermediates=return_intermediates)

    def forward_h1(
        self, lf: torch.Tensor, *, return_intermediates: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        return self._call_h1(lf, return_intermediates=return_intermediates)

    def forward_h2(
        self,
        lf: torch.Tensor,
        h1: torch.Tensor | None = None,
        *,
        return_intermediates: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if h1 is None:
            h1_result = self._call_h1(lf, return_intermediates=False)
            assert isinstance(h1_result, torch.Tensor)
            h1 = h1_result
        return self._call_h2(
            lf, h1, return_intermediates=return_intermediates
        )

    def forward_fusion(
        self,
        h1: torch.Tensor,
        h2: torch.Tensor,
        h2_correction: torch.Tensor | None = None,
        *,
        return_intermediates: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        return self.fusion(
            h1,
            h2,
            h2_correction,
            return_intermediates=return_intermediates,
        )

    def predict_stages(
        self,
        lf: torch.Tensor,
        *,
        return_intermediates: bool = False,
        return_loss_terms: bool = False,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        h1_result = self._call_h1(
            lf, return_intermediates=return_intermediates
        )
        h1 = (
            h1_result["reconstruction"]
            if isinstance(h1_result, dict)
            else h1_result
        )
        h2_result = self._call_h2(
            lf, h1, return_intermediates=True
        )
        assert isinstance(h2_result, dict)
        h2 = h2_result["reconstruction"]
        fusion_result = self.fusion(
            h1,
            h2,
            (
                h2_result["rl_correction"]
                if self.fusion.config.include_h2_correction_input
                else None
            ),
            return_intermediates=return_intermediates or return_loss_terms,
        )
        merged = (
            fusion_result["reconstruction"]
            if isinstance(fusion_result, dict)
            else fusion_result
        )
        result: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "h1": h1,
            "h2": h2,
            "merged": merged,
            "h2_correction": h2_result["rl_correction"],
        }
        if "additive_detail_residual" in h2_result:
            result["h2_multiplicative"] = h2_result[
                "multiplicative_reconstruction"
            ]
            result["h2_additive_detail"] = h2_result[
                "additive_detail_residual"
            ]
        if return_intermediates:
            result["h1_intermediates"] = h1_result
            result["h2_intermediates"] = h2_result
            result["fusion_intermediates"] = fusion_result
        if return_loss_terms:
            assert isinstance(fusion_result, dict)
            result["fusion_residual"] = fusion_result["scaled_residual"]
            result["h2_attention_weights"] = h2_result["attention_weights"]
            result["h2_attention_residual_logits"] = h2_result[
                "attention_residual_logits"
            ]
            result["h2_predicted_views"] = h2_result["predicted_views"]
            result["h2_observed_views"] = h2_result["observed_views"]
        return result

    def predict_domains(self, lf: torch.Tensor) -> dict[str, torch.Tensor]:
        stages = self.predict_stages(lf)
        return {
            "cutoff": stages["h1"],
            "raw_h2": stages["h2"],
            "raw": stages["merged"],
        }  # type: ignore[return-value]

    def forward(
        self, lf: torch.Tensor, *, return_stages: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        if self.active_stage == "h1":
            result = self.forward_h1(lf)
            assert isinstance(result, torch.Tensor)
            return result
        if self.active_stage == "h2":
            result = self.forward_h2(lf)
            assert isinstance(result, torch.Tensor)
            return result
        stages = self.predict_stages(lf, return_intermediates=return_stages)
        return stages if return_stages else stages["merged"]


def build_h1_h2_layerwise(
    *,
    h1_config: H1RLLiteConfig | None = None,
    h2_config: H2RLLiteConfig | None = None,
    fusion_config: H1H2FusionMergeConfig | None = None,
) -> H1H2Layerwise:
    h1 = H1RLLite(h1_config)
    depth = h1.config.depth
    h2 = H2RLLite(h2_config or H2RLLiteConfig(depth=depth))
    fusion = H1H2FusionMerge(
        fusion_config or H1H2FusionMergeConfig(depth=depth)
    )
    return H1H2Layerwise(h1=h1, h2=h2, fusion=fusion)
