from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .h1_lite_v2_model import H1LiteV2, H1LiteV2Config
from .synthetic import IncoherentObjectiveFilter, ObjectivePSFConfig


@dataclass(frozen=True)
class H1DualDomainV2Config(H1LiteV2Config):
    """Raw-emitter decoder defaults with a trainable sparse output head."""

    residual_scale: float = 1.0
    coarse_scale_initial: float = 0.01
    output_bias: float = -1.8


class H1DualDomainV2(H1LiteV2):
    """H1 V2 with a raw latent output and a fixed objective-filtered output.

    ``raw`` is the latent emitter estimate consumed by the raw-domain LF
    projector. ``cutoff`` is ``B(raw)`` and is the reconstruction compared
    with ``GT_emPSF`` and returned by the default forward call.
    """

    output_domain = "cutoff"
    supervised_output_domain = "cutoff"
    feedback_output_domain = "raw"

    def __init__(
        self,
        objective_filter: IncoherentObjectiveFilter,
        config: H1LiteV2Config | None = None,
    ) -> None:
        super().__init__(config=config or H1DualDomainV2Config())
        expected_depth = self.config.depth
        if objective_filter.psf.ndim != 3 or objective_filter.psf.shape[0] != expected_depth:
            raise ValueError(
                "Objective PSF must have shape [depth, height, width] with "
                f"depth={expected_depth}; got {tuple(objective_filter.psf.shape)}"
            )
        self.objective_filter = objective_filter
        self.objective_filter.requires_grad_(False)

    @classmethod
    def from_mat(
        cls,
        path: str | Path,
        *,
        variable_name: str = "objPSF",
        config: H1LiteV2Config | None = None,
        objective_config: ObjectivePSFConfig | None = None,
    ) -> "H1DualDomainV2":
        objective_filter = IncoherentObjectiveFilter.from_mat(
            path,
            variable_name=variable_name,
            config=objective_config,
        )
        return cls(objective_filter, config=config)

    def predict_domains(self, lf: torch.Tensor) -> dict[str, torch.Tensor]:
        result = self.forward(lf, return_domains=True)
        if not isinstance(result, dict):  # pragma: no cover - type guard
            raise RuntimeError("Dual-domain forward did not return a mapping")
        return {"raw": result["raw"], "cutoff": result["cutoff"]}

    def forward(
        self,
        lf: torch.Tensor,
        *,
        return_intermediates: bool = False,
        return_domains: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        backbone_result = super().forward(
            lf, return_intermediates=return_intermediates
        )
        if isinstance(backbone_result, dict):
            raw = backbone_result["reconstruction"]
        else:
            raw = backbone_result
        cutoff = self.objective_filter(raw)

        if return_intermediates:
            if not isinstance(backbone_result, dict):  # pragma: no cover
                raise RuntimeError("Expected H1 intermediates")
            return {
                **backbone_result,
                "raw": raw,
                "cutoff": cutoff,
                "raw_reconstruction": raw,
                "reconstruction": cutoff,
            }
        if return_domains:
            return {"raw": raw, "cutoff": cutoff}
        return cutoff


def build_h1_dual_domain_v2(
    objective_filter: IncoherentObjectiveFilter,
    config: H1LiteV2Config | None = None,
) -> H1DualDomainV2:
    return H1DualDomainV2(objective_filter, config=config)
