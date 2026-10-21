from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .h1_lite_model import (
    DepthwiseSeparableResidualBlock,
    H1Lite,
    H1LiteConfig,
)


@dataclass(frozen=True)
class H1LiteV2Config(H1LiteConfig):
    """Higher-resolution decoder settings for the second lightweight H1."""

    subpixel_scale: int = 2
    highres_channels: int = 32
    highres_blocks: int = 2
    residual_scale: float = 0.1
    coarse_scale_initial: float = 0.1
    output_softplus_beta: float = 5.0
    output_bias: float = -0.7

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.subpixel_scale <= 0:
            raise ValueError("subpixel_scale must be positive")
        if self.highres_channels <= 0:
            raise ValueError("highres_channels must be positive")
        if self.highres_blocks <= 0:
            raise ValueError("highres_blocks must be positive")
        if self.residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        if self.coarse_scale_initial <= 0:
            raise ValueError("coarse_scale_initial must be positive")
        if self.output_softplus_beta <= 0:
            raise ValueError("output_softplus_beta must be positive")


class H1LiteV2(H1Lite):
    """
    H1Lite with learned subpixel upsampling and a full-resolution residual path.

    The expensive RL/phase-space computation remains on the native H1 grid.
    A grouped subpixel layer expands each depth independently, while a compact
    branch extracts full-resolution evidence from the recorded LF image. The
    final prediction has no instance normalization, preserving background and
    photometric scale for the forward-projection loss.
    """

    output_domain = "cutoff"

    def __init__(self, config: H1LiteV2Config | None = None) -> None:
        config = config or H1LiteV2Config()
        super().__init__(config=config)
        self.config = config
        c = config

        # The V1 output modules are intentionally replaced. InstanceNorm at
        # the output destroys absolute intensity and raises empty background.
        self.up_refine = nn.Identity()
        self.output_norm = nn.Identity()
        self.output_activation = nn.Identity()

        scale_squared = c.subpixel_scale**2
        self.subpixel_expand = nn.Conv2d(
            c.depth,
            c.depth * scale_squared,
            kernel_size=3,
            padding=1,
            groups=c.depth,
        )
        self.subpixel_shuffle = nn.PixelShuffle(c.subpixel_scale)

        self.lf_stem = nn.Sequential(
            nn.Conv2d(1, c.highres_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(c.leaky_relu_slope),
        )
        self.lf_blocks = nn.ModuleList(
            DepthwiseSeparableResidualBlock(
                c.highres_channels,
                dilation=1 if index % 2 == 0 else 2,
                negative_slope=c.leaky_relu_slope,
            )
            for index in range(c.highres_blocks)
        )
        self.fuse = nn.Conv2d(
            c.depth + c.highres_channels, c.highres_channels, kernel_size=1
        )
        self.fuse_activation = nn.LeakyReLU(c.leaky_relu_slope)
        self.refine_blocks = nn.ModuleList(
            DepthwiseSeparableResidualBlock(
                c.highres_channels,
                dilation=1 if index % 2 == 0 else 2,
                negative_slope=c.leaky_relu_slope,
            )
            for index in range(c.highres_blocks)
        )
        self.to_depth_residual = nn.Conv2d(
            c.highres_channels, c.depth, kernel_size=1
        )
        self.output_bias_parameter = nn.Parameter(
            torch.full((1, c.depth, 1, 1), c.output_bias)
        )
        self.log_coarse_scale_parameter = nn.Parameter(
            torch.full((1, c.depth, 1, 1), math.log(c.coarse_scale_initial))
        )

        self._initialize_v2_layers()

    def _initialize_v2_layers(self) -> None:
        for module in (
            self.lf_stem[0],
            self.fuse,
        ):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        for blocks in (self.lf_blocks, self.refine_blocks):
            for block in blocks:
                nn.init.xavier_uniform_(block.depthwise.weight)
                nn.init.zeros_(block.depthwise.bias)
                nn.init.xavier_uniform_(block.pointwise.weight)
                nn.init.zeros_(block.pointwise.bias)

        # ICNR-like grouped initialization: every subpixel initially copies
        # the center value, avoiding checkerboard structure at epoch zero.
        with torch.no_grad():
            self.subpixel_expand.weight.zero_()
            center = self.subpixel_expand.kernel_size[0] // 2
            self.subpixel_expand.weight[:, 0, center, center] = 1.0
            self.subpixel_expand.bias.zero_()

        # Start as a coarse reconstruction; learn high-resolution corrections.
        nn.init.zeros_(self.to_depth_residual.weight)
        nn.init.zeros_(self.to_depth_residual.bias)

    def forward(
        self, lf: torch.Tensor, *, return_intermediates: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        intermediates = self._forward_to_volume_small(lf)
        volume_small = intermediates["volume_small"]

        subpixel_volume = self.subpixel_shuffle(self.subpixel_expand(volume_small))
        subpixel_volume = F.interpolate(
            subpixel_volume,
            size=lf.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        lf_features = self.lf_stem(lf)
        for block in self.lf_blocks:
            lf_features = block(lf_features)

        fused = self.fuse_activation(
            self.fuse(torch.cat((subpixel_volume, lf_features), dim=1))
        )
        for block in self.refine_blocks:
            fused = block(fused)
        highres_residual = self.to_depth_residual(fused)

        coarse_scale = self.log_coarse_scale_parameter.exp()
        logits = (
            coarse_scale * subpixel_volume
            + self.config.residual_scale * highres_residual
            + self.output_bias_parameter
        )
        reconstruction = F.softplus(
            logits, beta=self.config.output_softplus_beta
        )

        if not return_intermediates:
            return reconstruction
        return {
            **intermediates,
            "subpixel_volume": subpixel_volume,
            "coarse_scale": coarse_scale,
            "lf_features": lf_features,
            "highres_residual": highres_residual,
            "reconstruction": reconstruction,
        }


def build_h1_lite_v2(config: H1LiteV2Config | None = None) -> H1LiteV2:
    return H1LiteV2(config=config)
