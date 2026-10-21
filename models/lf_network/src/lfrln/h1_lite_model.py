from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .h1_model import ConvNormLeaky, ResidualConvBlock, ViewAttention
from .physics_rearrange import LFToViews, SpaceToViewDepth, ViewDepthToSpace


@dataclass(frozen=True)
class H1LiteConfig:
    """Configuration for the lightweight H1 student architecture."""

    depth: int = 91
    h1_nnum: int = 5
    h2_nnum: int = 11
    fp_channels: int = 9
    fp_layers: int = 4
    view_attention_channels: int = 9
    latent_channels: int = 192
    bp_dilations: tuple[int, ...] = (1, 2, 4, 2)
    latent_attention_reduction: int = 8
    ratio_epsilon: float = 1e-5
    instance_norm_epsilon: float = 1e-5
    leaky_relu_slope: float = 0.01
    initial_value: float = 1.0 / 2000.0

    def __post_init__(self) -> None:
        integer_fields = (
            "depth",
            "h1_nnum",
            "h2_nnum",
            "fp_channels",
            "fp_layers",
            "view_attention_channels",
            "latent_channels",
            "latent_attention_reduction",
        )
        for field in integer_fields:
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} must be positive")
        if self.fp_layers % 2:
            raise ValueError("fp_layers must be even")
        if not self.bp_dilations or any(value <= 0 for value in self.bp_dilations):
            raise ValueError("bp_dilations must contain positive values")

    @property
    def num_views(self) -> int:
        return self.h1_nnum**2

    @property
    def projected_channels(self) -> int:
        return self.depth * self.num_views


def _instance_norm(channels: int, config: H1LiteConfig) -> nn.InstanceNorm2d:
    return nn.InstanceNorm2d(
        channels,
        eps=config.instance_norm_epsilon,
        affine=True,
        track_running_stats=False,
    )


class DepthwiseSeparableResidualBlock(nn.Module):
    """Dilated spatial filtering followed by inexpensive channel mixing."""

    def __init__(self, channels: int, *, dilation: int, negative_slope: float) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
        )
        self.activation = nn.LeakyReLU(negative_slope)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.depthwise(x)
        residual = self.activation(residual)
        residual = self.pointwise(residual)
        return x + residual


class IdentityCenteredSE(nn.Module):
    """SE attention initialized to a neutral multiplier of one."""

    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        hidden_channels = max(1, channels // reduction)
        self.reduce = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.activation = nn.ReLU()
        self.expand = nn.Conv2d(hidden_channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.adaptive_avg_pool2d(x, output_size=1)
        weights = self.activation(self.reduce(weights))
        weights = 2.0 * torch.sigmoid(self.expand(weights))
        return x * weights


class DepthwiseSeparableRefine(nn.Module):
    """Full-resolution spatial/depth refinement at much lower MAC cost."""

    def __init__(self, channels: int, config: H1LiteConfig) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = _instance_norm(channels, config)
        self.activation = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.activation(self.norm(x))


class H1GridResize(nn.Module):
    """Resize an H2-grid tensor to the native H1 grid and enforce positivity."""

    def __init__(self, *, h1_nnum: int, h2_nnum: int) -> None:
        super().__init__()
        self.h1_nnum = h1_nnum
        self.h2_nnum = h2_nnum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height_num = x.shape[-2] * self.h1_nnum
        width_num = x.shape[-1] * self.h1_nnum
        size = (height_num // self.h2_nnum, width_num // self.h2_nnum)
        return F.relu(
            F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        )


class ResizeLike(nn.Module):
    """Bilinearly resize the first tensor to the reference spatial shape."""

    def forward(
        self, x: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        return F.interpolate(
            x,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )


class AddEpsilon(nn.Module):
    """Expose ``x + epsilon`` as a named executable graph operation."""

    def __init__(self, epsilon: float) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.epsilon


class AddPairEpsilon(nn.Module):
    """Expose ``x + y + epsilon`` as a named graph operation."""

    def __init__(self, epsilon: float) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y + self.epsilon


class ElementwiseAdd(nn.Module):
    """Named elementwise addition used by generated execution graphs."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


class ElementwiseDivide(nn.Module):
    """Named elementwise division used by generated execution graphs."""

    def forward(
        self, numerator: torch.Tensor, denominator: torch.Tensor
    ) -> torch.Tensor:
        return numerator / denominator


class ElementwiseMultiply(nn.Module):
    """Named elementwise multiplication used by generated execution graphs."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x * y


class H1Lite(nn.Module):
    """
    Lightweight H1 student using exact rearrangement physics and a compact BP.

    The default tensor path is ``[B, 1, 231, 231] -> [B, 91, 231, 231]``.
    Unlike :class:`H1RLN`, this model contains no dense fixed optical tensors;
    the audited one-hot operators are expressed as pixel rearrangements.
    """

    def __init__(self, config: H1LiteConfig | None = None) -> None:
        super().__init__()
        self.config = config or H1LiteConfig()
        c = self.config

        self.initial = nn.Conv2d(1, c.depth, kernel_size=1)

        # Central RL algebra and rearrangements are modules rather than hidden
        # functional calls, allowing the executed network graph to audit them.
        self.observed_resize = H1GridResize(
            h1_nnum=c.h1_nnum, h2_nnum=c.h2_nnum
        )
        self.estimate_resize = H1GridResize(
            h1_nnum=c.h1_nnum, h2_nnum=c.h2_nnum
        )
        self.forward_rearrange = SpaceToViewDepth(c.h1_nnum)
        self.observed_rearrange = LFToViews(c.h1_nnum)
        self.ratio_numerator_add = AddEpsilon(c.ratio_epsilon)
        self.ratio_denominator_add = AddPairEpsilon(c.ratio_epsilon)
        self.ratio_divide = ElementwiseDivide()
        self.backward_rearrange = ViewDepthToSpace(
            channels=c.depth, block_size=c.h1_nnum
        )

        # Keep the already-small learned forward branch from H1.
        self.fp_extend = nn.Conv2d(
            c.projected_channels, c.fp_channels, kernel_size=3, padding=1
        )
        fp_dilations = (1, 2, 4, 2, 1)
        self.fp_blocks = nn.ModuleList(
            ResidualConvBlock(
                c.fp_channels,
                dilation=fp_dilations[index],
                negative_slope=c.leaky_relu_slope,
            )
            for index in range(c.fp_layers // 2)
        )
        self.fp_refine = ConvNormLeaky(
            c.fp_channels, c.fp_channels, kernel_size=3, config=c
        )
        self.fp_to_views = ConvNormLeaky(
            c.fp_channels, c.num_views, kernel_size=3, config=c
        )

        self.ratio_norm = _instance_norm(c.num_views, c)
        self.view_attention = ViewAttention(c)

        # Compact learned backprojection.
        self.bp_input = nn.Conv2d(c.num_views, c.latent_channels, kernel_size=1)
        self.bp_blocks = nn.ModuleList(
            DepthwiseSeparableResidualBlock(
                c.latent_channels,
                dilation=dilation,
                negative_slope=c.leaky_relu_slope,
            )
            for dilation in c.bp_dilations
        )
        self.latent_attention = IdentityCenteredSE(
            c.latent_channels, c.latent_attention_reduction
        )
        self.bp_output = nn.Conv2d(
            c.latent_channels, c.projected_channels, kernel_size=1
        )
        self.bp_output_norm = _instance_norm(c.projected_channels, c)
        self.bp_output_activation = nn.LeakyReLU(c.leaky_relu_slope)

        self.up_refine = DepthwiseSeparableRefine(c.depth, c)
        self.output_norm = _instance_norm(c.depth, c)
        self.output_activation = nn.Softplus()

        self._initialize_layers()

    def _initialize_layers(self) -> None:
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        for layer in (
            self.view_attention.squeeze,
            self.view_attention.excite,
            self.latent_attention.reduce,
        ):
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")

        # Start latent attention as the identity: 2 * sigmoid(0) = 1.
        nn.init.zeros_(self.latent_attention.expand.weight)
        nn.init.zeros_(self.latent_attention.expand.bias)

        with torch.no_grad():
            self.initial.weight.zero_()
            self.initial.bias.fill_(self.config.initial_value)
        self.initial.requires_grad_(False)

    def _small_spatial_size(self, lf: torch.Tensor) -> tuple[int, int]:
        height, width = lf.shape[-2:]
        height_num = height * self.config.h1_nnum
        width_num = width * self.config.h1_nnum
        if height_num % self.config.h2_nnum or width_num % self.config.h2_nnum:
            raise ValueError(
                "Input must produce integer H1 dimensions under scale "
                f"{self.config.h1_nnum}/{self.config.h2_nnum}; "
                f"got {(height, width)}"
            )
        return height_num // self.config.h2_nnum, width_num // self.config.h2_nnum

    def _validate_input(self, lf: torch.Tensor) -> None:
        if lf.ndim != 4 or lf.shape[1] != 1:
            raise ValueError(f"Expected LF [B, 1, H, W], got {tuple(lf.shape)}")
        small_height, small_width = self._small_spatial_size(lf)
        if small_height % self.config.h1_nnum or small_width % self.config.h1_nnum:
            raise ValueError(
                "Resized LF dimensions must be divisible by h1_nnum; "
                f"got {(small_height, small_width)}"
            )

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def _predict_views(self, forward_features: torch.Tensor) -> torch.Tensor:
        """Map FP features to views; subclasses may enforce physical units."""
        return self.fp_to_views(forward_features)

    def _backprojection_ratio_features(
        self,
        normalized_ratio: torch.Tensor,
        ratio: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Prepare ratio features consumed by the learned backprojection.

        Subclasses can retain the positive pre-normalization ratio while the
        original lightweight model remains checkpoint-compatible.
        """
        return self.view_attention(normalized_ratio, ratio), {}

    def _forward_to_volume_small(
        self, lf: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Run the shared RL/phase-space path up to the native H1 grid."""
        self._validate_input(lf)
        c = self.config

        initial = self.initial(lf)
        observed_small = self.observed_resize(lf)
        estimate_small = self.estimate_resize(initial)

        forward_projected = self.forward_rearrange(estimate_small)
        forward_features = self.fp_extend(forward_projected)
        for block in self.fp_blocks:
            forward_features = block(forward_features)
        forward_features = self.fp_refine(forward_features)
        predicted_views = self._predict_views(forward_features)

        observed_views = self.observed_rearrange(observed_small)
        ratio_numerator = self.ratio_numerator_add(observed_views)
        ratio_denominator = self.ratio_denominator_add(
            observed_views, predicted_views
        )
        ratio = self.ratio_divide(ratio_numerator, ratio_denominator)
        normalized_ratio = self.ratio_norm(ratio)
        correction, ratio_intermediates = self._backprojection_ratio_features(
            normalized_ratio, ratio
        )

        latent = self.bp_input(correction)
        for block in self.bp_blocks:
            latent = block(latent)
        latent = self.latent_attention(latent)
        depth_view_features = self.bp_output_activation(
            self.bp_output_norm(self.bp_output(latent))
        )

        volume_small = self.backward_rearrange(depth_view_features)
        return {
            "initial": initial,
            "observed_views": observed_views,
            "predicted_views": predicted_views,
            "ratio_numerator": ratio_numerator,
            "ratio_denominator": ratio_denominator,
            "ratio": ratio,
            "normalized_ratio": normalized_ratio,
            "correction": correction,
            **ratio_intermediates,
            "latent": latent,
            "depth_view_features": depth_view_features,
            "volume_small": volume_small,
        }

    def forward(
        self, lf: torch.Tensor, *, return_intermediates: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        intermediates = self._forward_to_volume_small(lf)
        initial = intermediates["initial"]
        volume_small = intermediates["volume_small"]
        volume = F.interpolate(
            volume_small, size=lf.shape[-2:], mode="bilinear", align_corners=False
        )
        volume = self.up_refine(volume)
        reconstruction = self.output_activation(self.output_norm(volume * initial))

        if not return_intermediates:
            return reconstruction
        return {
            **intermediates,
            "volume_correction": volume,
            "reconstruction": reconstruction,
        }


def build_h1_lite(config: H1LiteConfig | None = None) -> H1Lite:
    return H1Lite(config=config)
