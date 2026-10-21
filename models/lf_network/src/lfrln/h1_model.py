from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class H1Config:
    """Architecture parameters for the MATLAB H1 RLN model."""

    depth: int = 91
    h1_nnum: int = 5
    h2_nnum: int = 11
    fp_channels: int = 9
    fp_layers: int = 4
    bp_channels_per_depth: int = 12
    bp_layers: int = 4
    view_attention_channels: int = 9
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
            "bp_channels_per_depth",
            "bp_layers",
            "view_attention_channels",
        )
        for field in integer_fields:
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} must be positive")

        if self.fp_layers % 2 != 0:
            raise ValueError("fp_layers must be even (two convolutions per block)")
        if self.bp_layers % 2 != 0:
            raise ValueError("bp_layers must be even (two convolutions per block)")

    @property
    def num_views(self) -> int:
        return self.h1_nnum**2

    @property
    def projected_channels(self) -> int:
        return self.num_views * self.depth

    @property
    def bp_channels(self) -> int:
        return self.bp_channels_per_depth * self.depth


def _instance_norm(channels: int, config: H1Config) -> nn.InstanceNorm2d:
    return nn.InstanceNorm2d(
        channels,
        eps=config.instance_norm_epsilon,
        affine=True,
        track_running_stats=False,
    )


class ResidualConvBlock(nn.Module):
    """Two-convolution residual block used by the H1 forward/back branches."""

    def __init__(
        self,
        channels: int,
        *,
        dilation: int,
        negative_slope: float,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.activation(self.conv1(x)))


class ConvNormLeaky(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        config: H1Config,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.norm = _instance_norm(out_channels, config)
        self.activation = nn.LeakyReLU(config.leaky_relu_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(x)))


class ViewAttention(nn.Module):
    """MATLAB H1 view-aware squeeze/excitation branch."""

    def __init__(self, config: H1Config) -> None:
        super().__init__()
        self.squeeze = nn.Conv2d(
            config.num_views, config.view_attention_channels, kernel_size=1
        )
        self.activation = nn.ReLU()
        self.excite = nn.Conv2d(
            config.view_attention_channels, config.num_views, kernel_size=1
        )

    def attention_weights(self, ratio: torch.Tensor) -> torch.Tensor:
        """Return positive, identity-centered per-view weights."""
        attention = F.adaptive_avg_pool2d(ratio, output_size=1)
        attention = self.squeeze(attention)
        attention = self.activation(attention)
        return 2.0 * torch.sigmoid(self.excite(attention))

    def forward(
        self, normalized_ratio: torch.Tensor, ratio: torch.Tensor
    ) -> torch.Tensor:
        return normalized_ratio * self.attention_weights(ratio)


class GlobalContextBlock(nn.Module):
    """
    Reproduce the active MATLAB H1 ``GC2`` graph.

    MATLAB's softmax layer acts over channels. Because the mask has one
    channel, the mask is identically one, but the operation is retained for
    checkpoint and graph fidelity.
    """

    def __init__(self, channels: int, config: H1Config) -> None:
        super().__init__()
        self.mask = nn.Conv2d(channels, 1, kernel_size=1)
        self.transform1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = _instance_norm(channels, config)
        self.activation = nn.ReLU()
        self.transform2 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.softmax(self.mask(x), dim=1)
        context = x * mask
        context = self.transform1(context)
        context = self.norm(context)
        context = self.activation(context)
        context = self.transform2(context)
        return x + context


class H1RLN(nn.Module):
    """
    PyTorch port of the active MATLAB H1 reconstruction network.

    Input
    -----
    ``lf``: ``[batch, 1, height, width]``

    Output
    ------
    Reconstructed volume: ``[batch, depth, height, width]``

    Notes
    -----
    The initial estimate and the three optical projection layers are frozen,
    as in the MATLAB graph. Call :func:`lfrln.matlab_weights.load_h1_matlab_export`
    before real inference/training so those optical layers contain the actual
    microscope kernels.
    """

    def __init__(self, config: H1Config | None = None) -> None:
        super().__init__()
        self.config = config or H1Config()
        c = self.config

        self.initial = nn.Conv2d(1, c.depth, kernel_size=1)

        # Fixed optical operators from the MATLAB realignment-layer files.
        self.forward_project = nn.Conv2d(
            c.depth,
            c.projected_channels,
            kernel_size=c.h1_nnum,
            stride=c.h1_nnum,
        )
        self.observed_backproject = nn.Conv2d(
            1,
            c.num_views,
            kernel_size=c.h1_nnum,
            stride=c.h1_nnum,
        )
        self.volume_backproject = nn.ConvTranspose2d(
            c.projected_channels,
            c.depth,
            kernel_size=c.h1_nnum,
            stride=c.h1_nnum,
        )

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

        self.bp_extend = nn.Conv2d(c.num_views, c.bp_channels, kernel_size=3, padding=1)
        self.bp_blocks = nn.ModuleList(
            ResidualConvBlock(
                c.bp_channels,
                dilation=fp_dilations[index],
                negative_slope=c.leaky_relu_slope,
            )
            for index in range(c.bp_layers // 2)
        )
        self.bp_refine = ConvNormLeaky(
            c.bp_channels, c.bp_channels, kernel_size=3, config=c
        )
        self.bp_to_depth_views = ConvNormLeaky(
            c.bp_channels, c.projected_channels, kernel_size=3, config=c
        )
        self.global_context = GlobalContextBlock(c.projected_channels, c)

        self.up_refine = nn.Conv2d(c.depth, c.depth, kernel_size=3, padding=1)
        self.up_norm = _instance_norm(c.depth, c)
        self.up_activation = nn.Softplus()

        self.output_norm = _instance_norm(c.depth, c)
        self.output_activation = nn.Softplus()

        self.register_buffer(
            "_optics_loaded", torch.tensor(False, dtype=torch.bool), persistent=True
        )
        self._initialize_learned_layers()
        self._initialize_fixed_layers()

    @property
    def optics_loaded(self) -> bool:
        return bool(self._optics_loaded.item())

    def mark_optics_loaded(self, value: bool = True) -> None:
        self._optics_loaded.fill_(value)

    def _initialize_learned_layers(self) -> None:
        # MATLAB convolution2dLayer defaults to Glorot initialization.
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # These two layers explicitly use WeightsInitializer="he" in MATLAB.
        for layer in (self.view_attention.squeeze, self.view_attention.excite):
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def _initialize_fixed_layers(self) -> None:
        with torch.no_grad():
            self.initial.weight.zero_()
            self.initial.bias.fill_(self.config.initial_value)

            # Zero is safer than silently using random "optical" kernels.
            for layer in (
                self.forward_project,
                self.observed_backproject,
                self.volume_backproject,
            ):
                layer.weight.zero_()
                if layer.bias is not None:
                    layer.bias.zero_()

        for layer in (
            self.initial,
            self.forward_project,
            self.observed_backproject,
            self.volume_backproject,
        ):
            layer.requires_grad_(False)

    def freeze_all(self) -> H1RLN:
        """Freeze H1 for later insertion before H2, matching MATLAB H1+H2."""
        self.requires_grad_(False)
        return self

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def _small_spatial_size(self, lf: torch.Tensor) -> tuple[int, int]:
        height, width = lf.shape[-2:]
        height_num = height * self.config.h1_nnum
        width_num = width * self.config.h1_nnum
        if height_num % self.config.h2_nnum or width_num % self.config.h2_nnum:
            raise ValueError(
                "Input height and width must produce integer H1 sizes under "
                f"scale {self.config.h1_nnum}/{self.config.h2_nnum}; "
                f"got {(height, width)}"
            )
        return (
            height_num // self.config.h2_nnum,
            width_num // self.config.h2_nnum,
        )

    def _validate_input(self, lf: torch.Tensor) -> None:
        if lf.ndim != 4:
            raise ValueError(
                f"Expected LF tensor [B, 1, H, W], got shape {tuple(lf.shape)}"
            )
        if lf.shape[1] != 1:
            raise ValueError(f"Expected one LF channel, got {lf.shape[1]}")
        small_height, small_width = self._small_spatial_size(lf)
        if small_height % self.config.h1_nnum or small_width % self.config.h1_nnum:
            raise ValueError(
                "The resized LF dimensions must be divisible by h1_nnum; "
                f"got {(small_height, small_width)}"
            )

    def forward(
        self, lf: torch.Tensor, *, return_intermediates: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        self._validate_input(lf)
        c = self.config
        output_size = lf.shape[-2:]
        small_size = self._small_spatial_size(lf)

        initial = self.initial(lf)
        observed_small = F.relu(
            F.interpolate(lf, size=small_size, mode="bilinear", align_corners=False)
        )
        estimate_small = F.relu(
            F.interpolate(
                initial, size=small_size, mode="bilinear", align_corners=False
            )
        )

        forward_projected = self.forward_project(estimate_small)
        forward_features = self.fp_extend(forward_projected)
        for block in self.fp_blocks:
            forward_features = block(forward_features)
        forward_features = self.fp_refine(forward_features)
        predicted_views = self.fp_to_views(forward_features)

        observed_views = self.observed_backproject(observed_small)
        ratio = (observed_views + c.ratio_epsilon) / (
            observed_views + predicted_views + c.ratio_epsilon
        )
        normalized_ratio = self.ratio_norm(ratio)
        correction = self.view_attention(normalized_ratio, ratio)

        back_features = self.bp_extend(correction)
        for block in self.bp_blocks:
            back_features = block(back_features)
        back_features = self.bp_refine(back_features)
        depth_view_pre_context = self.bp_to_depth_views(back_features)
        depth_view_features = self.global_context(depth_view_pre_context)

        volume_small = self.volume_backproject(depth_view_features)
        volume_upsampled = F.interpolate(
            volume_small, size=output_size, mode="bilinear", align_corners=False
        )
        volume = self.up_activation(self.up_norm(self.up_refine(volume_upsampled)))
        reconstruction = self.output_activation(self.output_norm(volume * initial))

        if not return_intermediates:
            return reconstruction

        return {
            "initial": initial,
            "observed_small": observed_small,
            "estimate_small": estimate_small,
            "forward_projected": forward_projected,
            "forward_features": forward_features,
            "observed_views": observed_views,
            "predicted_views": predicted_views,
            "ratio": ratio,
            "normalized_ratio": normalized_ratio,
            "correction": correction,
            "back_features": back_features,
            "depth_view_pre_context": depth_view_pre_context,
            "depth_view_features": depth_view_features,
            "volume_small": volume_small,
            "volume_upsampled": volume_upsampled,
            "volume_correction": volume,
            "reconstruction": reconstruction,
        }


def build_h1(config: H1Config | None = None) -> H1RLN:
    """Construct an H1 model; load exported MATLAB optics before real use."""
    return H1RLN(config=config)


def parameter_summary(model: nn.Module) -> dict[str, Any]:
    """Return compact parameter statistics without allocating extra tensors."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }
