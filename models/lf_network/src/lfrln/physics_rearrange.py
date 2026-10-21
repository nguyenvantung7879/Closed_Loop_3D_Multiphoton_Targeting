from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def space_to_view_depth(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """
    Rearrange spatial blocks into view-major, channel-minor features.

    ``[B, C, H*r, W*r] -> [B, r*r*C, H, W]`` with output channel order
    ``view_id * C + channel``. This reproduces the current one-hot MATLAB H1
    forward projection without a convolution.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected [B, C, H, W], got {tuple(x.shape)}")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if x.shape[-2] % block_size or x.shape[-1] % block_size:
        raise ValueError(
            f"Spatial dimensions {tuple(x.shape[-2:])} are not divisible by "
            f"block_size={block_size}"
        )

    batch, channels = x.shape[:2]
    view_count = block_size**2
    rearranged = F.pixel_unshuffle(x, block_size)
    height, width = rearranged.shape[-2:]
    return (
        rearranged.reshape(batch, channels, view_count, height, width)
        .permute(0, 2, 1, 3, 4)
        .reshape(batch, view_count * channels, height, width)
    )


def view_depth_to_space(
    x: torch.Tensor, *, channels: int, block_size: int
) -> torch.Tensor:
    """Inverse of :func:`space_to_view_depth`."""
    if x.ndim != 4:
        raise ValueError(f"Expected [B, V*C, H, W], got {tuple(x.shape)}")
    if channels <= 0 or block_size <= 0:
        raise ValueError("channels and block_size must be positive")

    batch, input_channels, height, width = x.shape
    view_count = block_size**2
    expected_channels = channels * view_count
    if input_channels != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} input channels "
            f"({channels} * {view_count}), got {input_channels}"
        )

    depth_major = (
        x.reshape(batch, view_count, channels, height, width)
        .permute(0, 2, 1, 3, 4)
        .reshape(batch, channels * view_count, height, width)
    )
    return F.pixel_shuffle(depth_major, block_size)


def lf_to_views(lf: torch.Tensor, block_size: int) -> torch.Tensor:
    """Rearrange a one-channel LF image into ``block_size**2`` views."""
    if lf.ndim != 4 or lf.shape[1] != 1:
        raise ValueError(f"Expected LF [B, 1, H, W], got {tuple(lf.shape)}")
    return F.pixel_unshuffle(lf, block_size)


class SpaceToViewDepth(nn.Module):
    """Module wrapper that exposes pixel-unshuffle in execution graphs."""

    def __init__(self, block_size: int) -> None:
        super().__init__()
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return space_to_view_depth(x, self.block_size)


class ViewDepthToSpace(nn.Module):
    """Module wrapper that exposes inverse pixel-shuffle in execution graphs."""

    def __init__(self, *, channels: int, block_size: int) -> None:
        super().__init__()
        self.channels = channels
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return view_depth_to_space(
            x,
            channels=self.channels,
            block_size=self.block_size,
        )


class LFToViews(nn.Module):
    """Module wrapper that exposes LF-to-view rearrangement in graphs."""

    def __init__(self, block_size: int) -> None:
        super().__init__()
        self.block_size = block_size

    def forward(self, lf: torch.Tensor) -> torch.Tensor:
        return lf_to_views(lf, self.block_size)
