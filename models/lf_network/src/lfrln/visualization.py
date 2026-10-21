from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from torch import nn


def plot_h1_block_details(model: nn.Module) -> Figure:
    """Expand the configured H1 FP, ratio/attention, and BP/decoder blocks.

    ``model`` may be an H1 model itself or a connected model exposing ``h1``.
    Channel counts, layer counts, dilations, ratio-path mode, and decoder mode
    are read from the live modules so the figure changes with the constructed
    network rather than documenting a separate hand-maintained architecture.
    """
    h1 = model.h1 if hasattr(model, "h1") else model
    for name in (
        "fp_extend",
        "fp_blocks",
        "fp_refine",
        "fp_to_views",
        "ratio_norm",
        "view_attention",
        "bp_input",
        "bp_blocks",
        "latent_attention",
        "bp_output",
        "backward_rearrange",
    ):
        if not hasattr(h1, name):
            raise TypeError(f"H1 model is missing its {name!r} module")

    colors = {
        "input": "#F1F3F5",
        "fp": "#DCEBFA",
        "operation": "#FFF1C9",
        "attention": "#E8DFF5",
        "bp": "#E9F5E4",
    }

    def conv_label(title: str, layer: nn.Conv2d) -> str:
        kernel = "×".join(str(value) for value in layer.kernel_size)
        dilation = "×".join(str(value) for value in layer.dilation)
        groups = f", groups={layer.groups}" if layer.groups != 1 else ""
        return (
            f"{title}\nConv2d {layer.in_channels}→{layer.out_channels}\n"
            f"k={kernel}, dilation={dilation}{groups}"
        )

    def fp_block_label(index: int, block: nn.Module) -> str:
        dilation = block.conv1.dilation[0]
        return (
            f"FP residual block {index}\n"
            f"Conv 3×3 d={dilation} → {type(block.activation).__name__}\n"
            "Conv 3×3 d=1 + identity skip"
        )

    def bp_block_label(index: int, block: nn.Module) -> str:
        depthwise = block.depthwise
        return (
            f"BP residual block {index}\n"
            f"DW 3×3 d={depthwise.dilation[0]}, groups={depthwise.groups}\n"
            f"{type(block.activation).__name__} → PW 1×1 + skip"
        )

    def node(
        x: float,
        y: float,
        label: str,
        kind: str,
        *,
        width: float = 0.11,
        height: float = 0.36,
    ) -> dict[str, object]:
        return {
            "x": x,
            "y": y,
            "label": label,
            "kind": kind,
            "width": width,
            "height": height,
        }

    def spaced(count: int, start: float, stop: float) -> tuple[float, ...]:
        if count == 1:
            return ((start + stop) / 2,)
        return tuple(
            start + (stop - start) * index / (count - 1)
            for index in range(count)
        )

    def draw_graph(
        axis: plt.Axes,
        nodes: dict[str, dict[str, object]],
        edges: list[tuple[str, str]],
        title: str,
    ) -> None:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")
        for item in nodes.values():
            x = float(item["x"])
            y = float(item["y"])
            width = float(item["width"])
            height = float(item["height"])
            axis.add_patch(
                FancyBboxPatch(
                    (x - width / 2, y - height / 2),
                    width,
                    height,
                    boxstyle="round,pad=0.012",
                    facecolor=colors[str(item["kind"])],
                    edgecolor="#29465B",
                    linewidth=1.15,
                )
            )
            axis.text(
                x,
                y,
                str(item["label"]),
                ha="center",
                va="center",
                fontsize=7.4,
            )
        for source_name, target_name in edges:
            source = nodes[source_name]
            target = nodes[target_name]
            sx, sy = float(source["x"]), float(source["y"])
            tx, ty = float(target["x"]), float(target["y"])
            sw, sh = float(source["width"]), float(source["height"])
            tw, th = float(target["width"]), float(target["height"])
            dx, dy = tx - sx, ty - sy
            if abs(dx) >= abs(dy):
                direction = 1 if dx >= 0 else -1
                start = (sx + direction * sw / 2, sy)
                end = (tx - direction * tw / 2, ty)
            else:
                direction = 1 if dy >= 0 else -1
                start = (sx, sy + direction * sh / 2)
                end = (tx, ty - direction * th / 2)
            axis.add_patch(
                FancyArrowPatch(
                    start,
                    end,
                    arrowstyle="-|>",
                    mutation_scale=10,
                    linewidth=1.0,
                    color="#29465B",
                )
            )
        axis.set_title(title, fontsize=12, weight="bold")

    config = h1.config
    fp_refine = h1.fp_refine.conv
    fp_head = (
        h1.fp_to_views
        if isinstance(h1.fp_to_views, nn.Conv2d)
        else h1.fp_to_views.conv
    )
    fp_nodes: dict[str, dict[str, object]] = {
        "input": node(
            0.045,
            0.5,
            f"Initial estimate x0\n{config.depth}×H×W",
            "input",
            width=0.08,
        ),
        "resize": node(
            0.135,
            0.5,
            f"H1GridResize\nscale {config.h1_nnum}/{config.h2_nnum}",
            "operation",
            width=0.08,
        ),
        "rearrange": node(
            0.235,
            0.5,
            f"SpaceToViewDepth S{config.h1_nnum}\n"
            f"{config.projected_channels} phase channels",
            "operation",
            width=0.095,
        ),
        "extend": node(
            0.345,
            0.5,
            conv_label("FP extend", h1.fp_extend),
            "fp",
            width=0.10,
        ),
    }
    fp_edges = [
        ("input", "resize"),
        ("resize", "rearrange"),
        ("rearrange", "extend"),
    ]
    previous = "extend"
    for index, (block, x) in enumerate(
        zip(
            h1.fp_blocks,
            spaced(len(h1.fp_blocks), 0.45, 0.65),
            strict=True,
        ),
        1,
    ):
        name = f"block_{index}"
        fp_nodes[name] = node(
            x,
            0.5,
            fp_block_label(index, block),
            "fp",
            width=0.09,
        )
        fp_edges.append((previous, name))
        previous = name
    fp_nodes.update(
        {
            "refine": node(
                0.75,
                0.5,
                conv_label("FP refine", fp_refine)
                + "\nInstanceNorm2d → LeakyReLU",
                "fp",
                width=0.095,
            ),
            "head": node(
                0.855,
                0.5,
                conv_label("FP view head", fp_head),
                "fp",
                width=0.095,
            ),
            "positive": node(
                0.955,
                0.5,
                (
                    f"Softplus β={config.fp_output_softplus_beta:g}\n"
                    "× exp(log sFP)\npositive predicted views"
                    if config.use_physical_fp_head
                    else "InstanceNorm2d → LeakyReLU\npredicted views"
                ),
                "operation",
                width=0.085,
            ),
        }
    )
    fp_edges.extend(
        [(previous, "refine"), ("refine", "head"), ("head", "positive")]
    )

    squeeze = h1.view_attention.squeeze
    excite = h1.view_attention.excite
    raw_ratio_path = bool(config.use_raw_ratio_bp_path)
    ratio_feature_label = (
        "Raw attended ratio\n+ α × normalized attended ratio\n"
        f"|α| < {config.normalized_ratio_residual_max:g}"
        if raw_ratio_path
        else "Normalized ratio × view weights"
    )
    attention_nodes = {
        "observed": node(
            0.045,
            0.72,
            f"Observed views y5\n{config.num_views} channels",
            "input",
            width=0.085,
        ),
        "predicted": node(
            0.045,
            0.28,
            f"Predicted views ŷ5\n{config.num_views} channels",
            "fp",
            width=0.085,
        ),
        "division": node(
            0.17,
            0.5,
            "Numerator y5 + ε\nDenominator y5 + ŷ5 + ε\nElementwise division",
            "operation",
            width=0.12,
        ),
        "norm": node(
            0.32,
            0.70,
            f"{type(h1.ratio_norm).__name__}\nnormalized-ratio branch",
            "operation",
            width=0.105,
        ),
        "pool": node(
            0.32,
            0.25,
            "Raw-ratio statistics\nGlobal average pool",
            "attention",
            width=0.105,
        ),
        "squeeze": node(
            0.47,
            0.25,
            conv_label("View squeeze", squeeze),
            "attention",
            width=0.115,
        ),
        "activation": node(
            0.59,
            0.25,
            type(h1.view_attention.activation).__name__,
            "attention",
            width=0.08,
        ),
        "excite": node(
            0.70,
            0.25,
            conv_label("View excite", excite),
            "attention",
            width=0.105,
        ),
        "weights": node(
            0.81,
            0.25,
            f"2 × sigmoid\n{config.num_views} weights\nrange [0, 2]",
            "attention",
            width=0.09,
        ),
        "feature": node(
            0.84,
            0.70,
            ratio_feature_label,
            "operation",
            width=0.15,
        ),
        "output": node(
            0.965,
            0.70,
            f"BP ratio feature\n{config.num_views} channels",
            "bp",
            width=0.08,
        ),
    }
    attention_edges = [
        ("observed", "division"),
        ("predicted", "division"),
        ("division", "norm"),
        ("division", "pool"),
        ("pool", "squeeze"),
        ("squeeze", "activation"),
        ("activation", "excite"),
        ("excite", "weights"),
        ("norm", "feature"),
        ("weights", "feature"),
        ("feature", "output"),
    ]
    if raw_ratio_path:
        attention_edges.append(("division", "feature"))

    bp_nodes: dict[str, dict[str, object]] = {
        "input": node(
            0.04,
            0.73,
            f"BP ratio feature\n{config.num_views} channels",
            "input",
            width=0.075,
        ),
        "bp_input": node(
            0.125,
            0.73,
            conv_label("BP input", h1.bp_input),
            "bp",
            width=0.085,
        ),
    }
    bp_edges = [("input", "bp_input")]
    previous = "bp_input"
    for index, (block, x) in enumerate(
        zip(
            h1.bp_blocks,
            spaced(len(h1.bp_blocks), 0.21, 0.64),
            strict=True,
        ),
        1,
    ):
        name = f"bp_block_{index}"
        bp_nodes[name] = node(
            x,
            0.73,
            bp_block_label(index, block),
            "bp",
            width=0.074,
        )
        bp_edges.append((previous, name))
        previous = name
    latent = h1.latent_attention
    latent_hidden = latent.reduce.out_channels
    bp_nodes.update(
        {
            "latent_attention": node(
                0.73,
                0.73,
                "Latent SE attention\nGlobal average pool\n"
                f"1×1 {latent.reduce.in_channels}→{latent_hidden} → ReLU\n"
                f"1×1 {latent_hidden}→{latent.expand.out_channels} → 2 sigmoid",
                "attention",
                width=0.095,
            ),
            "bp_output": node(
                0.835,
                0.73,
                conv_label("BP output", h1.bp_output)
                + "\nInstanceNorm2d → LeakyReLU",
                "bp",
                width=0.095,
            ),
            "shuffle": node(
                0.95,
                0.73,
                f"ViewDepthToSpace S{config.h1_nnum}⁻¹\n"
                f"{config.depth} volume channels",
                "operation",
                width=0.10,
            ),
        }
    )
    bp_edges.extend(
        [
            (previous, "latent_attention"),
            ("latent_attention", "bp_output"),
            ("bp_output", "shuffle"),
        ]
    )
    if config.decoder_mode == "phase_mixing_resize":
        decoder_label = (
            f"Depthwise phase mixer {config.phase_mixer_kernel_size}×"
            f"{config.phase_mixer_kernel_size}\n"
            f"zero-init residual × {config.phase_mixer_residual_scale:g}"
        )
    else:
        decoder_label = "Learned 2× subpixel decoder\nConv → PixelShuffle"
    bp_nodes.update(
        {
            "decoder": node(
                0.82,
                0.25,
                decoder_label,
                "bp",
                width=0.12,
            ),
            "resize": node(
                0.68,
                0.25,
                "Bilinear resize\nback to LF H×W",
                "operation",
                width=0.09,
            ),
            "logits": node(
                0.55,
                0.25,
                "Correction logits\n× learned coarse scale\n+ learned depth bias",
                "bp",
                width=0.105,
            ),
            "positive": node(
                0.41,
                0.25,
                f"Softplus β={config.output_softplus_beta:g}\n"
                "× learned positive depth scale\ncomputed correction c1",
                "operation",
                width=0.115,
            ),
            "multiply": node(
                0.27,
                0.25,
                "Explicit RL multiplication\nx1 = x0 ⊙ c1",
                "operation",
                width=0.105,
            ),
            "output": node(
                0.135,
                0.25,
                f"H1 cutoff output x1\n{config.depth}×H×W",
                "input",
                width=0.10,
            ),
        }
    )
    bp_edges.extend(
        [
            ("shuffle", "decoder"),
            ("decoder", "resize"),
            ("resize", "logits"),
            ("logits", "positive"),
            ("positive", "multiply"),
            ("multiply", "output"),
        ]
    )

    figure, axes = plt.subplots(3, 1, figsize=(23, 12), layout="constrained")
    draw_graph(axes[0], fp_nodes, fp_edges, "H1 forward projector (FP)")
    draw_graph(
        axes[1],
        attention_nodes,
        attention_edges,
        "H1 division, raw/normalized ratio paths, and view attention",
    )
    draw_graph(
        axes[2],
        bp_nodes,
        bp_edges,
        "H1 backward projector (BP), correction decoder, and RL update",
    )
    return figure


def plot_h2_fusion_block_details(model: nn.Module) -> tuple[Figure, Figure]:
    """Draw H2 and fusion internals from the modules actually constructed.

    The first figure expands the H2 forward projector, ratio/view-attention
    path, and backward-projector/update path.  The second expands the reusable
    depthwise-separable residual block and the final H1/H2 fusion module.
    Labels are derived from the configured layers so architectural changes are
    reflected when Notebook 08 is rerun.
    """
    for name in ("h2", "fusion"):
        if not hasattr(model, name):
            raise TypeError(f"Connected model is missing its {name!r} module")
    h2 = model.h2
    fusion = model.fusion
    for name in (
        "fp_input",
        "fp_blocks",
        "fp_to_views",
        "view_attention",
        "bp_input",
        "bp_blocks",
        "bp_output",
        "volume_blocks",
        "correction_head",
    ):
        if not hasattr(h2, name):
            raise TypeError(f"H2 model is missing its {name!r} module")
    if h2.config.use_additive_detail_path and not hasattr(
        h2, "additive_detail_head"
    ):
        raise TypeError("Configured H2 is missing its additive detail head")

    colors = {
        "input": "#F1F3F5",
        "fp": "#DCEBFA",
        "operation": "#FFF1C9",
        "attention": "#E8DFF5",
        "bp": "#E9F5E4",
        "fusion": "#E5F3EE",
    }

    def conv_label(title: str, layer: nn.Conv2d) -> str:
        kernel = "×".join(str(value) for value in layer.kernel_size)
        dilation = "×".join(str(value) for value in layer.dilation)
        groups = f", groups={layer.groups}" if layer.groups != 1 else ""
        return (
            f"{title}\nConv2d {layer.in_channels}→{layer.out_channels}\n"
            f"k={kernel}, dilation={dilation}{groups}"
        )

    def residual_label(title: str, block: nn.Module) -> str:
        depthwise = block.depthwise
        pointwise = block.pointwise
        dilation = depthwise.dilation[0]
        return (
            f"{title}\nDW 3×3, d={dilation}, groups={depthwise.groups}\n"
            f"activation → PW 1×1 {pointwise.in_channels}→{pointwise.out_channels}\n"
            "+ identity skip"
        )

    def node(
        x: float,
        y: float,
        label: str,
        kind: str,
        width: float = 0.115,
        height: float = 0.38,
    ) -> dict[str, object]:
        return {
            "x": x,
            "y": y,
            "label": label,
            "kind": kind,
            "width": width,
            "height": height,
        }

    def spaced(count: int, start: float, stop: float) -> tuple[float, ...]:
        if count <= 0:
            return ()
        if count == 1:
            return ((start + stop) / 2,)
        return tuple(
            start + (stop - start) * index / (count - 1)
            for index in range(count)
        )

    def draw_graph(
        axis: plt.Axes,
        nodes: dict[str, dict[str, object]],
        edges: list[tuple[str, str]],
        title: str,
    ) -> None:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")
        for item in nodes.values():
            x = float(item["x"])
            y = float(item["y"])
            width = float(item["width"])
            height = float(item["height"])
            axis.add_patch(
                FancyBboxPatch(
                    (x - width / 2, y - height / 2),
                    width,
                    height,
                    boxstyle="round,pad=0.012",
                    facecolor=colors[str(item["kind"])],
                    edgecolor="#29465B",
                    linewidth=1.15,
                )
            )
            axis.text(
                x,
                y,
                str(item["label"]),
                ha="center",
                va="center",
                fontsize=7.6,
            )
        for source_name, target_name in edges:
            source = nodes[source_name]
            target = nodes[target_name]
            sx, sy = float(source["x"]), float(source["y"])
            tx, ty = float(target["x"]), float(target["y"])
            sw, sh = float(source["width"]), float(source["height"])
            tw, th = float(target["width"]), float(target["height"])
            dx, dy = tx - sx, ty - sy
            if abs(dx) >= abs(dy):
                direction = 1 if dx >= 0 else -1
                start = (sx + direction * sw / 2, sy)
                end = (tx - direction * tw / 2, ty)
            else:
                direction = 1 if dy >= 0 else -1
                start = (sx, sy + direction * sh / 2)
                end = (tx, ty - direction * th / 2)
            axis.add_patch(
                FancyArrowPatch(
                    start,
                    end,
                    arrowstyle="-|>",
                    mutation_scale=10,
                    linewidth=1.0,
                    color="#29465B",
                    connectionstyle="arc3,rad=0.0",
                )
            )
        axis.set_title(title, fontsize=12, weight="bold")

    config = h2.config
    residual_fp = getattr(config, "fp_mode", "full_projection") == (
        "observed_residual"
    )
    fp_nodes: dict[str, dict[str, object]] = {
        "input": node(
            0.055,
            0.5,
            (
                "Observed LF y\n1×231×231"
                if residual_fp
                else f"H1 x1\n{config.depth}×231×231"
            ),
            "input",
            width=0.09,
        ),
        "rearrange": node(
            0.16,
            0.5,
            (
                f"LFToViews S{config.h2_nnum}\n"
                f"{config.num_views}×21×21"
                if residual_fp
                else f"SpaceToViewDepth S{config.h2_nnum}\n"
                f"{config.projected_channels}×21×21"
            ),
            "operation",
            width=0.11,
        ),
        "input_conv": node(
            0.28,
            0.5,
            conv_label("FP input", h2.fp_input),
            "fp",
            width=0.12,
        ),
    }
    fp_edges = [("input", "rearrange"), ("rearrange", "input_conv")]
    fp_x = spaced(len(h2.fp_blocks), 0.405, 0.645)
    previous = "input_conv"
    for index, (block, x) in enumerate(zip(h2.fp_blocks, fp_x, strict=True), 1):
        name = f"block_{index}"
        fp_nodes[name] = node(
            x,
            0.5,
            residual_label(f"FP block {index}", block),
            "fp",
            width=0.105,
        )
        fp_edges.append((previous, name))
        previous = name
    fp_nodes.update(
        {
            "to_views": node(
                0.765,
                0.5,
                conv_label("FP to views", h2.fp_to_views),
                "fp",
                width=0.105,
            ),
            "positive": node(
                0.865,
                0.5,
                f"Softplus β={config.fp_output_softplus_beta:g}\n"
                + (
                    "positive residual feature"
                    if residual_fp
                    else "enforce positive views"
                ),
                "operation",
                width=0.085,
            ),
            "scale": node(
                0.955,
                0.5,
                (
                    f"Fixed residual scale {config.fp_residual_scale:g}\n"
                    "add observed views\nŷ = y + Δy"
                    if residual_fp
                    else "Learned positive scale\nexp(log sFP)\n→ predicted views"
                ),
                "operation",
                width=0.085,
            ),
        }
    )
    fp_edges.extend(
        [(previous, "to_views"), ("to_views", "positive"), ("positive", "scale")]
    )
    if residual_fp:
        fp_edges.append(("rearrange", "scale"))

    squeeze = h2.view_attention.squeeze
    excite = h2.view_attention.excite
    ratio_mode = type(h2.ratio_norm).__name__
    if config.view_attention_mode == "mean_preserving_bounded":
        deviation = config.view_attention_max_deviation
        weights_label = (
            "Mean-centered tanh\n"
            f"mean=1, range [{1 - deviation:g}, {1 + deviation:g}]\n"
            f"{config.num_views} view weights"
        )
    elif config.view_attention_mode == "psf_energy_prior":
        residual_scale = config.view_attention_residual_scale
        weights_label = (
            "Positive mean-one weights\n"
            f"121 × normalize(wPSF · exp({residual_scale:g} δ))\n"
            "no tanh / hard range"
        )
    else:
        weights_label = (
            f"2 × sigmoid\n{config.num_views} view weights\nrange [0, 2]"
        )
    fixed_psf_attention = (
        config.view_attention_mode == "psf_energy_prior"
        and config.view_attention_residual_scale == 0
    )
    if fixed_psf_attention:
        pool_label = "Fixed PSF component energy\nEquation 13 prior wPSF"
        squeeze_label = "Normalize prior\nsum over 121 views = 1"
        activation_label = "Convert to BP gain\n121 × wPSF; mean = 1"
        excite_label = "Learned residual disabled\nresidual scale = 0"
    elif config.view_attention_mode == "psf_energy_prior":
        pool_label = (
            "Two raw-ratio branches\nGlobal average pool\nGlobal maximum pool"
        )
        squeeze_label = conv_label("Shared residual squeeze", squeeze)
        activation_label = "Elementwise sum of branches\nReLU"
        excite_label = (
            conv_label("Log-weight residual δ", excite)
            + "\ncombine with log wPSF"
        )
    else:
        pool_label = (
            "Two raw-ratio branches\nGlobal average pool\nGlobal maximum pool"
        )
        squeeze_label = conv_label("Shared squeeze per branch", squeeze)
        activation_label = "Elementwise sum of branches\nReLU"
        excite_label = conv_label("View excite", excite)

    attention_nodes = {
        "observed": node(
            0.055,
            0.73,
            f"Observed LF views\n{config.num_views}×21×21\n+ ε",
            "input",
            width=0.095,
        ),
        "predicted": node(
            0.055,
            0.27,
            (
                f"Residual FP denominator\ny + Δy, "
                f"{config.num_views}×21×21\n+ ε"
                if residual_fp
                else f"Predicted views\n{config.num_views}×21×21\n+ ε"
            ),
            "fp",
            width=0.095,
        ),
        "divide": node(
            0.18,
            0.5,
            (
                f"Elementwise division\nr=(y+ε)/(y+Δy+ε)\n"
                f"clamp max={config.ratio_max}"
                if residual_fp
                else f"Elementwise division\nr=(y+ε)/(ŷ+ε)\n"
                f"clamp max={config.ratio_max}"
            ),
            "operation",
            width=0.115,
        ),
        "ratio_path": node(
            0.33,
            0.68,
            f"BP ratio feature\n{ratio_mode}\n{config.num_views}×21×21",
            "operation",
            width=0.115,
        ),
        "pool": node(
            0.33,
            0.27,
            pool_label,
            "attention",
            width=0.115,
        ),
        "squeeze": node(
            0.48,
            0.27,
            squeeze_label,
            "attention",
            width=0.115,
        ),
        "activation": node(
            0.60,
            0.27,
            activation_label,
            "attention",
            width=0.095,
        ),
        "excite": node(
            0.72,
            0.27,
            excite_label,
            "attention",
            width=0.11,
        ),
        "weights": node(
            0.84,
            0.27,
            weights_label,
            "attention",
            width=0.105,
        ),
        "multiply": node(
            0.84,
            0.68,
            "Elementwise multiply\nratio feature × weights",
            "operation",
            width=0.105,
        ),
        "output": node(
            0.955,
            0.68,
            f"Attended ratio\nBP input\n{config.num_views}×21×21",
            "bp",
            width=0.085,
        ),
    }
    attention_edges = [
        ("observed", "divide"),
        ("predicted", "divide"),
        ("divide", "ratio_path"),
        ("divide", "pool"),
        ("pool", "squeeze"),
        ("squeeze", "activation"),
        ("activation", "excite"),
        ("excite", "weights"),
        ("ratio_path", "multiply"),
        ("weights", "multiply"),
        ("multiply", "output"),
    ]
    if fixed_psf_attention:
        # The physical prior is a buffer and is not computed from the sample.
        # Keeping its branch disconnected from division makes the rendered
        # execution diagram match the actual fixed-attention forward pass.
        attention_edges.remove(("divide", "pool"))

    bp_nodes = {
        "input": node(
            0.05,
            0.73,
            f"Attended ratio\n{config.num_views}×21×21",
            "input",
            width=0.085,
        ),
        "bp_input": node(
            0.145,
            0.73,
            conv_label("BP input", h2.bp_input),
            "bp",
            width=0.095,
        ),
    }
    bp_edges = [("input", "bp_input")]
    bp_x = spaced(len(h2.bp_blocks), 0.245, 0.745)
    previous = "bp_input"
    for index, (block, x) in enumerate(zip(h2.bp_blocks, bp_x, strict=True), 1):
        name = f"bp_block_{index}"
        bp_nodes[name] = node(
            x,
            0.73,
            residual_label(f"BP block {index}", block),
            "bp",
            width=0.085,
        )
        bp_edges.append((previous, name))
        previous = name
    bp_nodes.update(
        {
            "bp_output": node(
                0.855,
                0.73,
                conv_label("BP output", h2.bp_output),
                "bp",
                width=0.095,
            ),
            "shuffle": node(
                0.955,
                0.73,
                f"ViewDepthToSpace S{config.h2_nnum}⁻¹\n"
                f"{config.depth}×231×231",
                "operation",
                width=0.085,
            ),
        }
    )
    bp_edges.extend([(previous, "bp_output"), ("bp_output", "shuffle")])
    volume_x = spaced(len(h2.volume_blocks), 0.79, 0.53)
    previous = "shuffle"
    for index, (block, x) in enumerate(
        zip(h2.volume_blocks, volume_x, strict=True), 1
    ):
        name = f"volume_block_{index}"
        bp_nodes[name] = node(
            x,
            0.25,
            residual_label(f"Volume block {index}", block),
            "bp",
            width=0.105,
        )
        bp_edges.append((previous, name))
        previous = name
    detail_enabled = config.use_additive_detail_path
    correction_y = 0.39 if detail_enabled else 0.25
    bp_nodes.update(
        {
            "head": node(
                0.40,
                correction_y,
                conv_label("Correction head", h2.correction_head),
                "bp",
                width=0.105,
                height=0.27 if detail_enabled else 0.38,
            ),
            "positive": node(
                0.27,
                correction_y,
                f"Softplus β={config.correction_softplus_beta:g}\n"
                "÷ Softplus(0)\nidentity c2=1 initially",
                "operation",
                width=0.11,
                height=0.27 if detail_enabled else 0.38,
            ),
            "multiply": node(
                0.145,
                correction_y,
                "Explicit RL update\nx2,RL = x1 ⊙ c2",
                "operation",
                width=0.105,
                height=0.27 if detail_enabled else 0.38,
            ),
            "output": node(
                0.045,
                0.25,
                f"H2 proposal x2\n{config.depth}×231×231",
                "input",
                width=0.08,
                height=0.27 if detail_enabled else 0.38,
            ),
        }
    )
    bp_edges.extend(
        [
            (previous, "head"),
            ("head", "positive"),
            ("positive", "multiply"),
        ]
    )
    if detail_enabled:
        bp_nodes.update(
            {
                "detail_head": node(
                    0.40,
                    0.10,
                    conv_label("Additive detail head", h2.additive_detail_head),
                    "fusion",
                    width=0.105,
                    height=0.27,
                ),
                "detail_residual": node(
                    0.27,
                    0.10,
                    f"Centered Softplus β="
                    f"{config.additive_detail_softplus_beta:g}\n"
                    f"× α={config.additive_detail_scale:g}\n"
                    "zero initially",
                    "operation",
                    width=0.11,
                    height=0.27,
                ),
                "detail_add": node(
                    0.145,
                    0.10,
                    "Add detail\nx2,RL + αΔHF\nclamp minimum to 0",
                    "operation",
                    width=0.105,
                    height=0.27,
                ),
            }
        )
        bp_edges.extend(
            [
                (previous, "detail_head"),
                ("detail_head", "detail_residual"),
                ("detail_residual", "detail_add"),
                ("multiply", "detail_add"),
                ("detail_add", "output"),
            ]
        )
    else:
        bp_edges.append(("multiply", "output"))

    core_figure, core_axes = plt.subplots(
        3, 1, figsize=(23, 12), layout="constrained"
    )
    draw_graph(
        core_axes[0],
        fp_nodes,
        fp_edges,
        (
            "H2 observed-residual forward projector (FP)"
            if residual_fp
            else "H2 learned full forward projector (FP)"
        ),
    )
    draw_graph(
        core_axes[1],
        attention_nodes,
        attention_edges,
        "H2 division and dual-pool view attention",
    )
    draw_graph(
        core_axes[2],
        bp_nodes,
        bp_edges,
        (
            "H2 BP, RL update, and additive detail path"
            if detail_enabled
            else "H2 backward projector (BP), volume refinement, and RL update"
        ),
    )

    exemplar = h2.bp_blocks[0]
    primitive_nodes = {
        "input": node(0.08, 0.5, "Input feature u", "input", width=0.10),
        "depthwise": node(
            0.27,
            0.5,
            conv_label("Depthwise spatial filter", exemplar.depthwise),
            "bp",
            width=0.15,
        ),
        "activation": node(
            0.45,
            0.5,
            type(exemplar.activation).__name__,
            "operation",
            width=0.10,
        ),
        "pointwise": node(
            0.62,
            0.5,
            conv_label("Pointwise channel mixer", exemplar.pointwise),
            "bp",
            width=0.15,
        ),
        "add": node(
            0.80,
            0.5,
            "Elementwise add\nu + residual",
            "operation",
            width=0.12,
        ),
        "output": node(0.94, 0.5, "Output feature", "input", width=0.09),
    }
    primitive_edges = [
        ("input", "depthwise"),
        ("depthwise", "activation"),
        ("activation", "pointwise"),
        ("pointwise", "add"),
        ("add", "output"),
    ]

    fusion_config = fusion.config
    fusion_input_names = (
        "H1, H2, c2"
        if fusion_config.include_h2_correction_input
        else "H1, H2"
    )
    fusion_input_channels = (
        3 * fusion_config.depth
        if fusion_config.include_h2_correction_input
        else 2 * fusion_config.depth
    )
    fusion_nodes = {
        "inputs": node(
            0.06,
            0.5,
            f"Concatenate\n{fusion_input_names}\n{fusion_input_channels} channels",
            "input",
            width=0.105,
        ),
        "projection": node(
            0.18,
            0.5,
            conv_label("Fusion input", fusion.input_projection),
            "fusion",
            width=0.11,
        ),
        "activation": node(
            0.29,
            0.5,
            type(fusion.activation).__name__,
            "operation",
            width=0.08,
        ),
    }
    fusion_edges = [("inputs", "projection"), ("projection", "activation")]
    previous = "activation"
    fusion_x = spaced(len(fusion.blocks), 0.40, 0.62)
    for index, (block, x) in enumerate(
        zip(fusion.blocks, fusion_x, strict=True), 1
    ):
        name = f"fusion_block_{index}"
        fusion_nodes[name] = node(
            x,
            0.5,
            residual_label(f"Fusion block {index}", block),
            "fusion",
            width=0.10,
        )
        fusion_edges.append((previous, name))
        previous = name
    fusion_nodes.update(
        {
            "gate": node(
                0.75,
                0.72 if fusion_config.use_additive_residual else 0.5,
                conv_label("Gate head", fusion.gate_head)
                + "\nSigmoid g∈[0,1]",
                "fusion",
                width=0.11,
            ),
            "base": node(
                0.88,
                0.72 if fusion_config.use_additive_residual else 0.5,
                "Gated base\n(1−g)⊙H1 + g⊙H2",
                "operation",
                width=0.12,
            ),
            "output": node(
                0.975,
                0.5,
                f"Final volume\n{fusion_config.depth}×231×231",
                "input",
                width=0.075,
            ),
        }
    )
    if fusion_config.use_additive_residual:
        assert fusion.residual_head is not None
        fusion_nodes.update(
            {
                "residual": node(
                    0.75,
                    0.28,
                    conv_label("Residual head", fusion.residual_head)
                    + f"\ncentered Softplus × {fusion_config.residual_scale:g}",
                    "fusion",
                    width=0.12,
                ),
                "merge": node(
                    0.88,
                    0.28,
                    "Add base + RF\nclamp minimum to 0",
                    "operation",
                    width=0.12,
                ),
            }
        )
        fusion_edges.extend(
            [
                (previous, "gate"),
                (previous, "residual"),
                ("gate", "base"),
                ("base", "merge"),
                ("residual", "merge"),
                ("merge", "output"),
            ]
        )
    else:
        fusion_edges.extend(
            [
                (previous, "gate"),
                ("gate", "base"),
                ("base", "output"),
            ]
        )

    detail_figure, detail_axes = plt.subplots(
        2, 1, figsize=(23, 7.5), layout="constrained"
    )
    draw_graph(
        detail_axes[0],
        primitive_nodes,
        primitive_edges,
        "Depthwise-separable residual block used by H2",
    )
    detail_axes[0].add_patch(
        FancyArrowPatch(
            (0.08, 0.30),
            (0.80, 0.30),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color="#29465B",
            connectionstyle="arc3,rad=0.18",
        )
    )
    detail_axes[0].text(
        0.44,
        0.12,
        "identity skip",
        ha="center",
        va="center",
        fontsize=7.6,
        color="#29465B",
    )
    draw_graph(
        detail_axes[1],
        fusion_nodes,
        fusion_edges,
        (
            "Legacy H1/H2 fusion with additive residual"
            if fusion_config.use_additive_residual
            else "Final H1/H2-only gated fusion"
        ),
    )
    return core_figure, detail_figure


def plot_training_curves(history: Sequence[Mapping[str, Any]]) -> Figure:
    """Plot scheduled, supervised, domain, and learning-rate curves."""
    if not history:
        raise ValueError("Training history is empty")

    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(1, 4, figsize=(19, 4))

    def values(split: str, name: str) -> list[float]:
        return [float(row[split].get(name, 0.0)) for row in history]

    def positive(series: list[float]) -> list[float]:
        return [max(value, 1e-12) for value in series]

    for split, color in (("train", "tab:blue"), ("validation", "tab:orange")):
        axes[0].semilogy(
            epochs,
            positive(values(split, "total")),
            color=color,
            label=split,
        )

    for split, color in (("train", "tab:blue"), ("validation", "tab:orange")):
        supervised = [
            total - feedback
            for total, feedback in zip(
                values(split, "total"), values(split, "feedback"), strict=True
            )
        ]
        axes[1].semilogy(
            epochs, positive(supervised), color=color, label=split
        )

    metric_names = set(history[0]["train"]).intersection(
        history[0]["validation"]
    )
    domain_series = {
        "raw": sorted(name for name in metric_names if name.startswith("raw_")),
        "cutoff": sorted(
            name for name in metric_names if name.startswith("cutoff_")
        ),
        "feedback": ["feedback"] if "feedback" in metric_names else [],
    }
    styles = {"raw": "-", "cutoff": "--", "feedback": ":"}
    for split, color in (("train", "tab:blue"), ("validation", "tab:orange")):
        for domain, names in domain_series.items():
            if not names:
                continue
            series = [
                sum(float(row[split].get(name, 0.0)) for name in names)
                for row in history
            ]
            if domain == "feedback" and not any(value > 0 for value in series):
                continue
            axes[2].semilogy(
                epochs,
                positive(series),
                color=color,
                linestyle=styles[domain],
                label=f"{split} {domain}",
            )

    axes[3].semilogy(
        epochs,
        [float(row["learning_rate"]) for row in history],
        color="tab:green",
    )

    axes[0].set_title("Scheduled optimization loss")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Fixed supervised loss")
    axes[1].set_ylabel("Loss")
    axes[2].set_title("Domain loss totals")
    axes[2].set_ylabel("Loss")
    axes[3].set_title("Learning rate")
    axes[3].set_ylabel("Learning rate")
    axes[2].set_ylabel("Learning rate")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].legend()
    if axes[2].lines:
        axes[2].legend(fontsize="small", ncols=2)
    figure.tight_layout()
    return figure


def _volume_to_cpu(volume: torch.Tensor, name: str) -> torch.Tensor:
    if volume.ndim != 3:
        raise ValueError(f"{name} must have shape [D, H, W], got {tuple(volume.shape)}")
    return volume.detach().float().cpu()


def _lf_to_cpu(lf: torch.Tensor, name: str) -> torch.Tensor:
    if lf.ndim == 3 and lf.shape[0] == 1:
        lf = lf[0]
    if lf.ndim != 2:
        raise ValueError(
            f"{name} must have shape [H, W] or [1, H, W], got {tuple(lf.shape)}"
        )
    return lf.detach().float().cpu()



def safe_quantile(value: torch.Tensor, q: float) -> torch.Tensor:
    """``torch.quantile`` without its 2**24 element limit.

    ``torch.quantile`` raises "input tensor is too large" above 2**24 elements,
    which a 91 x 517 x 517 volume (24.3 M) exceeds while a 231 x 231 one
    (4.9 M) does not -- so display code silently works at one frame size and
    fails at the other. ``kthvalue`` has no such limit and returns the same
    order statistic; it differs from ``torch.quantile`` only in not
    interpolating between neighbouring ranks, which is immaterial for display
    ranges and normalisation.
    """
    flat = value.detach().reshape(-1)
    if flat.numel() == 0:
        raise ValueError("safe_quantile requires a non-empty tensor")
    if not 0.0 <= float(q) <= 1.0:
        raise ValueError("q must lie in [0, 1]")
    flat = flat.float()
    if flat.numel() <= 2 ** 24:
        return torch.quantile(flat, float(q))
    rank = min(max(int(round(float(q) * (flat.numel() - 1))) + 1, 1), flat.numel())
    return flat.kthvalue(rank).values


def plot_lf_noise_comparison(
    clean_lf: torch.Tensor,
    noisy_lf: torch.Tensor,
    *,
    title: str = "LF input comparison",
    display_quantile: float = 0.999,
    zoom_size: int | None = None,
    zoom_center: tuple[int, int] | None = None,
) -> Figure:
    """Compare clean/noisy LF inputs on one intensity scale plus their residual.

    The shared percentile prevents a few hot pixels from making both LF images
    appear black. This affects display only; neither tensor is normalized or
    changed. The residual uses its own symmetric scale. An optional second
    row shows the same ROI without interpolation, using the full-frame scales.
    ``zoom_center`` is (y, x); otherwise the peak of the CLEAN image is used,
    so selecting the ROI never depends on a noise draw. Coordinates refer to
    the network LF grid, which may differ from the native camera grid.
    """
    if not 0.0 < display_quantile <= 1.0:
        raise ValueError("display_quantile must be in (0, 1]")
    clean = _lf_to_cpu(clean_lf, "clean_lf")
    noisy = _lf_to_cpu(noisy_lf, "noisy_lf")
    if clean.shape != noisy.shape:
        raise ValueError(
            "Clean and noisy LF shapes differ: "
            f"{tuple(clean.shape)} vs {tuple(noisy.shape)}"
        )
    if not bool(torch.isfinite(clean).all() and torch.isfinite(noisy).all()):
        raise ValueError("LF comparison requires finite inputs")
    if zoom_size is not None and zoom_size < 1:
        raise ValueError("zoom_size must be positive")

    shared_values = torch.cat((clean.flatten(), noisy.flatten()))
    shared_max = float(safe_quantile(shared_values, display_quantile))
    if not torch.isfinite(torch.tensor(shared_max)) or shared_max <= 0:
        shared_max = max(float(shared_values.max()), 1e-12)
    residual = noisy - clean
    residual_limit = float(
        safe_quantile(residual.abs(), display_quantile)
    )
    if not torch.isfinite(torch.tensor(residual_limit)) or residual_limit <= 0:
        residual_limit = max(float(residual.abs().max()), 1e-12)

    rows = 1 if zoom_size is None else 2
    figure, grid_axes = plt.subplots(
        rows, 3, figsize=(15, 4.5 * rows), layout="constrained", squeeze=False
    )
    axes = grid_axes[0]
    clean_image = axes[0].imshow(
        clean.numpy(), cmap="magma", origin="lower", vmin=0, vmax=shared_max,
        interpolation="nearest",
    )
    axes[1].imshow(
        noisy.numpy(), cmap="magma", origin="lower", vmin=0, vmax=shared_max,
        interpolation="nearest",
    )
    residual_image = axes[2].imshow(
        residual.numpy(),
        cmap="coolwarm",
        origin="lower",
        vmin=-residual_limit,
        vmax=residual_limit,
        interpolation="nearest",
    )
    axes[0].set_title("Clean LF")
    axes[1].set_title("Noisy LF")
    axes[2].set_title("Noisy - clean")
    if zoom_size is not None:
        height, width = clean.shape
        cy, cx = zoom_center if zoom_center is not None else divmod(int(clean.argmax()), width)
        if not (0 <= cy < height and 0 <= cx < width):
            plt.close(figure)
            raise ValueError("zoom_center must lie within the LF image")
        roi_h, roi_w = min(zoom_size, height), min(zoom_size, width)
        y0 = max(0, min(int(cy) - roi_h // 2, height - roi_h))
        x0 = max(0, min(int(cx) - roi_w // 2, width - roi_w))
        extent = (x0 - .5, x0 + roi_w - .5, y0 - .5, y0 + roi_h - .5)
        for column, values in enumerate((clean, noisy, residual)):
            axes[column].add_patch(Rectangle(
                (x0 - .5, y0 - .5), roi_w, roi_h, fill=False,
                edgecolor="#36d5c3", linewidth=1.2,
            ))
            grid_axes[1, column].imshow(
                values[y0:y0 + roi_h, x0:x0 + roi_w].numpy(),
                origin="lower", interpolation="nearest", extent=extent,
                cmap="coolwarm" if column == 2 else "magma",
                vmin=-residual_limit if column == 2 else 0,
                vmax=residual_limit if column == 2 else shared_max,
            )
            grid_axes[1, column].set_title(
                f"{('Clean', 'Noisy', 'Difference')[column]} zoom ({roi_h} x {roi_w})"
            )
    for axis in grid_axes.flat:
        axis.set_xlabel("LF input x (pixel)")
        axis.set_ylabel("LF input y (pixel)")
    figure.colorbar(
        clean_image,
        ax=list(grid_axes[:, :2].flat),
        label=f"LF intensity (shared q={display_quantile:g})",
    )
    figure.colorbar(residual_image, ax=list(grid_axes[:, 2]), label="LF intensity difference")
    figure.suptitle(title)
    return figure


def _orthogonal_projection_layout(
    shape: tuple[int, int, int],
    spacing_um: tuple[float, float, float],
) -> list[
    tuple[str, int, bool, tuple[float, float, float, float], str, str]
]:
    if len(spacing_um) != 3 or any(value <= 0 for value in spacing_um):
        raise ValueError("spacing_um must contain positive (z, y, x) values")
    dz, dy, dx = spacing_um
    depth, height, width = shape
    x_extent = (-width * dx / 2, width * dx / 2)
    y_extent = (-height * dy / 2, height * dy / 2)
    z_extent = (-depth * dz / 2, depth * dz / 2)
    return [
        ("XY", 0, False, (*x_extent, *y_extent), "x (µm)", "y (µm)"),
        # YZ is transposed so its vertical y-axis aligns with XY when the
        # panel is placed to the right. Its horizontal z-axis then aligns
        # with the vertical z-axis of XZ around the shared XY corner.
        ("YZ", 2, True, (*z_extent, *y_extent), "z (µm)", "y (µm)"),
        ("XZ", 1, False, (*x_extent, *z_extent), "x (µm)", "z (µm)"),
    ]


def _draw_mip(
    axis: Any,
    mip: torch.Tensor,
    *,
    title: str,
    extent: tuple[float, float, float, float],
    xlabel: str,
    ylabel: str,
    value_min: float,
    value_max: float,
) -> None:
    axis.imshow(
        mip.numpy(),
        cmap="magma",
        origin="lower",
        extent=extent,
        aspect="equal",
        vmin=value_min,
        vmax=value_max,
    )
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)


def plot_orthogonal_mips(
    volume: torch.Tensor,
    *,
    title: str = "Orthogonal maximum-intensity projections",
    spacing_um: tuple[float, float, float] = (
        0.5,
        100.0 / 11.0 / (40.0 * 80.0 / 75.0),
        100.0 / 11.0 / (40.0 * 80.0 / 75.0),
    ),
    value_range: tuple[float, float] | None = None,
) -> Figure:
    """Show one volume in its spatially related ``XY YZ / XZ`` layout."""
    volume = _volume_to_cpu(volume, "volume")
    projection_layout = _orthogonal_projection_layout(
        tuple(volume.shape), spacing_um
    )
    if value_range is None:
        value_min, value_max = float(volume.min()), float(volume.max())
    else:
        value_min, value_max = value_range
        if value_max < value_min:
            raise ValueError("value_range must be ordered (minimum, maximum)")

    figure = plt.figure(figsize=(9, 8), layout="constrained")
    grid = figure.add_gridspec(2, 2)
    positions = ((0, 0), (0, 1), (1, 0))
    for (name, dimension, transpose, extent, xlabel, ylabel), position in zip(
        projection_layout, positions, strict=True
    ):
        axis = figure.add_subplot(grid[position])
        mip = volume.amax(dim=dimension)
        if transpose:
            mip = mip.transpose(0, 1)
        _draw_mip(
            axis,
            mip,
            title=f"{name} MIP",
            extent=extent,
            xlabel=xlabel,
            ylabel=ylabel,
            value_min=value_min,
            value_max=value_max,
        )
    figure.suptitle(title)
    return figure


def plot_volume_mips(
    ground_truth: torch.Tensor,
    prediction: torch.Tensor,
    *,
    epoch: int | None = None,
    spacing_um: tuple[float, float, float] = (
        0.5,
        100.0 / 11.0 / (40.0 * 80.0 / 75.0),
        100.0 / 11.0 / (40.0 * 80.0 / 75.0),
    ),
) -> Figure:
    """Compare orthogonal MIPs with one common physical micrometre scale.

    ``spacing_um`` follows volume order ``(z, y, x)``. X therefore has the
    same displayed length in XY and XZ, while Z retains its coarser sampling.
    """
    ground_truth = _volume_to_cpu(ground_truth, "ground_truth")
    prediction = _volume_to_cpu(prediction, "prediction")
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            "Prediction and ground-truth shapes differ: "
            f"{tuple(prediction.shape)} vs {tuple(ground_truth.shape)}"
        )
    projection_layout = _orthogonal_projection_layout(
        tuple(ground_truth.shape), spacing_um
    )
    figure = plt.figure(figsize=(17, 8), layout="constrained")
    grid = figure.add_gridspec(2, 4)
    block_positions = (
        ((0, 0), (0, 1), (1, 0)),
        ((0, 2), (0, 3), (1, 2)),
    )
    projections = []
    for name, dimension, transpose, extent, xlabel, ylabel in projection_layout:
        gt_mip = ground_truth.amax(dim=dimension)
        pred_mip = prediction.amax(dim=dimension)
        if transpose:
            gt_mip = gt_mip.transpose(0, 1)
            pred_mip = pred_mip.transpose(0, 1)
        projections.append(
            (
                name,
                extent,
                xlabel,
                ylabel,
                gt_mip,
                pred_mip,
                min(float(gt_mip.min()), float(pred_mip.min())),
                max(float(gt_mip.max()), float(pred_mip.max())),
            )
        )

    for block_index, prefix in enumerate(("GT", "Prediction")):
        for projection_index, projection in enumerate(projections):
            name, extent, xlabel, ylabel, gt_mip, pred_mip, value_min, value_max = (
                projection
            )
            mip = gt_mip if block_index == 0 else pred_mip
            axis = figure.add_subplot(
                grid[block_positions[block_index][projection_index]]
            )
            _draw_mip(
                axis,
                mip,
                title=f"{prefix} — {name} MIP",
                extent=extent,
                xlabel=xlabel,
                ylabel=ylabel,
                value_min=value_min,
                value_max=value_max,
            )

    title = "Validation reconstruction"
    if epoch is not None:
        title += f" — epoch {epoch}"
    figure.suptitle(title)
    return figure


def plot_volume_noise_comparison(
    ground_truth: torch.Tensor,
    clean_prediction: torch.Tensor,
    noisy_prediction: torch.Tensor,
    *,
    title: str = "Matched reconstruction: clean vs noisy input",
    spacing_um: tuple[float, float, float] = (0.5, 100 / 11 / (40 * 80 / 75), 100 / 11 / (40 * 80 / 75)),
) -> Figure:
    """GT and two outputs of the SAME model, on one amplitude/physical scale."""
    volumes = [
        _volume_to_cpu(value, name) for name, value in (
            ("ground_truth", ground_truth), ("clean_prediction", clean_prediction),
            ("noisy_prediction", noisy_prediction),
        )
    ]
    shape = tuple(volumes[0].shape)
    if any(tuple(value.shape) != shape for value in volumes):
        raise ValueError("GT and clean/noisy predictions must have matching shapes")
    layout = _orthogonal_projection_layout(shape, spacing_um)
    lower = min(float(value.min()) for value in volumes)
    upper = max(float(value.max()) for value in volumes)
    figure = plt.figure(figsize=(21, 7), layout="constrained")
    grid = figure.add_gridspec(2, 6)
    for block, (prefix, volume) in enumerate(zip(
        ("GT", "Output from CLEAN LF", "Output from NOISY LF"), volumes, strict=True
    )):
        for position, (name, dim, transpose, extent, xlabel, ylabel) in zip(
            ((0, 2 * block), (0, 2 * block + 1), (1, 2 * block)), layout, strict=True
        ):
            mip = volume.amax(dim=dim)
            if transpose:
                mip = mip.transpose(0, 1)
            _draw_mip(
                figure.add_subplot(grid[position]), mip, title=f"{prefix} - {name}",
                extent=extent, xlabel=xlabel, ylabel=ylabel, value_min=lower, value_max=upper,
            )
    figure.suptitle(title + f"\nAll panels share intensity range [{lower:.3g}, {upper:.3g}]")
    return figure


def plot_h1_h2_stages(
    lf: torch.Tensor,
    raw_target: torch.Tensor,
    h1: torch.Tensor,
    h2: torch.Tensor,
    final: torch.Tensor,
    *,
    title: str = "Connected H1 → H2 → fusion reconstruction",
    spacing_um: tuple[float, float, float] = (
        0.5,
        100.0 / 11.0 / (40.0 * 80.0 / 75.0),
        100.0 / 11.0 / (40.0 * 80.0 / 75.0),
    ),
    scale_mode: Literal["shared", "domain", "per_volume"] = "shared",
) -> Figure:
    """Show LF input and four volume stages in aligned ``XY YZ / XZ`` blocks.

    ``shared`` uses one range for every volume and therefore preserves direct
    amplitude comparisons. ``domain`` gives the cutoff-domain H1 output its
    own range while Raw GT, H2, and final share a raw-domain range. This makes
    H1 structure visible without implying that cutoff and raw-domain peak
    intensities are equal. ``per_volume`` is a structure-only view in which
    every volume has an independent range.
    """
    lf_image = lf.detach().float().cpu().squeeze()
    if lf_image.ndim != 2:
        raise ValueError(f"Expected a single LF image, got {tuple(lf.shape)}")
    named_volumes = (
        ("Raw GT", _volume_to_cpu(raw_target, "raw_target")),
        ("H1 cutoff", _volume_to_cpu(h1, "h1")),
        ("H2 proposal", _volume_to_cpu(h2, "h2")),
        ("Final merge", _volume_to_cpu(final, "final")),
    )
    target_shape = tuple(named_volumes[0][1].shape)
    if any(tuple(volume.shape) != target_shape for _, volume in named_volumes):
        raise ValueError("Raw target, H1, H2, and final volumes must match")
    if scale_mode not in {"shared", "domain", "per_volume"}:
        raise ValueError(
            "scale_mode must be 'shared', 'domain', or 'per_volume', "
            f"got {scale_mode!r}"
        )

    projection_layout = _orthogonal_projection_layout(target_shape, spacing_um)
    shared_range = (
        min(float(volume.min()) for _, volume in named_volumes),
        max(float(volume.max()) for _, volume in named_volumes),
    )
    raw_domain_volumes = (
        named_volumes[0][1], named_volumes[2][1], named_volumes[3][1]
    )
    raw_domain_range = (
        min(float(volume.min()) for volume in raw_domain_volumes),
        max(float(volume.max()) for volume in raw_domain_volumes),
    )
    volume_ranges: dict[str, tuple[float, float]] = {}
    for prefix, volume in named_volumes:
        if scale_mode == "shared":
            volume_ranges[prefix] = shared_range
        elif scale_mode == "domain" and prefix != "H1 cutoff":
            volume_ranges[prefix] = raw_domain_range
        else:
            volume_ranges[prefix] = (float(volume.min()), float(volume.max()))
    figure = plt.figure(figsize=(24, 8), layout="constrained")
    grid = figure.add_gridspec(2, 9, width_ratios=(1.2, *([1.0] * 8)))

    lf_axis = figure.add_subplot(grid[:, 0])
    lf_axis.imshow(lf_image, cmap="magma", origin="lower")
    lf_axis.set_title("Recorded LF input")
    lf_axis.set_xlabel("X pixel")
    lf_axis.set_ylabel("Y pixel")

    for block_index, (prefix, volume) in enumerate(named_volumes):
        value_min, value_max = volume_ranges[prefix]
        display_prefix = prefix
        if scale_mode == "domain" and prefix == "H1 cutoff":
            display_prefix += " [own scale]"
        elif scale_mode == "per_volume":
            display_prefix += f" [own scale, max={value_max:.3g}]"
        start = 1 + 2 * block_index
        positions = ((0, start), (0, start + 1), (1, start))
        for (name, dimension, transpose, extent, xlabel, ylabel), position in zip(
            projection_layout, positions, strict=True
        ):
            mip = volume.amax(dim=dimension)
            if transpose:
                mip = mip.transpose(0, 1)
            axis = figure.add_subplot(grid[position])
            _draw_mip(
                axis,
                mip,
                title=f"{display_prefix} — {name}",
                extent=extent,
                xlabel=xlabel,
                ylabel=ylabel,
                value_min=value_min,
                value_max=value_max,
            )
        unused_axis = figure.add_subplot(grid[1, start + 1])
        unused_axis.axis("off")

    if scale_mode == "domain":
        h1_max = volume_ranges["H1 cutoff"][1]
        title += (
            f"\nH1 uses an independent cutoff-domain scale (max={h1_max:.3g}); "
            "Raw GT, H2, and final share the raw-domain scale."
        )
    elif scale_mode == "per_volume":
        title += "\nEach volume uses an independent structure-only display scale."
    figure.suptitle(title)
    return figure


def plot_experimental_h1_h2_stages(
    lf: torch.Tensor,
    h1: torch.Tensor,
    h2: torch.Tensor,
    final: torch.Tensor,
    *,
    title: str = "Experimental LF reconstruction",
    spacing_um: tuple[float, float, float] = (
        0.5,
        100.0 / 11.0 / (40.0 * 80.0 / 75.0),
        100.0 / 11.0 / (40.0 * 80.0 / 75.0),
    ),
    scale_mode: Literal["shared", "domain", "per_volume"] = "domain",
) -> Figure:
    """Show an experimental LF and H1/H2/final stages without inventing GT.

    In ``domain`` mode H1 has an independent cutoff-domain intensity range,
    while H2 and final share a raw-domain range. This is intended for
    qualitative real-data inference where no ground-truth volume exists.
    """
    lf_image = lf.detach().float().cpu().squeeze()
    if lf_image.ndim != 2:
        raise ValueError(f"Expected a single LF image, got {tuple(lf.shape)}")
    named_volumes = (
        ("H1 cutoff", _volume_to_cpu(h1, "h1")),
        ("H2 proposal", _volume_to_cpu(h2, "h2")),
        ("Final merge", _volume_to_cpu(final, "final")),
    )
    target_shape = tuple(named_volumes[0][1].shape)
    if any(tuple(volume.shape) != target_shape for _, volume in named_volumes):
        raise ValueError("H1, H2, and final volumes must have matching shapes")
    if scale_mode not in {"shared", "domain", "per_volume"}:
        raise ValueError(
            "scale_mode must be 'shared', 'domain', or 'per_volume', "
            f"got {scale_mode!r}"
        )

    projection_layout = _orthogonal_projection_layout(target_shape, spacing_um)
    shared_range = (
        min(float(volume.min()) for _, volume in named_volumes),
        max(float(volume.max()) for _, volume in named_volumes),
    )
    raw_range = (
        min(float(volume.min()) for _, volume in named_volumes[1:]),
        max(float(volume.max()) for _, volume in named_volumes[1:]),
    )
    volume_ranges: dict[str, tuple[float, float]] = {}
    for prefix, volume in named_volumes:
        if scale_mode == "shared":
            volume_ranges[prefix] = shared_range
        elif scale_mode == "domain" and prefix != "H1 cutoff":
            volume_ranges[prefix] = raw_range
        else:
            volume_ranges[prefix] = (float(volume.min()), float(volume.max()))

    figure = plt.figure(figsize=(19, 8), layout="constrained")
    grid = figure.add_gridspec(2, 7, width_ratios=(1.2, *([1.0] * 6)))
    lf_axis = figure.add_subplot(grid[:, 0])
    lf_axis.imshow(lf_image, cmap="magma", origin="lower")
    lf_axis.set_title("Preprocessed LF input")
    lf_axis.set_xlabel("X pixel")
    lf_axis.set_ylabel("Y pixel")

    for block_index, (prefix, volume) in enumerate(named_volumes):
        value_min, value_max = volume_ranges[prefix]
        display_prefix = prefix
        if scale_mode == "domain" and prefix == "H1 cutoff":
            display_prefix += " [own scale]"
        elif scale_mode == "per_volume":
            display_prefix += f" [own scale, max={value_max:.3g}]"
        start = 1 + 2 * block_index
        positions = ((0, start), (0, start + 1), (1, start))
        for (name, dimension, transpose, extent, xlabel, ylabel), position in zip(
            projection_layout, positions, strict=True
        ):
            mip = volume.amax(dim=dimension)
            if transpose:
                mip = mip.transpose(0, 1)
            axis = figure.add_subplot(grid[position])
            _draw_mip(
                axis,
                mip,
                title=f"{display_prefix} — {name}",
                extent=extent,
                xlabel=xlabel,
                ylabel=ylabel,
                value_min=value_min,
                value_max=value_max,
            )
        figure.add_subplot(grid[1, start + 1]).axis("off")

    if scale_mode == "domain":
        title += (
            "\nH1 uses its cutoff-domain scale; H2 and final share a "
            "raw-domain display scale."
        )
    elif scale_mode == "per_volume":
        title += "\nEach stage uses an independent structure-only scale."
    figure.suptitle(title)
    return figure
