from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import tifffile
import torch


ExperimentalPreprocessMode = Literal["manual", "stack_quantile", "raw_counts"]


# Lenslet centre offset within each pitch-sized cell in the training LF.
# Measured from the synthetic validation split (mean, max-projection and
# temporal-standard-deviation estimators all agree to within 0.1 pixel).
TRAINING_LENSLET_OFFSET = (5.0, 5.0)


@dataclass(frozen=True)
class ExperimentalLFPreprocessConfig:
    """Convert an experimental TIFF stack to the fixed network input scale.

    ``stack_quantile`` estimates one background and one scale per TIFF stack;
    it never normalizes individual frames. ``manual`` is preferred when a
    dark-frame background and calibrated count scale are available.

    ``background_quantile`` must sit below the diffuse signal, not in the
    middle of it. The median is not a background estimate: on the current
    experimental stacks it lies roughly twice the true dark floor and hard
    zeros between a quarter and nine tenths of every frame once
    ``clamp_nonnegative`` applies.

    ``target_quantile_value`` is the network-input value that
    ``scale_quantile`` is mapped to. The synthetic training LF has a median
    per-sample q99.9 of about 0.10, so mapping to 1.0 makes the experimental
    input roughly ten times brighter than a typical training sample. That
    matters because the connected checkpoint was trained with
    ``use_ratio_instance_norm=False``, which sends the absolute
    Richardson--Lucy ratio level to the backprojector.

    ``lenslet_offset`` is the measured lenslet centre position within each
    ``h2_nnum``-sized cell of the raw frames. When supplied, preprocessing
    shifts the frame so the lattice matches ``reference_lenslet_offset``.
    ``lf_to_views`` is ``pixel_unshuffle``, which is phase-locked to pixel
    zero, so an unaligned lattice assigns every pixel to the wrong angular
    view. Use :func:`estimate_lenslet_offset` to measure it.
    """

    mode: ExperimentalPreprocessMode = "stack_quantile"
    manual_background_counts: float = 0.0
    manual_count_scale: float = 1.0
    background_quantile: float = 0.25
    scale_quantile: float = 0.999
    target_quantile_value: float = 0.10
    calibration_frame_count: int = 16
    clamp_nonnegative: bool = True
    clip_max: float | None = 1.0
    lenslet_offset: tuple[float, float] | None = None
    reference_lenslet_offset: tuple[float, float] = TRAINING_LENSLET_OFFSET

    def __post_init__(self) -> None:
        if self.mode not in ("manual", "stack_quantile", "raw_counts"):
            raise ValueError(f"Unsupported preprocessing mode {self.mode!r}")
        if self.manual_count_scale <= 0:
            raise ValueError("manual_count_scale must be positive")
        if not 0 <= self.background_quantile < self.scale_quantile <= 1:
            raise ValueError(
                "Require 0 <= background_quantile < scale_quantile <= 1"
            )
        if self.target_quantile_value <= 0:
            raise ValueError("target_quantile_value must be positive")
        if self.calibration_frame_count <= 0:
            raise ValueError("calibration_frame_count must be positive")
        if self.clip_max is not None and self.clip_max <= 0:
            raise ValueError("clip_max must be positive when provided")
        for name in ("lenslet_offset", "reference_lenslet_offset"):
            value = getattr(self, name)
            if value is None:
                continue
            if len(value) != 2 or not all(np.isfinite(v) for v in value):
                raise ValueError(f"{name} must be two finite numbers")


@dataclass(frozen=True)
class ExperimentalLFCalibration:
    mode: ExperimentalPreprocessMode
    background_counts: float
    count_scale: float
    sampled_frame_indices: tuple[int, ...] = ()
    background_quantile: float | None = None
    scale_quantile: float | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.background_counts):
            raise ValueError("background_counts must be finite")
        if not np.isfinite(self.count_scale) or self.count_scale <= 0:
            raise ValueError("count_scale must be finite and positive")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InferenceTiming:
    repetitions: int
    mean_ms: float
    std_ms: float
    median_ms: float
    minimum_ms: float
    p95_ms: float
    frames_per_second: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def inspect_experimental_tiff(path: str | Path) -> dict[str, Any]:
    """Read TIFF metadata without loading the complete stack."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        shape = tuple(int(value) for value in series.shape)
        axes = str(series.axes)
        if len(shape) == 2:
            frames, height, width = 1, *shape
        elif len(shape) == 3:
            frames, height, width = shape
        else:
            raise ValueError(
                f"Expected a 2D image or 3D frame stack, got {shape} ({axes})"
            )
        return {
            "path": str(path.resolve()),
            "name": path.name,
            "shape": shape,
            "axes": axes,
            "dtype": str(series.dtype),
            "frames": frames,
            "height": height,
            "width": width,
            "is_bigtiff": bool(tif.is_bigtiff),
        }


def evenly_spaced_frame_indices(
    frame_count: int,
    requested_count: int,
) -> tuple[int, ...]:
    if frame_count <= 0 or requested_count <= 0:
        raise ValueError("frame_count and requested_count must be positive")
    count = min(frame_count, requested_count)
    indices = np.linspace(0, frame_count - 1, count).round().astype(int)
    return tuple(int(value) for value in np.unique(indices))


def read_experimental_tiff_frame(
    path: str | Path,
    frame_index: int,
) -> torch.Tensor:
    """Load one TIFF page as a count-valued tensor shaped ``[1, H, W]``."""
    path = Path(path)
    with tifffile.TiffFile(path) as tif:
        if not 0 <= frame_index < len(tif.pages):
            raise IndexError(
                f"frame_index {frame_index} outside [0, {len(tif.pages) - 1}]"
            )
        array = tif.pages[frame_index].asarray()
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D TIFF page, got {array.shape}")
    return torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)


def calibrate_experimental_tiff(
    path: str | Path,
    config: ExperimentalLFPreprocessConfig | None = None,
) -> ExperimentalLFCalibration:
    """Estimate one affine count conversion for an entire TIFF stack."""
    config = config or ExperimentalLFPreprocessConfig()
    if config.mode == "manual":
        return ExperimentalLFCalibration(
            mode=config.mode,
            background_counts=config.manual_background_counts,
            count_scale=config.manual_count_scale,
        )
    if config.mode == "raw_counts":
        return ExperimentalLFCalibration(
            mode=config.mode,
            background_counts=0.0,
            count_scale=1.0,
        )

    metadata = inspect_experimental_tiff(path)
    indices = evenly_spaced_frame_indices(
        metadata["frames"], config.calibration_frame_count
    )
    sampled = torch.cat(
        [read_experimental_tiff_frame(path, index).flatten() for index in indices]
    ).to(torch.float32)
    background = float(torch.quantile(sampled, config.background_quantile))
    upper = float(torch.quantile(sampled, config.scale_quantile))
    # Map the scale quantile onto target_quantile_value rather than onto 1.0,
    # so the input lands inside the training LF distribution instead of at its
    # extreme bright edge.
    count_scale = (upper - background) / config.target_quantile_value
    if not np.isfinite(count_scale) or count_scale <= 0:
        raise ValueError(
            "Stack quantiles did not produce a positive count scale: "
            f"background={background:g}, upper={upper:g}"
        )
    return ExperimentalLFCalibration(
        mode=config.mode,
        background_counts=background,
        count_scale=count_scale,
        sampled_frame_indices=indices,
        background_quantile=config.background_quantile,
        scale_quantile=config.scale_quantile,
    )


@dataclass(frozen=True)
class LensletPhase:
    """Measured lenslet lattice position and the confidence of that estimate."""

    offset: tuple[float, float]
    strength: tuple[float, float]
    pitch: int
    sampled_frame_indices: tuple[int, ...] = ()
    estimator: str = "temporal_std"

    @property
    def reliable(self) -> bool:
        """Whether the lattice is visible enough to trust the offset.

        The synthetic training LF scores about 0.15 on both axes. Stacks below
        this threshold do not show a usable lenslet modulation and need a
        flat-field or uniform-dye calibration frame instead.
        """
        return min(self.strength) >= 0.03

    def shift_to(self, reference: tuple[float, float]) -> tuple[int, int]:
        """Integer pixel shift that moves this lattice onto ``reference``."""
        return tuple(  # type: ignore[return-value]
            int(round((measured - target) % self.pitch)) % self.pitch
            for measured, target in zip(self.offset, reference)
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _axis_lattice_phase(
    image: np.ndarray,
    axis: int,
    pitch: int,
) -> tuple[float, float]:
    """Phase and relative strength of the period-``pitch`` component."""
    profile = image.mean(axis=1 - axis)
    size = profile.shape[0]
    if size % pitch:
        raise ValueError(f"Axis length {size} is not divisible by pitch {pitch}")
    profile = profile - profile.mean()
    spectrum = np.fft.rfft(profile)
    fundamental = spectrum[size // pitch]
    offset = float(
        (-np.angle(fundamental) / (2 * np.pi) * pitch) % pitch
    )
    total = float(np.abs(spectrum[1:]).sum())
    strength = float(np.abs(fundamental) / total) if total > 0 else 0.0
    return offset, strength


def estimate_lenslet_offset(
    path: str | Path,
    *,
    pitch: int = 11,
    frame_count: int = 48,
    dead_pixel_quantile: float = 0.9995,
) -> LensletPhase:
    """Measure the lenslet lattice position of an experimental TIFF stack.

    The temporal standard deviation across frames is used rather than the mean:
    it suppresses vignetting and illumination gradients, which otherwise
    dominate the frame average and bias the estimate. Compare the result with
    :data:`TRAINING_LENSLET_OFFSET` and check :attr:`LensletPhase.reliable`
    before applying any correction.
    """
    metadata = inspect_experimental_tiff(path)
    height, width = metadata["height"], metadata["width"]
    usable_h, usable_w = height - height % pitch, width - width % pitch
    indices = evenly_spaced_frame_indices(metadata["frames"], frame_count)

    frames = []
    for index in indices:
        frame = (
            read_experimental_tiff_frame(path, index)[0]
            .to(torch.float32)
            .numpy()
            .astype(np.float64)[:usable_h, :usable_w]
        )
        # Saturated and defective pixels are single-pixel outliers; they carry
        # no lattice information but do skew the per-frame normalization.
        frame = np.minimum(frame, np.quantile(frame, dead_pixel_quantile))
        low = np.quantile(frame, 0.01)
        high = np.quantile(frame, 0.999)
        frames.append((frame - low) / max(high - low, 1e-9))

    lattice = np.stack(frames).std(axis=0)
    offset_y, strength_y = _axis_lattice_phase(lattice, 0, pitch)
    offset_x, strength_x = _axis_lattice_phase(lattice, 1, pitch)
    return LensletPhase(
        offset=(offset_y, offset_x),
        strength=(strength_y, strength_x),
        pitch=pitch,
        sampled_frame_indices=indices,
    )


def align_lenslet_phase(
    image: torch.Tensor,
    shift: tuple[int, int],
    *,
    lenslet_pitch: int,
) -> torch.Tensor:
    """Shift a frame so its lenslet lattice sits on the pixel-zero phase.

    The image is cropped rather than rolled: wrapping would move real edge
    lenslets to the opposite side of the frame. The result is trimmed to a
    whole number of lenslets, so a nonzero shift costs one lenslet row and
    one lenslet column.
    """
    shift_y, shift_x = (int(value) % lenslet_pitch for value in shift)
    height, width = image.shape[-2:]
    usable_h = (height - shift_y) // lenslet_pitch * lenslet_pitch
    usable_w = (width - shift_x) // lenslet_pitch * lenslet_pitch
    if usable_h <= 0 or usable_w <= 0:
        raise ValueError(
            f"Shift {shift} leaves no whole lenslets in {(height, width)}"
        )
    return image[
        ..., shift_y : shift_y + usable_h, shift_x : shift_x + usable_w
    ]


def lenslet_aligned_center_crop(
    image: torch.Tensor,
    crop_shape: tuple[int, int],
    *,
    lenslet_pitch: int,
) -> torch.Tensor:
    """Center-crop while keeping the crop origin on the lenslet lattice."""
    if image.ndim < 2:
        raise ValueError(f"Expected at least two dimensions, got {image.shape}")
    if lenslet_pitch <= 0:
        raise ValueError("lenslet_pitch must be positive")
    crop_height, crop_width = crop_shape
    height, width = image.shape[-2:]
    if crop_height <= 0 or crop_width <= 0:
        raise ValueError("crop dimensions must be positive")
    if crop_height > height or crop_width > width:
        raise ValueError(
            f"Crop {crop_shape} exceeds image dimensions {(height, width)}"
        )
    if crop_height % lenslet_pitch or crop_width % lenslet_pitch:
        raise ValueError(
            f"Crop {crop_shape} must be divisible by lenslet_pitch={lenslet_pitch}"
        )

    def aligned_start(size: int, crop: int) -> int:
        centered = (size - crop) / 2
        start = int(round(centered / lenslet_pitch)) * lenslet_pitch
        maximum = size - crop
        return min(max(start, 0), maximum - maximum % lenslet_pitch)

    top = aligned_start(height, crop_height)
    left = aligned_start(width, crop_width)
    return image[..., top : top + crop_height, left : left + crop_width]


def preprocess_experimental_lf(
    counts: torch.Tensor,
    calibration: ExperimentalLFCalibration,
    config: ExperimentalLFPreprocessConfig | None = None,
    *,
    crop_shape: tuple[int, int] | None = None,
    lenslet_pitch: int = 11,
) -> torch.Tensor:
    """Apply a stack-fixed count conversion and optional aligned crop."""
    config = config or ExperimentalLFPreprocessConfig(mode=calibration.mode)
    if counts.ndim == 2:
        counts = counts.unsqueeze(0)
    if counts.ndim != 3 or counts.shape[0] != 1:
        raise ValueError(f"Expected counts [1, H, W], got {tuple(counts.shape)}")
    output = (
        counts.to(torch.float32) - calibration.background_counts
    ) / calibration.count_scale
    if config.clamp_nonnegative:
        output = output.clamp_min(0)
    if config.clip_max is not None:
        output = output.clamp_max(config.clip_max)
    # Align the lenslet lattice before any view rearrangement. This has to
    # happen before cropping so the crop inherits the corrected phase.
    if config.lenslet_offset is not None:
        shift = tuple(
            int(round((measured - target) % lenslet_pitch)) % lenslet_pitch
            for measured, target in zip(
                config.lenslet_offset, config.reference_lenslet_offset
            )
        )
        output = align_lenslet_phase(
            output, shift, lenslet_pitch=lenslet_pitch  # type: ignore[arg-type]
        )
    if crop_shape is not None:
        output = lenslet_aligned_center_crop(
            output,
            crop_shape,
            lenslet_pitch=lenslet_pitch,
        )
    return output.contiguous()


def validate_lf_shape_for_model(
    lf: torch.Tensor,
    *,
    h1_nnum: int,
    h2_nnum: int,
) -> None:
    """Validate the divisibility required by both H1 and H2 rearrangements."""
    if lf.ndim not in (3, 4):
        raise ValueError(f"Expected [1,H,W] or [B,1,H,W], got {tuple(lf.shape)}")
    if lf.shape[-3] != 1:
        raise ValueError(f"Expected one LF channel, got {tuple(lf.shape)}")
    height, width = lf.shape[-2:]
    for size in (height, width):
        if size * h1_nnum % h2_nnum:
            raise ValueError(
                f"LF size {(height, width)} is incompatible with H1 scale "
                f"{h1_nnum}/{h2_nnum}"
            )
        small = size * h1_nnum // h2_nnum
        if small % h1_nnum:
            raise ValueError(
                f"H1 working-grid size {small} is not divisible by {h1_nnum}"
            )
        if size % h2_nnum:
            raise ValueError(
                f"LF size {(height, width)} is not divisible by H2 nnum={h2_nnum}"
            )


def benchmark_inference(
    function: Callable[[], Any],
    *,
    device: torch.device,
    warmup: int = 3,
    repetitions: int = 10,
) -> InferenceTiming:
    """Benchmark a callable with CUDA events or a CPU high-resolution timer."""
    if warmup < 0 or repetitions <= 0:
        raise ValueError("warmup must be nonnegative and repetitions positive")
    with torch.inference_mode():
        for _ in range(warmup):
            function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        elapsed_ms: list[float] = []
        for _ in range(repetitions):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                function()
                end.record()
                torch.cuda.synchronize(device)
                elapsed_ms.append(float(start.elapsed_time(end)))
            else:
                start_time = time.perf_counter()
                function()
                elapsed_ms.append((time.perf_counter() - start_time) * 1000)

    mean_ms = statistics.fmean(elapsed_ms)
    return InferenceTiming(
        repetitions=repetitions,
        mean_ms=mean_ms,
        std_ms=statistics.pstdev(elapsed_ms),
        median_ms=statistics.median(elapsed_ms),
        minimum_ms=min(elapsed_ms),
        p95_ms=float(np.quantile(elapsed_ms, 0.95)),
        frames_per_second=1000.0 / mean_ms,
    )
