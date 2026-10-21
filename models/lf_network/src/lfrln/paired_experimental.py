from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import re
from typing import Any, Iterable, Literal

import h5py
import numpy as np
import tifffile
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .data import load_lf_tensor


PAIRED_EXPERIMENTAL_SCHEMA = "lfrln.paired_experimental.patch.v1"


def registered_reference_window(
    directory: str | Path, center_index: int, *, prefix: str, depth: int = 91,
) -> tuple[Path, ...]:
    """Resolve an indexed LF's odd-depth window in a zero-numbered TIFF stack.

    The experimental filenames preserve slice numbers: LF 000065 maps to
    slices 020..110 for depth 91. Reject incomplete windows; never pad or shift
    the reference silently. PS and TF calls use their own directories/prefixes.
    """
    if isinstance(depth, bool) or not isinstance(depth, int) or depth <= 0 or depth % 2 != 1:
        raise ValueError("Reference depth must be a positive odd integer")
    if isinstance(center_index, bool) or not isinstance(center_index, int):
        raise ValueError("LF registration index must be an integer")
    first = center_index - depth // 2
    if first < 0:
        raise ValueError(f"LF index {center_index} needs a negative reference slice at depth {depth}")
    paths = tuple(Path(directory) / f"{prefix}{i:03d}.tif"
                  for i in range(first, first + depth))
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete {depth}-slice reference window; first missing: {missing[0]}")
    return paths


def _natural_key(path: Path) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    )


def _files(directory: Path, pattern: str) -> list[Path]:
    return sorted(directory.glob(pattern), key=_natural_key)


@dataclass(frozen=True)
class PairedAcquisitionAudit:
    """File-level audit of one point-scanning/light-field acquisition pair."""

    name: str
    point_scanning_directory: str
    light_field_directory: str
    point_scanning_slices: int
    light_field_frames: int
    point_scanning_shape: tuple[int, ...] | None
    point_scanning_dtype: str | None
    point_scanning_channels_equal: bool | None
    light_field_shape: tuple[int, ...] | None
    light_field_dtype: str | None
    point_scanning_metadata: bool
    light_field_metadata: bool
    status: Literal["candidate", "review", "unusable"]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["reasons"] = "; ".join(self.reasons)
        return row


def discover_paired_acquisition_directories(
    root: str | Path,
) -> list[tuple[Path, Path]]:
    """Find acquisition folders shared by the PS and LF modality trees."""

    root = Path(root)
    ps_root = root / "01_Point Scanning" / "Biosamples"
    lf_root = root / "03_Light Field" / "Biosamples"
    if not ps_root.is_dir() or not lf_root.is_dir():
        raise FileNotFoundError(
            "Expected '01_Point Scanning/Biosamples' and "
            f"'03_Light Field/Biosamples' below {root}"
        )

    ps_directories = {
        directory.name: directory
        for directory in ps_root.iterdir()
        if directory.is_dir() and directory.name != "DB"
    }
    lf_directories = {
        directory.name: directory
        for directory in lf_root.iterdir()
        if directory.is_dir() and directory.name != "DB"
    }
    return [
        (ps_directories[name], lf_directories[name])
        for name in sorted(ps_directories.keys() & lf_directories.keys())
    ]


def audit_paired_acquisition(
    point_scanning_directory: str | Path,
    light_field_directory: str | Path,
    *,
    expected_slices: int = 131,
    expected_shape: tuple[int, int] = (517, 517),
    expected_point_scanning_dtype: str = "uint16",
    lf_variable: str = "LF",
) -> PairedAcquisitionAudit:
    """Audit dimensions, storage format, metadata, and file completeness."""

    ps_directory = Path(point_scanning_directory)
    lf_directory = Path(light_field_directory)
    if ps_directory.name != lf_directory.name:
        raise ValueError(
            "Point-scanning and LF directories must have the same acquisition name"
        )

    ps_paths = _files(ps_directory, "*.tif")
    lf_paths = _files(lf_directory, "*.mat")
    ps_shape: tuple[int, ...] | None = None
    ps_dtype: str | None = None
    channels_equal: bool | None = None
    lf_shape: tuple[int, ...] | None = None
    lf_dtype: str | None = None
    reasons: list[str] = []

    if ps_paths:
        first_ps = tifffile.imread(ps_paths[0])
        ps_shape = tuple(first_ps.shape)
        ps_dtype = str(first_ps.dtype)
        if first_ps.ndim == 3 and first_ps.shape[-1] in (3, 4):
            channels_equal = all(
                np.array_equal(first_ps[..., 0], first_ps[..., channel])
                for channel in range(1, first_ps.shape[-1])
            )
    else:
        reasons.append("missing point-scanning TIFF slices")

    if lf_paths:
        first_lf = load_lf_tensor(lf_paths[0], lf_variable)[0].numpy()
        lf_shape = tuple(first_lf.shape)
        lf_dtype = str(first_lf.dtype)
    else:
        reasons.append("missing LF MAT frames")

    if len(ps_paths) != expected_slices:
        reasons.append(
            f"point-scanning slice count is {len(ps_paths)}, expected {expected_slices}"
        )
    if ps_shape is not None and ps_shape != expected_shape:
        if not (ps_shape == (*expected_shape, 3) and channels_equal is True):
            reasons.append(
                f"point-scanning shape is {ps_shape}, expected {expected_shape}"
            )
    if ps_dtype is not None and ps_dtype != expected_point_scanning_dtype:
        reasons.append(
            f"point-scanning dtype is {ps_dtype}, expected "
            f"{expected_point_scanning_dtype}"
        )
    if ps_shape is not None and len(ps_shape) == 3:
        if channels_equal:
            reasons.append("point-scanning target is duplicated RGB, not native grayscale")
        else:
            reasons.append("point-scanning target has non-identical color channels")
    if lf_shape is not None and lf_shape != expected_shape:
        reasons.append(f"LF shape is {lf_shape}, expected {expected_shape}")

    ps_metadata = (ps_directory / "parameters.txt").is_file()
    lf_metadata = (lf_directory / "parameters.txt").is_file()
    if not ps_metadata:
        reasons.append("missing point-scanning parameters.txt")
    if not lf_metadata:
        reasons.append("missing LF parameters.txt")

    if not ps_paths or not lf_paths:
        status: Literal["candidate", "review", "unusable"] = "unusable"
    elif reasons:
        status = "review"
    else:
        status = "candidate"

    return PairedAcquisitionAudit(
        name=ps_directory.name,
        point_scanning_directory=str(ps_directory),
        light_field_directory=str(lf_directory),
        point_scanning_slices=len(ps_paths),
        light_field_frames=len(lf_paths),
        point_scanning_shape=ps_shape,
        point_scanning_dtype=ps_dtype,
        point_scanning_channels_equal=channels_equal,
        light_field_shape=lf_shape,
        light_field_dtype=lf_dtype,
        point_scanning_metadata=ps_metadata,
        light_field_metadata=lf_metadata,
        status=status,
        reasons=tuple(reasons),
    )


def audit_paired_dataset(
    root: str | Path,
    **audit_kwargs: Any,
) -> list[PairedAcquisitionAudit]:
    """Audit every acquisition name shared by the PS and LF trees."""

    return [
        audit_paired_acquisition(ps_directory, lf_directory, **audit_kwargs)
        for ps_directory, lf_directory in discover_paired_acquisition_directories(root)
    ]


def load_point_scanning_stack(
    directory: str | Path,
    *,
    allow_equal_rgb: bool = False,
    dtype: np.dtype[Any] = np.dtype(np.float32),
) -> np.ndarray:
    """Load a directory of TIFF slices as a ``[Z, Y, X]`` volume.

    Equal-channel RGB files are accepted only when explicitly requested. This
    prevents an 8-bit visualization export from silently becoming a target.
    """

    paths = _files(Path(directory), "*.tif")
    if not paths:
        raise FileNotFoundError(f"No TIFF slices found in {directory}")
    slices = []
    expected_shape: tuple[int, int] | None = None
    for path in paths:
        image = tifffile.imread(path)
        if image.ndim == 3 and image.shape[-1] in (3, 4):
            channels_equal = all(
                np.array_equal(image[..., 0], image[..., channel])
                for channel in range(1, image.shape[-1])
            )
            if not channels_equal:
                raise ValueError(f"Color channels differ in {path}")
            if not allow_equal_rgb:
                raise ValueError(
                    f"{path} is duplicated RGB. Set allow_equal_rgb=True only "
                    "after accepting its reduced bit depth as a target."
                )
            image = image[..., 0]
        if image.ndim != 2:
            raise ValueError(f"Expected a 2D TIFF slice, got {image.shape} in {path}")
        if expected_shape is None:
            expected_shape = tuple(image.shape)
        elif tuple(image.shape) != expected_shape:
            raise ValueError(
                f"Inconsistent slice shape {image.shape} in {path}; "
                f"expected {expected_shape}"
            )
        slices.append(image)
    return np.stack(slices).astype(dtype, copy=False)


def load_light_field_frame(
    path: str | Path,
    *,
    variable_name: str = "LF",
) -> np.ndarray:
    """Load one MATLAB LF frame as a contiguous ``[Y, X]`` float32 array."""

    return np.ascontiguousarray(
        load_lf_tensor(path, variable_name)[0].numpy(), dtype=np.float32
    )


def load_light_field_mean(
    paths: Iterable[str | Path],
    *,
    variable_name: str = "LF",
) -> np.ndarray:
    """Average LF repeats without keeping every full frame in memory."""

    total: np.ndarray | None = None
    count = 0
    for path in paths:
        frame = load_light_field_frame(path, variable_name=variable_name)
        if total is None:
            total = np.zeros_like(frame, dtype=np.float64)
        if frame.shape != total.shape:
            raise ValueError(
                f"Inconsistent LF shape {frame.shape}; expected {total.shape}"
            )
        total += frame
        count += 1
    if total is None or count == 0:
        raise ValueError("At least one LF path is required")
    return (total / count).astype(np.float32)


def block_reduce_2d(
    image: np.ndarray,
    block_shape: tuple[int, int],
    *,
    reduction: Literal["mean", "sum", "max"] = "mean",
) -> np.ndarray:
    """Reduce non-overlapping image blocks without interpolation."""

    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, got {image.shape}")
    by, bx = block_shape
    if by <= 0 or bx <= 0 or image.shape[0] % by or image.shape[1] % bx:
        raise ValueError(
            f"Image shape {image.shape} is not divisible by block shape {block_shape}"
        )
    blocks = image.reshape(image.shape[0] // by, by, image.shape[1] // bx, bx)
    if reduction == "mean":
        return blocks.mean(axis=(1, 3))
    if reduction == "sum":
        return blocks.sum(axis=(1, 3))
    if reduction == "max":
        return blocks.max(axis=(1, 3))
    raise ValueError(f"Unsupported reduction: {reduction}")


def lf_lenslet_projection(
    lf: np.ndarray,
    *,
    pixels_per_lenslet: int = 11,
    reduction: Literal["mean", "sum", "max"] = "sum",
) -> np.ndarray:
    """Collapse angular pixels to one intensity value per lenslet."""

    return block_reduce_2d(
        lf,
        (pixels_per_lenslet, pixels_per_lenslet),
        reduction=reduction,
    )


def _pearson_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first = first - first.mean()
    second = second - second.mean()
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= np.finfo(np.float64).eps:
        return 0.0
    return float(np.sum(first * second) / denominator)


def _orientation_variants(image: np.ndarray) -> Iterable[tuple[str, np.ndarray]]:
    for rotation in range(4):
        rotated = np.rot90(image, rotation)
        yield f"rot{rotation * 90}", rotated
        yield f"rot{rotation * 90}_flip_x", np.fliplr(rotated)


def _overlap_for_shift(
    reference: np.ndarray,
    moving: np.ndarray,
    dy: int,
    dx: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = reference.shape
    ref_y = slice(max(0, dy), min(height, height + dy))
    ref_x = slice(max(0, dx), min(width, width + dx))
    mov_y = slice(max(0, -dy), min(height, height - dy))
    mov_x = slice(max(0, -dx), min(width, width - dx))
    return reference[ref_y, ref_x], moving[mov_y, mov_x]


def best_integer_registration(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    max_shift: int = 4,
) -> dict[str, Any]:
    """Find the best dihedral orientation and integer shift by correlation."""

    reference = np.asarray(reference)
    moving = np.asarray(moving)
    if reference.ndim != 2 or moving.ndim != 2:
        raise ValueError("reference and moving must both be 2D")
    if reference.shape != moving.shape:
        raise ValueError(
            f"reference and moving shapes differ: {reference.shape} vs {moving.shape}"
        )
    if max_shift < 0:
        raise ValueError("max_shift must be non-negative")

    best: dict[str, Any] | None = None
    for orientation, variant in _orientation_variants(moving):
        for dy in range(-max_shift, max_shift + 1):
            for dx in range(-max_shift, max_shift + 1):
                ref_overlap, mov_overlap = _overlap_for_shift(
                    reference, variant, dy, dx
                )
                correlation = _pearson_correlation(ref_overlap, mov_overlap)
                if best is None or correlation > best["correlation"]:
                    best = {
                        "correlation": correlation,
                        "orientation": orientation,
                        "shift_y": dy,
                        "shift_x": dx,
                    }
    assert best is not None
    return best


def paired_spatial_registration_diagnostic(
    point_scanning_volume: np.ndarray,
    light_field: np.ndarray,
    *,
    pixels_per_lenslet: int = 11,
    point_scanning_background_quantile: float = 0.10,
    light_field_background_quantile: float = 0.60,
    max_shift_lenslets: int = 4,
) -> dict[str, Any]:
    """Compare the LF spatial projection with the axial-sum PS projection.

    This inexpensive diagnostic tests lateral orientation and coarse alignment.
    It does not establish axial registration or replace visual inspection.
    """

    volume = np.asarray(point_scanning_volume, dtype=np.float32)
    lf = np.asarray(light_field, dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError(f"Expected PS volume [Z,Y,X], got {volume.shape}")
    if lf.ndim != 2 or tuple(volume.shape[-2:]) != tuple(lf.shape):
        raise ValueError(
            "PS lateral shape and LF shape must match; got "
            f"{volume.shape[-2:]} and {lf.shape}"
        )

    ps_floor = np.quantile(volume, point_scanning_background_quantile)
    lf_floor = np.quantile(lf, light_field_background_quantile)
    ps_foreground = np.maximum(volume - ps_floor, 0.0)
    lf_foreground = np.maximum(lf - lf_floor, 0.0)
    ps_projection = block_reduce_2d(
        ps_foreground.sum(axis=0),
        (pixels_per_lenslet, pixels_per_lenslet),
        reduction="mean",
    )
    lf_projection = lf_lenslet_projection(
        lf_foreground,
        pixels_per_lenslet=pixels_per_lenslet,
        reduction="sum",
    )
    registration = best_integer_registration(
        lf_projection,
        ps_projection,
        max_shift=max_shift_lenslets,
    )
    return {
        **registration,
        "light_field_projection": lf_projection,
        "point_scanning_projection": ps_projection,
        "point_scanning_background": float(ps_floor),
        "light_field_background": float(lf_floor),
    }


@dataclass(frozen=True)
class PairedExperimentalPreprocessConfig:
    """Preprocessing and patch selection for registered LF/PS pairs."""

    lenslet_pitch: int = 11
    patch_lenslets: int = 21
    output_depth: int = 91
    axial_start: int = 20
    lf_background_quantile: float = 0.25
    lf_scale_quantile: float = 0.999
    lf_target_quantile_value: float = 0.10
    lf_calibration_frames: int = 16
    lf_stored_repeats: int = 5
    ps_background_quantile: float = 0.10
    ps_scale_quantile: float = 0.999
    ps_target_quantile_value: float = 2.4276368618011475
    ps_clip_multiple: float = 4.0
    patch_grid_starts_lenslets: tuple[int, ...] = (0, 13, 26)
    patches_per_acquisition: int = 3
    compression: str | None = "gzip"
    compression_level: int = 1

    def __post_init__(self) -> None:
        if self.lenslet_pitch <= 0 or self.patch_lenslets <= 0:
            raise ValueError("lenslet_pitch and patch_lenslets must be positive")
        if self.output_depth <= 0 or self.axial_start < 0:
            raise ValueError("output_depth must be positive and axial_start nonnegative")
        for name in (
            "lf_background_quantile",
            "lf_scale_quantile",
            "ps_background_quantile",
            "ps_scale_quantile",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.lf_scale_quantile <= self.lf_background_quantile:
            raise ValueError("LF scale quantile must exceed its background quantile")
        if self.ps_scale_quantile <= self.ps_background_quantile:
            raise ValueError("PS scale quantile must exceed its background quantile")
        if self.lf_target_quantile_value <= 0 or self.ps_target_quantile_value <= 0:
            raise ValueError("Target quantile values must be positive")
        if self.lf_calibration_frames <= 0 or self.lf_stored_repeats <= 0:
            raise ValueError("LF frame counts must be positive")
        if self.patches_per_acquisition <= 0:
            raise ValueError("patches_per_acquisition must be positive")
        if not self.patch_grid_starts_lenslets:
            raise ValueError("At least one patch-grid start is required")

    @property
    def patch_pixels(self) -> int:
        return self.lenslet_pitch * self.patch_lenslets


def extract_specimen_id(acquisition_name: str) -> str:
    """Extract specimen IDs such as ``F01``, ``JY05``, or ``S03``."""

    match = re.search(r"_(F\d{2}|JY\d{2}|S\d{2})_40X", acquisition_name)
    if match is None:
        raise ValueError(f"Cannot extract specimen ID from {acquisition_name!r}")
    return match.group(1)


def shift_volume_xy(
    volume: np.ndarray,
    shift_y: int,
    shift_x: int,
) -> np.ndarray:
    """Translate the last two dimensions with zero fill and no wraparound."""

    source = np.asarray(volume)
    if source.ndim < 2:
        raise ValueError("volume must have at least two dimensions")
    height, width = source.shape[-2:]
    if abs(shift_y) >= height or abs(shift_x) >= width:
        return np.zeros_like(source)
    destination = np.zeros_like(source)
    destination_y = slice(max(0, shift_y), min(height, height + shift_y))
    destination_x = slice(max(0, shift_x), min(width, width + shift_x))
    source_y = slice(max(0, -shift_y), min(height, height - shift_y))
    source_x = slice(max(0, -shift_x), min(width, width - shift_x))
    destination[..., destination_y, destination_x] = source[..., source_y, source_x]
    return destination


def _evenly_spaced_indices(length: int, count: int) -> list[int]:
    if length <= 0 or count <= 0:
        raise ValueError("length and count must be positive")
    if count >= length:
        return list(range(length))
    if count == 1:
        return [length // 2]
    return sorted({round(index * (length - 1) / (count - 1)) for index in range(count)})


def calibrate_paired_lf_frames(
    frames: Iterable[np.ndarray],
    *,
    background_quantile: float = 0.25,
    scale_quantile: float = 0.999,
    target_quantile_value: float = 0.10,
) -> tuple[list[np.ndarray], dict[str, float]]:
    """Apply one robust background and scale to all repeats in an acquisition."""

    arrays = [np.asarray(frame, dtype=np.float32) for frame in frames]
    if not arrays:
        raise ValueError("At least one LF frame is required")
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("All LF frames must have the same shape")
    values = np.concatenate([array.reshape(-1) for array in arrays])
    background = float(np.quantile(values, background_quantile))
    foreground_values = np.maximum(values - background, 0.0)
    scale = float(np.quantile(foreground_values, scale_quantile))
    if scale <= np.finfo(np.float32).eps:
        raise ValueError("LF calibration scale is zero")
    processed = [
        np.clip(
            np.maximum(array - background, 0.0)
            / scale
            * target_quantile_value,
            0.0,
            1.0,
        ).astype(np.float32)
        for array in arrays
    ]
    return processed, {
        "background": background,
        "scale": scale,
        "target_quantile_value": float(target_quantile_value),
    }


def preprocess_point_scanning_volume(
    volume: np.ndarray,
    *,
    background_quantile: float = 0.10,
    scale_quantile: float = 0.999,
    target_quantile_value: float = 2.4276368618011475,
    clip_multiple: float = 4.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Background-subtract and robustly scale a PS volume for supervision."""

    value = np.asarray(volume, dtype=np.float32)
    if value.ndim != 3:
        raise ValueError(f"Expected [Z,Y,X] point-scanning volume, got {value.shape}")
    background = float(np.quantile(value, background_quantile))
    foreground = np.maximum(value - background, 0.0)
    scale = float(np.quantile(foreground, scale_quantile))
    if scale <= np.finfo(np.float32).eps:
        raise ValueError("Point-scanning calibration scale is zero")
    processed = foreground / scale * target_quantile_value
    if clip_multiple > 0:
        processed = np.minimum(processed, target_quantile_value * clip_multiple)
    return processed.astype(np.float32), {
        "background": background,
        "scale": scale,
        "target_quantile_value": float(target_quantile_value),
        "clip_multiple": float(clip_multiple),
    }


def common_band_relative_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    kernel_size: tuple[int, int, int] = (3, 5, 5),
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Scale-invariant L2 after retaining only common coarse 3D structure.

    Both arguments use ``[B,Z,Y,X]``. Applying the same low-pass operation to
    both sides avoids treating a point-scanning target as raw-emitter truth;
    this is a conservative adaptation loss, not a point-scanning forward model.
    """

    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError(
            "prediction and target must share [B,Z,Y,X] shape, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if any(size <= 0 or size % 2 == 0 for size in kernel_size):
        raise ValueError("kernel_size must contain positive odd values")
    padding = tuple(size // 2 for size in kernel_size)
    filtered_prediction = F.avg_pool3d(
        prediction.unsqueeze(1), kernel_size, stride=1, padding=padding
    )[:, 0]
    filtered_target = F.avg_pool3d(
        target.unsqueeze(1), kernel_size, stride=1, padding=padding
    )[:, 0]
    dimensions = tuple(range(1, filtered_prediction.ndim))
    # Floor the energy before sqrt: sqrt(0) has an infinite derivative, which
    # can produce NaN gradients even if a later clamp masks its output.
    prediction_rms = filtered_prediction.square().mean(dim=dimensions, keepdim=True).clamp_min(epsilon ** 2).sqrt()
    target_rms = filtered_target.square().mean(dim=dimensions, keepdim=True).clamp_min(epsilon ** 2).sqrt()
    normalized_prediction = filtered_prediction / prediction_rms.clamp_min(epsilon)
    normalized_target = filtered_target / target_rms.clamp_min(epsilon)
    return (normalized_prediction - normalized_target).square().mean(dim=dimensions).mean()


class PairedExperimentalH5Dataset(Dataset[dict[str, Any]]):
    """Read cached experimental LF repeats and registered PS target patches."""

    def __init__(
        self,
        root: str | Path,
        *,
        split: Literal["training", "validation", "testing"],
        augment: bool = False,
        random_repeat: bool = True,
        include_metadata: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema") != PAIRED_EXPERIMENTAL_SCHEMA:
            raise ValueError(f"Unsupported paired-data schema in {manifest_path}")
        self.directory = self.root / split
        self.files = sorted(self.directory.glob("*.h5"), key=_natural_key)
        if not self.files:
            raise ValueError(f"No paired HDF5 samples found in {self.directory}")
        self.augment = bool(augment)
        self.random_repeat = bool(random_repeat)
        self.include_metadata = bool(include_metadata)
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.files[index]
        with h5py.File(path, "r") as handle:
            repeat_count = int(handle["lf"].shape[0])
            if self.random_repeat:
                repeat = int(torch.randint(repeat_count, size=()).item())
            else:
                repeat = repeat_count // 2
            lf = torch.as_tensor(np.asarray(handle["lf"][repeat]), dtype=self.dtype)
            target = torch.as_tensor(np.asarray(handle["gt"]), dtype=self.dtype)
            metadata = json.loads(handle.attrs["metadata_json"])
        if self.augment:
            transform = int(torch.randint(8, size=()).item())
            rotations = transform % 4
            if rotations:
                lf = torch.rot90(lf, rotations, dims=(-2, -1))
                target = torch.rot90(target, rotations, dims=(-2, -1))
            if transform >= 4:
                lf = torch.flip(lf, dims=(-1,))
                target = torch.flip(target, dims=(-1,))
        result: dict[str, Any] = {
            "lf": lf.contiguous(),
            "gt": target.contiguous(),
            "name": metadata["sample_name"],
            "acquisition": metadata["acquisition"],
            "specimen": metadata["specimen"],
        }
        if self.include_metadata:
            result["metadata"] = metadata
        return result


def build_paired_experimental_patch_cache(
    dataset_root: str | Path,
    registration_csv: str | Path,
    output_root: str | Path,
    *,
    validation_specimens: tuple[str, ...],
    testing_specimens: tuple[str, ...],
    config: PairedExperimentalPreprocessConfig | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    """Create compact lenslet-aligned HDF5 patches from accepted paired data.

    Existing cache directories are never overwritten. Choose a new output name
    when preprocessing or registration settings change.
    """

    config = config or PairedExperimentalPreprocessConfig()
    dataset_root = Path(dataset_root)
    registration_csv = Path(registration_csv)
    output_root = Path(output_root)
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != PAIRED_EXPERIMENTAL_SCHEMA:
            raise ValueError(f"Existing incompatible cache: {manifest_path}")
        return manifest
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite incomplete/nonempty cache {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    for split in ("training", "validation", "testing"):
        (output_root / split).mkdir(exist_ok=True)

    audits = {item.name: item for item in audit_paired_dataset(dataset_root)}
    with registration_csv.open(newline="", encoding="utf-8-sig") as handle:
        registration_rows = list(csv.DictReader(handle))
    accepted_rows = [
        row for row in registration_rows
        if row.get("accepted", "").strip().lower() in {"1", "true", "yes"}
    ]
    if not accepted_rows:
        raise ValueError(f"No accepted registrations in {registration_csv}")

    manifest_samples: list[dict[str, Any]] = []
    patch_pixels = config.patch_pixels
    for row in accepted_rows:
        name = row["name"]
        audit = audits.get(name)
        if audit is None or audit.status != "candidate":
            raise ValueError(f"Accepted registration is not a file-level candidate: {name}")
        specimen = extract_specimen_id(name)
        split = (
            "testing" if specimen in testing_specimens
            else "validation" if specimen in validation_specimens
            else "training"
        )
        ps_volume = load_point_scanning_stack(audit.point_scanning_directory)
        ps_processed, ps_calibration = preprocess_point_scanning_volume(
            ps_volume,
            background_quantile=config.ps_background_quantile,
            scale_quantile=config.ps_scale_quantile,
            target_quantile_value=config.ps_target_quantile_value,
            clip_multiple=config.ps_clip_multiple,
        )
        shift_y_lenslets = int(row["shift_y_lenslets"])
        shift_x_lenslets = int(row["shift_x_lenslets"])
        # Calibrate before inserting zero-filled registration borders so those
        # borders cannot bias the raw-count background quantile.
        ps_processed = shift_volume_xy(
            ps_processed,
            shift_y_lenslets * config.lenslet_pitch,
            shift_x_lenslets * config.lenslet_pitch,
        )
        z_end = config.axial_start + config.output_depth
        if z_end > ps_processed.shape[0]:
            raise ValueError(f"Axial crop exceeds PS volume for {name}")
        ps_processed = ps_processed[config.axial_start:z_end]

        lf_paths = _files(Path(audit.light_field_directory), "*.mat")
        calibration_indices = _evenly_spaced_indices(
            len(lf_paths), min(len(lf_paths), config.lf_calibration_frames)
        )
        stored_indices = _evenly_spaced_indices(
            len(lf_paths), min(len(lf_paths), config.lf_stored_repeats)
        )
        all_indices = sorted(set(calibration_indices) | set(stored_indices))
        loaded = {index: load_light_field_frame(lf_paths[index]) for index in all_indices}
        calibrated_frames, lf_calibration = calibrate_paired_lf_frames(
            [loaded[index] for index in calibration_indices],
            background_quantile=config.lf_background_quantile,
            scale_quantile=config.lf_scale_quantile,
            target_quantile_value=config.lf_target_quantile_value,
        )
        # Reuse the fitted acquisition-level background and scale on stored repeats.
        background = lf_calibration["background"]
        scale = lf_calibration["scale"]
        stored_frames = [
            np.clip(
                np.maximum(loaded[index] - background, 0.0)
                / scale
                * config.lf_target_quantile_value,
                0.0,
                1.0,
            ).astype(np.float32)
            for index in stored_indices
        ]
        del calibrated_frames

        candidate_starts = [
            (y * config.lenslet_pitch, x * config.lenslet_pitch)
            for y in config.patch_grid_starts_lenslets
            for x in config.patch_grid_starts_lenslets
        ]
        candidate_energies = []
        for y_start, x_start in candidate_starts:
            y_end, x_end = y_start + patch_pixels, x_start + patch_pixels
            if y_end > ps_processed.shape[-2] or x_end > ps_processed.shape[-1]:
                continue
            energy = float(ps_processed[:, y_start:y_end, x_start:x_end].sum())
            candidate_energies.append((energy, y_start, x_start))
        selected_starts = sorted(candidate_energies, reverse=True)[
            : config.patches_per_acquisition
        ]
        if not selected_starts:
            raise ValueError(f"No valid patches for {name}")

        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        for patch_index, (energy, y_start, x_start) in enumerate(selected_starts):
            y_end, x_end = y_start + patch_pixels, x_start + patch_pixels
            lf_patch = np.stack(
                [frame[y_start:y_end, x_start:x_end][None] for frame in stored_frames]
            )
            ps_patch = ps_processed[:, y_start:y_end, x_start:x_end]
            sample_name = f"{safe_name}__p{patch_index}_y{y_start}_x{x_start}"
            metadata = {
                "sample_name": sample_name,
                "acquisition": name,
                "specimen": specimen,
                "split": split,
                "crop_start_yx": [y_start, x_start],
                "crop_shape_yx": [patch_pixels, patch_pixels],
                "axial_start": config.axial_start,
                "depth": config.output_depth,
                "registration_correlation": float(row["correlation"]),
                "registration_shift_lenslets": [
                    shift_y_lenslets,
                    shift_x_lenslets,
                ],
                "lf_stored_indices": stored_indices,
                "lf_calibration": lf_calibration,
                "ps_calibration": ps_calibration,
                "target_energy": energy,
            }
            destination = output_root / split / f"{sample_name}.h5"
            compression_options = (
                config.compression_level if config.compression == "gzip" else None
            )
            with h5py.File(destination, "w") as handle:
                handle.attrs["schema"] = PAIRED_EXPERIMENTAL_SCHEMA
                handle.attrs["metadata_json"] = json.dumps(metadata)
                handle.create_dataset(
                    "lf",
                    data=lf_patch.astype(np.float16),
                    compression=config.compression,
                    compression_opts=compression_options,
                )
                handle.create_dataset(
                    "gt",
                    data=ps_patch.astype(np.float16),
                    compression=config.compression,
                    compression_opts=compression_options,
                )
            manifest_samples.append(metadata)
        if progress:
            print(
                f"cached {name}: split={split}, patches={len(selected_starts)}, "
                f"LF repeats={len(stored_frames)}",
                flush=True,
            )

    split_counts = {
        split: sum(sample["split"] == split for sample in manifest_samples)
        for split in ("training", "validation", "testing")
    }
    if any(count == 0 for count in split_counts.values()):
        raise ValueError(f"Every split must contain samples, got {split_counts}")
    manifest = {
        "schema": PAIRED_EXPERIMENTAL_SCHEMA,
        "source_root": str(dataset_root.resolve()),
        "registration_csv": str(registration_csv.resolve()),
        "config": asdict(config),
        "validation_specimens": list(validation_specimens),
        "testing_specimens": list(testing_specimens),
        "split_counts": split_counts,
        "samples": manifest_samples,
        "target_domain": "normalized_point_scanning_measurement",
        "input_domain": "acquisition_calibrated_light_field",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
