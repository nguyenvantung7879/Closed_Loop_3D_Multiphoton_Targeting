from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from .data import load_volume_tensor
from .emccd import EMCCDConfig, get_emccd_model


PhantomKind = Literal[
    "1um_spheres",
    "2um_spheres",
    "10um_shells",
    "200nm_lines",
    "multiscale_lines",
    "random_spheres",
    "random_shells",
]
OverlapMode = Literal["sum", "max"]
PSFNormalization = Literal["sum", "peak", "none"]
CameraMode = Literal["none", "poisson_gaussian", "legacy_em_power", "emccd"]
ForwardDomain = Literal["raw", "cutoff"]


PHANTOM_KINDS: tuple[PhantomKind, ...] = (
    "1um_spheres",
    "2um_spheres",
    "10um_shells",
    "200nm_lines",
    "random_spheres",
    "random_shells",
)

# Keep PHANTOM_KINDS unchanged so old six-family datasets and notebooks remain
# reproducible. New generators may explicitly opt into multiscale_lines.
AVAILABLE_PHANTOM_KINDS: tuple[PhantomKind, ...] = (
    *PHANTOM_KINDS,
    "multiscale_lines",
)


@dataclass(frozen=True)
class PhantomConfig:
    """Physical grid and intensity settings for synthetic emitter volumes."""

    depth: int = 91
    height: int = 517 #231
    width: int = 517 #231
    axial_step_um: float = 0.5
    lateral_pixel_um: float = 100.0 / 11.0 / (40.0 * 80.0 / 75.0)
    intensity_range: tuple[float, float] = (0.4, 1.0)
    overlap_mode: OverlapMode = "sum"
    sphere_lateral_supersampling: int = 4
    sphere_axial_supersampling: int = 4
    line_samples_per_lateral_voxel: int = 4
    line_cross_section_samples: int = 16
    multiscale_line_diameters_um: tuple[float, ...] = (0.4, 0.5, 0.75, 1.0)

    def __post_init__(self) -> None:
        if min(self.depth, self.height, self.width) <= 0:
            raise ValueError("Phantom dimensions must be positive")
        if self.axial_step_um <= 0 or self.lateral_pixel_um <= 0:
            raise ValueError("Voxel spacing must be positive")
        low, high = self.intensity_range
        if not 0 <= low < high:
            raise ValueError("intensity_range must satisfy 0 <= low < high")
        if self.overlap_mode not in ("sum", "max"):
            raise ValueError("overlap_mode must be 'sum' or 'max'")
        sampling_fields = (
            "sphere_lateral_supersampling",
            "sphere_axial_supersampling",
            "line_samples_per_lateral_voxel",
            "line_cross_section_samples",
        )
        for field in sampling_fields:
            if not isinstance(getattr(self, field), int) or getattr(self, field) <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if not self.multiscale_line_diameters_um or any(
            diameter <= 0 for diameter in self.multiscale_line_diameters_um
        ):
            raise ValueError(
                "multiscale_line_diameters_um must contain positive diameters"
            )

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.depth, self.height, self.width

    @property
    def spacing_um(self) -> tuple[float, float, float]:
        return self.axial_step_um, self.lateral_pixel_um, self.lateral_pixel_um


@dataclass(frozen=True)
class ObjectivePSFConfig:
    """Settings for the incoherent objective-PSF convolution."""

    normalization: PSFNormalization = "sum"
    padding_factor: int = 2
    clamp_nonnegative: bool = True
    output_minmax: bool = False

    def __post_init__(self) -> None:
        if self.normalization not in ("sum", "peak", "none"):
            raise ValueError("normalization must be 'sum', 'peak', or 'none'")
        if self.padding_factor < 2:
            raise ValueError("padding_factor must be at least 2")


@dataclass(frozen=True)
class CameraConfig:
    """Count-domain camera simulation without per-sample normalization.

    For ``emccd``, photons_per_lf_unit denotes detected input electrons per
    LF unit. The measured blank maps replace background_counts/read_noise_std;
    only EMCCDConfig.extra_background_electrons adds new background charge.
    gain_variation changes signal exposure, not the EM multiplication gain.
    If signal_adu_per_lf_unit is supplied, it replaces photons_per_lf_unit:
    lambda = signal_adu_per_lf_unit * clean_lf / G. Thus G can change the
    signal-noise strength at fixed mean ADU, as in LF_synthMeasured.m.
    """

    mode: CameraMode = "poisson_gaussian"
    photons_per_lf_unit: float = 3000.0
    background_counts: float = 60.0
    read_noise_std: float = 1.5
    gain_variation: tuple[float, float] = (1.04, 1.20)
    background_variation: tuple[float, float] = (1.04, 1.20)
    binning_factor: float = (21.0 / 11.0) ** 2
    legacy_power: float = 6.0 / 5.0
    emccd: EMCCDConfig | None = None
    signal_adu_per_lf_unit: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("none", "poisson_gaussian", "legacy_em_power", "emccd"):
            raise ValueError(f"Unsupported camera mode {self.mode!r}")
        if self.mode == "emccd" and self.emccd is None:
            raise ValueError("emccd mode requires EMCCDConfig with a measured calibration and explicit ADU/electron gain")
        if self.mode != "emccd" and self.emccd is not None:
            raise ValueError("EMCCDConfig requires camera mode='emccd'")
        if self.signal_adu_per_lf_unit is not None:
            if (self.mode != "emccd" or not math.isfinite(self.signal_adu_per_lf_unit)
                    or self.signal_adu_per_lf_unit <= 0):
                raise ValueError("signal_adu_per_lf_unit requires emccd mode and a finite positive value")
        if self.photons_per_lf_unit < 0 or self.background_counts < 0:
            raise ValueError("Camera count scales must be nonnegative")
        if self.read_noise_std < 0 or self.binning_factor <= 0:
            raise ValueError("Invalid read noise or binning factor")
        for interval in (self.gain_variation, self.background_variation):
            if interval[0] <= 0 or interval[1] < interval[0]:
                raise ValueError("Variation intervals must be positive and ordered")


@dataclass(frozen=True)
class LFPreprocessConfig:
    """Fixed conversion from camera counts to the H1 input scale."""

    subtract_background: float = 60.0
    count_scale: float = 3600.0
    clamp_nonnegative: bool = True

    def __post_init__(self) -> None:
        if self.count_scale <= 0:
            raise ValueError("count_scale must be positive")


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    matrix, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(matrix) < 0:
        matrix[:, 0] *= -1
    return matrix.astype(np.float32)


def _random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    vector = rng.normal(size=3)
    norm = np.linalg.norm(vector)
    while norm == 0:
        vector = rng.normal(size=3)
        norm = np.linalg.norm(vector)
    return (vector / norm).astype(np.float32)


def _sample_intensity(config: PhantomConfig, rng: np.random.Generator) -> float:
    return float(rng.uniform(*config.intensity_range))


def _physical_center(
    config: PhantomConfig,
    rng: np.random.Generator,
    margin_um: tuple[float, float, float],
) -> np.ndarray:
    sizes = np.asarray(config.shape, dtype=np.float32)
    spacing = np.asarray(config.spacing_um, dtype=np.float32)
    low = np.asarray(margin_um, dtype=np.float32) / spacing
    high = sizes - 1 - low
    low = np.minimum(low, (sizes - 1) / 2)
    high = np.maximum(high, low)
    return rng.uniform(low, high + 1).astype(np.float32) * spacing


def _bounding_slices(
    config: PhantomConfig,
    center_um: np.ndarray,
    extent_um: np.ndarray,
) -> tuple[slice, slice, slice]:
    spacing = np.asarray(config.spacing_um, dtype=np.float32)
    center_index = center_um / spacing
    extent_index = extent_um / spacing
    starts = np.maximum(0, np.floor(center_index - extent_index - 1).astype(int))
    stops = np.minimum(
        np.asarray(config.shape),
        np.ceil(center_index + extent_index + 1).astype(int) + 1,
    )
    return tuple(slice(int(start), int(stop)) for start, stop in zip(starts, stops))  # type: ignore[return-value]


def _coordinates(
    slices: tuple[slice, slice, slice],
    config: PhantomConfig,
    center_um: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    axes = []
    for axis_slice, spacing, center in zip(
        slices, config.spacing_um, center_um.tolist()
    ):
        axis = (
            torch.arange(
                axis_slice.start,
                axis_slice.stop,
                device=device,
                dtype=dtype,
            )
            * spacing
            - center
        )
        axes.append(axis)
    return torch.meshgrid(*axes, indexing="ij")


def _merge(
    target: torch.Tensor,
    slices: tuple[slice, slice, slice],
    object_volume: torch.Tensor,
    mode: OverlapMode,
) -> None:
    if mode == "sum":
        target[slices].add_(object_volume)
    else:
        target[slices].copy_(torch.maximum(target[slices], object_volume))


def _centered_subvoxel_offsets(
    samples: int,
    spacing: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return midpoint-rule offsets spanning one voxel."""
    return (
        (torch.arange(samples, device=device, dtype=dtype) + 0.5) / samples
        - 0.5
    ) * spacing


def _supersampled_sphere_occupancy(
    coordinates: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    config: PhantomConfig,
    radius_um: float,
) -> torch.Tensor:
    """Estimate partial-volume sphere occupancy on the anisotropic grid."""
    dz, dy, dx = coordinates
    occupancy = torch.zeros_like(dz)
    z_offsets = _centered_subvoxel_offsets(
        config.sphere_axial_supersampling,
        config.axial_step_um,
        device=dz.device,
        dtype=dz.dtype,
    )
    lateral_offsets = _centered_subvoxel_offsets(
        config.sphere_lateral_supersampling,
        config.lateral_pixel_um,
        device=dz.device,
        dtype=dz.dtype,
    )
    for z_offset in z_offsets:
        for y_offset in lateral_offsets:
            for x_offset in lateral_offsets:
                occupancy.add_(
                    (
                        (dz + z_offset).square()
                        + (dy + y_offset).square()
                        + (dx + x_offset).square()
                        <= radius_um**2
                    ).to(occupancy.dtype)
                )
    sample_count = (
        config.sphere_axial_supersampling
        * config.sphere_lateral_supersampling**2
    )
    return occupancy / sample_count


def _line_cross_section_offsets(
    direction: np.ndarray,
    radius_um: float,
    samples: int,
) -> np.ndarray:
    """Deterministically sample equal-area points on a line cross-section."""
    helper = np.zeros(3, dtype=np.float32)
    helper[int(np.argmin(np.abs(direction)))] = 1.0
    first = np.cross(direction, helper)
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    indices = np.arange(samples, dtype=np.float32)
    radii = radius_um * np.sqrt((indices + 0.5) / samples)
    angles = indices * np.float32(np.pi * (3.0 - np.sqrt(5.0)))
    return (
        radii[:, None] * np.cos(angles)[:, None] * first[None]
        + radii[:, None] * np.sin(angles)[:, None] * second[None]
    ).astype(np.float32)


def _trilinear_splat_occupancy(
    points_um: torch.Tensor,
    slices: tuple[slice, slice, slice],
    config: PhantomConfig,
    sample_volume_um3: float,
) -> torch.Tensor:
    """Deposit physical volume samples as fractional voxel occupancy."""
    device, dtype = points_um.device, points_um.dtype
    spacing = torch.as_tensor(config.spacing_um, device=device, dtype=dtype)
    starts = torch.as_tensor(
        [axis_slice.start for axis_slice in slices], device=device, dtype=torch.long
    )
    local_shape = tuple(axis_slice.stop - axis_slice.start for axis_slice in slices)
    local_shape_tensor = torch.as_tensor(local_shape, device=device, dtype=torch.long)
    continuous_index = points_um / spacing
    lower = torch.floor(continuous_index).to(torch.long)
    fraction = continuous_index - lower.to(dtype)
    voxel_volume_um3 = float(np.prod(config.spacing_um))
    sample_occupancy = sample_volume_um3 / voxel_volume_um3
    flat = torch.zeros(int(np.prod(local_shape)), device=device, dtype=dtype)

    for z_upper in (0, 1):
        for y_upper in (0, 1):
            for x_upper in (0, 1):
                corner = torch.tensor(
                    (z_upper, y_upper, x_upper), device=device, dtype=torch.long
                )
                local_index = lower + corner - starts
                valid = ((local_index >= 0) & (local_index < local_shape_tensor)).all(
                    dim=1
                )
                if not bool(valid.any()):
                    continue
                corner_weight = torch.where(
                    corner.bool(), fraction, 1.0 - fraction
                ).prod(dim=1)
                valid_index = local_index[valid]
                linear_index = (
                    (valid_index[:, 0] * local_shape[1] + valid_index[:, 1])
                    * local_shape[2]
                    + valid_index[:, 2]
                )
                flat.scatter_add_(
                    0,
                    linear_index,
                    corner_weight[valid] * sample_occupancy,
                )
    return flat.reshape(local_shape).clamp_(0.0, 1.0)


def _add_sphere(
    volume: torch.Tensor,
    config: PhantomConfig,
    rng: np.random.Generator,
    radius_um: float,
    intensity: float,
    *,
    antialias: bool = False,
) -> dict[str, Any]:
    center = _physical_center(config, rng, (radius_um,) * 3)
    slices = _bounding_slices(config, center, np.full(3, radius_um, np.float32))
    coordinates = _coordinates(
        slices, config, center, device=volume.device, dtype=volume.dtype
    )
    if antialias:
        occupancy = _supersampled_sphere_occupancy(
            coordinates, config, radius_um
        )
        voxelization = "partial_volume_midpoint"
    else:
        dz, dy, dx = coordinates
        occupancy = (
            dz.square() + dy.square() + dx.square() <= radius_um**2
        ).to(volume.dtype)
        voxelization = "binary_voxel_center"
    _merge(volume, slices, occupancy * intensity, config.overlap_mode)
    return {
        "center_um": center.tolist(),
        "radius_um": radius_um,
        "intensity": intensity,
        "voxelization": voxelization,
    }


def _add_shell(
    volume: torch.Tensor,
    config: PhantomConfig,
    rng: np.random.Generator,
    radii_um: np.ndarray,
    intensity: float,
    *,
    rotate: bool,
) -> dict[str, Any]:
    rotation = _random_rotation(rng) if rotate else np.eye(3, dtype=np.float32)
    # The MATLAB-parity shell occupies 0.5 <= r_normalized^2 <= 1.5, so its
    # outer semi-axes are sqrt(1.5) times the nominal radii.  Bounding by only
    # radii_um clips the shell on every axis and gives its projection a square
    # outline.
    outer_radii_um = np.sqrt(np.float32(1.5)) * radii_um
    extent = np.abs(rotation) @ outer_radii_um
    center = _physical_center(config, rng, tuple(extent.tolist()))
    slices = _bounding_slices(config, center, extent)
    dz, dy, dx = _coordinates(
        slices, config, center, device=volume.device, dtype=volume.dtype
    )
    points = torch.stack(torch.broadcast_tensors(dz, dy, dx), dim=-1)
    rotation_tensor = torch.as_tensor(rotation, device=volume.device, dtype=volume.dtype)
    principal = points @ rotation_tensor
    radii = torch.as_tensor(radii_um, device=volume.device, dtype=volume.dtype)
    normalized_radius = (principal / radii).square().sum(dim=-1)
    # Preserve the MATLAB shell thickness (0.5 <= r^2 <= 1.5).
    mask = (normalized_radius >= 0.5) & (normalized_radius <= 1.5)
    _merge(volume, slices, mask.to(volume.dtype) * intensity, config.overlap_mode)
    return {
        "center_um": center.tolist(),
        "radii_um": radii_um.tolist(),
        "intensity": intensity,
        "rotation": rotation.tolist(),
    }


def _line_occupancy_from_geometry(
    config: PhantomConfig,
    *,
    center: np.ndarray,
    direction: np.ndarray,
    radius_um: float,
    length_um: float,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[tuple[slice, slice, slice], torch.Tensor, int]:
    """Voxelize one known physical line without drawing new randomness."""

    extent = np.abs(direction) * (length_um / 2) + radius_um
    slices = _bounding_slices(config, center, extent)
    along_step_um = config.lateral_pixel_um / config.line_samples_per_lateral_voxel
    along_samples = max(2, int(np.ceil(length_um / along_step_um)))
    along = (
        (torch.arange(along_samples, device=device, dtype=dtype) + 0.5)
        / along_samples
        - 0.5
    ) * length_um
    direction_tensor = torch.as_tensor(
        direction, device=device, dtype=dtype
    )
    cross_section = torch.as_tensor(
        _line_cross_section_offsets(
            direction,
            radius_um,
            config.line_cross_section_samples,
        ),
        device=device,
        dtype=dtype,
    )
    center_tensor = torch.as_tensor(
        center, device=device, dtype=dtype
    )
    points_um = (
        center_tensor[None, None, :]
        + along[:, None, None] * direction_tensor[None, None, :]
        + cross_section[None, :, :]
    ).reshape(-1, 3)
    physical_volume_um3 = np.pi * radius_um**2 * length_um
    occupancy = _trilinear_splat_occupancy(
        points_um,
        slices,
        config,
        sample_volume_um3=physical_volume_um3 / points_um.shape[0],
    )
    return slices, occupancy, along_samples


def _add_line(
    volume: torch.Tensor,
    config: PhantomConfig,
    rng: np.random.Generator,
    radius_um: float,
    length_um: float,
    intensity: float,
) -> dict[str, Any]:
    direction = _random_unit_vector(rng)
    extent = np.abs(direction) * (length_um / 2) + radius_um
    center = _physical_center(config, rng, tuple(extent.tolist()))
    slices, occupancy, along_samples = _line_occupancy_from_geometry(
        config,
        center=center,
        direction=direction,
        radius_um=radius_um,
        length_um=length_um,
        device=volume.device,
        dtype=volume.dtype,
    )
    _merge(volume, slices, occupancy * intensity, config.overlap_mode)
    return {
        "center_um": center.tolist(),
        "direction": direction.tolist(),
        "radius_um": radius_um,
        "length_um": length_um,
        "intensity": intensity,
        "voxelization": "physical_samples_trilinear",
        "along_samples": along_samples,
        "cross_section_samples": config.line_cross_section_samples,
    }


def line_diameter_support_masks(
    metadata: Mapping[str, Any],
    *,
    threshold_relative: float = 1e-3,
    device: torch.device | str = "cpu",
) -> dict[float, torch.Tensor]:
    """Reconstruct per-diameter voxel support from saved line metadata.

    Dataset samples store the complete phantom mapping under ``metadata['phantom']``;
    the direct phantom mapping is accepted as well. Each returned boolean mask
    has shape ``[D, H, W]`` and unions all objects with the same physical
    diameter. This is a geometry diagnostic only: it does not regenerate an LF
    image and does not alter the saved target.
    """

    if not 0 <= threshold_relative < 1:
        raise ValueError("threshold_relative must be in [0, 1)")
    phantom = metadata.get("phantom", metadata)
    if not isinstance(phantom, Mapping):
        raise TypeError("Line metadata must contain a phantom mapping")
    config_payload = phantom.get("config")
    objects = phantom.get("objects")
    if not isinstance(config_payload, Mapping) or not isinstance(objects, list):
        raise ValueError("Line metadata is missing phantom config or objects")
    config = PhantomConfig(**dict(config_payload))
    masks: dict[float, torch.Tensor] = {}
    for item in objects:
        if not isinstance(item, Mapping):
            raise TypeError("Every phantom object must be a mapping")
        required = {"center_um", "direction", "radius_um", "length_um"}
        missing = required - set(item)
        if missing:
            raise ValueError(f"Line object is missing {sorted(missing)}")
        radius_um = float(item["radius_um"])
        diameter_um = round(2.0 * radius_um, 6)
        slices, occupancy, _ = _line_occupancy_from_geometry(
            config,
            center=np.asarray(item["center_um"], dtype=np.float32),
            direction=np.asarray(item["direction"], dtype=np.float32),
            radius_um=radius_um,
            length_um=float(item["length_um"]),
            device=device,
            dtype=torch.float32,
        )
        support = occupancy > threshold_relative * occupancy.amax()
        mask = masks.setdefault(
            diameter_um,
            torch.zeros(config.shape, dtype=torch.bool, device=device),
        )
        mask[slices] |= support
    return dict(sorted(masks.items()))


def generate_phantom(
    config: PhantomConfig | None = None,
    *,
    seed: int,
    kind: PhantomKind | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Generate one deterministic volume in ``[D, H, W]`` order.

    Shapes are drawn analytically in physical coordinates, so rotations do not
    need MATLAB's expensive ``imrotate3`` and axial anisotropy is explicit.
    """
    config = config or PhantomConfig()
    rng = np.random.default_rng(seed)
    selected_kind = kind or PHANTOM_KINDS[int(rng.integers(0, len(PHANTOM_KINDS)))]
    volume = torch.zeros(config.shape, device=device, dtype=dtype)
    objects: list[dict[str, Any]] = []

    if selected_kind == "1um_spheres":
        count = int(rng.integers(5, 45))
        radius_um = 0.5
        for _ in range(count):
            objects.append(
                _add_sphere(
                    volume,
                    config,
                    rng,
                    radius_um,
                    _sample_intensity(config, rng),
                    antialias=True,
                )
            )
    elif selected_kind == "2um_spheres":
        count = int(rng.integers(5, 45))
        radius_um = 1.0
        for _ in range(count):
            objects.append(
                _add_sphere(
                    volume,
                    config,
                    rng,
                    radius_um,
                    _sample_intensity(config, rng),
                    antialias=True,
                )
            )
    elif selected_kind == "10um_shells":
        count = int(rng.integers(1, 11))
        radius_um = float(np.floor(5.0 / config.lateral_pixel_um)) * config.lateral_pixel_um
        for _ in range(count):
            objects.append(
                _add_shell(
                    volume,
                    config,
                    rng,
                    np.full(3, radius_um, dtype=np.float32),
                    _sample_intensity(config, rng),
                    rotate=False,
                )
            )
    elif selected_kind == "200nm_lines":
        count = int(rng.integers(10, 70))
        field_width = config.width * config.lateral_pixel_um
        radius_um = 0.1
        for _ in range(count):
            objects.append(
                _add_line(
                    volume,
                    config,
                    rng,
                    radius_um=radius_um,
                    length_um=float(rng.uniform(0.25, 1.0) * field_width),
                    intensity=_sample_intensity(config, rng),
                )
            )
    elif selected_kind == "multiscale_lines":
        # These diameters span approximately 1.9--4.7 lateral voxels for the
        # default grid. They remain a fine-structure task without making one
        # sub-voxel, below-cutoff width the sole supervised answer.
        count = int(rng.integers(5, 26))
        field_width = config.width * config.lateral_pixel_um
        diameters_um = np.asarray(
            config.multiscale_line_diameters_um, dtype=np.float32
        )
        for _ in range(count):
            diameter_um = float(rng.choice(diameters_um))
            objects.append(
                _add_line(
                    volume,
                    config,
                    rng,
                    radius_um=diameter_um / 2.0,
                    length_um=float(rng.uniform(0.25, 1.0) * field_width),
                    intensity=_sample_intensity(config, rng),
                )
            )
    elif selected_kind == "random_spheres":
        count = int(rng.integers(1, 21))
        for _ in range(count):
            radius_pixels = float(rng.integers(5, 20))
            objects.append(
                _add_sphere(
                    volume,
                    config,
                    rng,
                    radius_pixels * config.lateral_pixel_um,
                    _sample_intensity(config, rng),
                )
            )
    elif selected_kind == "random_shells":
        count = int(rng.integers(1, 21))
        for _ in range(count):
            radii_pixels = rng.integers(5, 40, size=3).astype(np.float32)
            objects.append(
                _add_shell(
                    volume,
                    config,
                    rng,
                    radii_pixels * config.lateral_pixel_um,
                    _sample_intensity(config, rng),
                    rotate=True,
                )
            )
    else:  # pragma: no cover - guarded by the Literal/API
        raise ValueError(f"Unsupported phantom kind {selected_kind!r}")

    metadata = {
        "seed": seed,
        "kind": selected_kind,
        "count": count,
        "config": asdict(config),
        "objects": objects,
    }
    return volume, metadata


def _centered_bounds(source: int, padded: int) -> tuple[int, int]:
    # Matches paddingVolume.m for its factor-two, odd-to-even case.
    start = (padded - source + 1) // 2
    return start, start + source


def _minmax_per_sample(volume: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    minimum = volume.amin(dim=(-3, -2, -1), keepdim=True)
    maximum = volume.amax(dim=(-3, -2, -1), keepdim=True)
    return (volume - minimum) / (maximum - minimum).clamp_min(epsilon)


class IncoherentObjectiveFilter(nn.Module):
    """FFT convolution with the measured incoherent objective PSF.

    The centering and crop reproduce ``paddingVolume`` + ``ifftshift`` from
    ``adjustDataGT2GTcutoff_20260217.m``. The default PSF sum normalization is
    energy-preserving; use ``normalization='peak', output_minmax=True`` only for
    parity with the old MATLAB target files.
    """

    def __init__(
        self,
        psf: torch.Tensor,
        config: ObjectivePSFConfig | None = None,
    ) -> None:
        super().__init__()
        config = config or ObjectivePSFConfig()
        if psf.ndim != 3 or not psf.is_floating_point():
            raise ValueError("Expected a floating PSF in [D, H, W] order")
        psf = psf.detach().to(dtype=torch.float32)
        if config.normalization == "sum":
            psf = psf / psf.sum().clamp_min(torch.finfo(psf.dtype).eps)
        elif config.normalization == "peak":
            psf = psf / psf.amax().clamp_min(torch.finfo(psf.dtype).eps)
        self.config = config
        self.register_buffer("psf", psf, persistent=True)
        self._otf_cache: dict[
            tuple[torch.device, torch.dtype, tuple[int, int, int]], torch.Tensor
        ] = {}

    def _apply(self, function):  # type: ignore[no-untyped-def]
        self._otf_cache.clear()
        return super()._apply(function)

    @classmethod
    def from_mat(
        cls,
        path: str | Path,
        *,
        variable_name: str = "objPSF",
        config: ObjectivePSFConfig | None = None,
    ) -> "IncoherentObjectiveFilter":
        return cls(load_volume_tensor(path, variable_name), config)

    @classmethod
    def from_h5(
        cls,
        path: str | Path,
        *,
        dataset_name: str = "I_PSF_Vec_binned",
        axis_order: Literal["yxz", "zyx"] = "yxz",
        depth: int | None = None,
        config: ObjectivePSFConfig | None = None,
    ) -> "IncoherentObjectiveFilter":
        """Load the objective PSF saved with a Python-generated LF PSF.

        ``I_PSF_Vec_binned`` is written in logical ``[Y,X,Z]`` order and is
        converted here to the filter's ``[Z,Y,X]`` convention. A centered
        depth crop keeps it aligned with the centered LF-forward export.
        """
        import h5py

        path = Path(path)
        with h5py.File(path, "r") as handle:
            if dataset_name not in handle:
                raise KeyError(f"{path} has no dataset {dataset_name!r}")
            values = np.asarray(handle[dataset_name], dtype=np.float32)
        if values.ndim != 3:
            raise ValueError(
                f"Expected a three-dimensional objective PSF, got {values.shape}"
            )
        if axis_order == "yxz":
            values = np.transpose(values, (2, 0, 1))
        elif axis_order != "zyx":
            raise ValueError("axis_order must be 'yxz' or 'zyx'")
        if depth is not None:
            if depth <= 0 or depth > values.shape[0]:
                raise ValueError(
                    f"Requested depth={depth}, but PSF contains {values.shape[0]} planes"
                )
            start = (values.shape[0] - depth) // 2
            values = values[start : start + depth]
        return cls(torch.from_numpy(values.copy()), config)

    def _padded_shape(
        self, volume_shape: tuple[int, int, int]
    ) -> tuple[int, int, int]:
        """Return an alias-free FFT shape for a volume and the PSF kernel."""

        return tuple(
            max(
                self.config.padding_factor * kernel_size,
                volume_size + kernel_size - 1,
            )
            for volume_size, kernel_size in zip(
                volume_shape, self.psf.shape, strict=True
            )
        )  # type: ignore[return-value]

    @staticmethod
    def _pad_center(
        tensor: torch.Tensor, padded_shape: tuple[int, int, int]
    ) -> torch.Tensor:
        output = tensor.new_zeros((*tensor.shape[:-3], *padded_shape))
        bounds = [
            _centered_bounds(source, padded)
            for source, padded in zip(tensor.shape[-3:], padded_shape, strict=True)
        ]
        output[
            ...,
            bounds[0][0] : bounds[0][1],
            bounds[1][0] : bounds[1][1],
            bounds[2][0] : bounds[2][1],
        ] = tensor
        return output

    def _otf(
        self,
        device: torch.device,
        dtype: torch.dtype,
        padded_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        key = (device, dtype, padded_shape)
        cached = self._otf_cache.get(key)
        if cached is None or cached.is_inference():
            # The filter is often first called while preparing a notebook
            # preview under inference_mode().  Never retain an inference-only
            # OTF: autograd cannot save it later when B(raw_pred) is trained.
            with torch.inference_mode(False), torch.no_grad():
                psf = self._pad_center(
                    self.psf.to(device=device, dtype=dtype), padded_shape
                )
                cached = torch.fft.rfftn(psf, dim=(-3, -2, -1))
            self._otf_cache[key] = cached
        return cached

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        squeeze = volume.ndim == 3
        if squeeze:
            volume = volume.unsqueeze(0)
        if volume.ndim != 4:
            raise ValueError(
                "Expected a volume in [B,D,H,W] or [D,H,W] order, got "
                f"{tuple(volume.shape)}"
            )
        output_dtype = volume.dtype
        compute_dtype = torch.float32 if volume.dtype in (torch.float16, torch.bfloat16) else volume.dtype
        volume_shape = tuple(volume.shape[-3:])
        padded_shape = self._padded_shape(volume_shape)
        padded = self._pad_center(volume.to(dtype=compute_dtype), padded_shape)
        spectrum = torch.fft.rfftn(padded, dim=(-3, -2, -1))
        filtered = torch.fft.irfftn(
            spectrum * self._otf(volume.device, compute_dtype, padded_shape),
            s=padded.shape[-3:],
            dim=(-3, -2, -1),
        )
        filtered = torch.fft.ifftshift(filtered, dim=(-3, -2, -1))
        bounds = [
            _centered_bounds(source, padded_size)
            for source, padded_size in zip(
                volume_shape, padded.shape[-3:], strict=True
            )
        ]
        filtered = filtered[
            ...,
            bounds[0][0] : bounds[0][1],
            bounds[1][0] : bounds[1][1],
            bounds[2][0] : bounds[2][1],
        ]
        if self.config.clamp_nonnegative:
            filtered = filtered.clamp_min(0)
        if self.config.output_minmax:
            filtered = _minmax_per_sample(filtered)
        filtered = filtered.to(dtype=output_dtype)
        return filtered.squeeze(0) if squeeze else filtered


def _uniform_scalar(
    interval: tuple[float, float],
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> torch.Tensor:
    low, high = interval
    if high == low:
        return torch.tensor(low, device=device, dtype=dtype)
    return torch.empty((), device=device, dtype=dtype).uniform_(
        low, high, generator=generator
    )


def simulate_camera(
    clean_lf: torch.Tensor,
    config: CameraConfig | None = None,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply camera statistics while retaining an explicit count scale."""
    config = config or CameraConfig()
    if not clean_lf.is_floating_point():
        raise TypeError("clean_lf must be floating point")
    if config.mode == "emccd":
        model = get_emccd_model(config.emccd, str(clean_lf.device))
        electrons_per_unit = (config.photons_per_lf_unit if config.signal_adu_per_lf_unit is None
                              else config.signal_adu_per_lf_unit / config.emccd.gain_adu_per_electron)
        counts, metadata = model(
            clean_lf, electrons_per_lf_unit=electrons_per_unit,
            signal_variation=config.gain_variation,
            background_variation=config.background_variation, generator=generator,
        )
        metadata["signal_adu_per_lf_unit"] = electrons_per_unit * config.emccd.gain_adu_per_electron
        metadata["signal_scale_mode"] = "electrons" if config.signal_adu_per_lf_unit is None else "adu"
        return counts, metadata
    clean_lf = clean_lf.clamp_min(0)
    gain_factor = _uniform_scalar(
        config.gain_variation,
        device=clean_lf.device,
        dtype=clean_lf.dtype,
        generator=generator,
    )
    background_factor = _uniform_scalar(
        config.background_variation,
        device=clean_lf.device,
        dtype=clean_lf.dtype,
        generator=generator,
    )
    signal_mean = clean_lf * config.photons_per_lf_unit * gain_factor
    background_mean = torch.round(
        torch.as_tensor(
            config.background_counts,
            device=clean_lf.device,
            dtype=clean_lf.dtype,
        )
        * background_factor
    )

    if config.mode == "none":
        observed = signal_mean + background_mean
    elif config.mode == "poisson_gaussian":
        observed = torch.poisson(signal_mean + background_mean, generator=generator)
        if config.read_noise_std:
            read_noise = torch.randn(
                clean_lf.shape,
                device=clean_lf.device,
                dtype=clean_lf.dtype,
                generator=generator,
            ) * config.read_noise_std
            observed = observed + read_noise
    else:
        # Exact algebra of the current MATLAB empirical 6/5-power model.
        variance = (
            (signal_mean / config.binning_factor).clamp_min(0)
            ** config.legacy_power
        ) * config.binning_factor
        signal_noise = torch.poisson(variance, generator=generator) - variance
        background = torch.poisson(
            torch.ones_like(clean_lf) * background_mean,
            generator=generator,
        )
        observed = signal_mean + signal_noise + background

    metadata = {
        "gain_factor": float(gain_factor),
        "background_factor": float(background_factor),
        "photons_per_lf_unit": config.photons_per_lf_unit,
        "background_mean": float(background_mean),
    }
    return observed, metadata


def preprocess_lf_counts(
    counts: torch.Tensor,
    config: LFPreprocessConfig | None = None,
    *,
    camera_config: CameraConfig | None = None,
) -> torch.Tensor:
    """Create the H1 input using one dataset-wide affine conversion."""
    config = config or LFPreprocessConfig()
    if camera_config is not None and camera_config.mode == "emccd":
        emccd = camera_config.emccd
        if emccd.subtract_calibrated_offset:
            if config.subtract_background != 0:
                raise ValueError("Use subtract_background=0 with EMCCD calibrated-offset subtraction; avoid subtracting the baseline twice")
            counts = counts - get_emccd_model(emccd, str(counts.device)).baseline
    output = (counts - config.subtract_background) / config.count_scale
    return output.clamp_min(0) if config.clamp_nonnegative else output


def camera_input_signal_gain(
    camera: CameraConfig | Mapping[str, Any],
    preprocess: LFPreprocessConfig | Mapping[str, Any],
) -> float:
    """Expected LF-unit -> network-unit slope, excluding exposure jitter.

    This is a slope, not a fit to noisy/clipped pixels. Constant background
    offsets and clipping must be treated separately from multiplicative gain.
    Accepts manifest mappings so training does not have to load calibration maps.
    """
    c = asdict(camera) if isinstance(camera, CameraConfig) else camera
    p = asdict(preprocess) if isinstance(preprocess, LFPreprocessConfig) else preprocess
    if c.get("mode") == "emccd" and c.get("signal_adu_per_lf_unit") is not None:
        return float(c["signal_adu_per_lf_unit"]) / float(p["count_scale"])
    slope = float(c["photons_per_lf_unit"]) / float(p["count_scale"])
    if c.get("mode") == "emccd":
        slope *= float(c["emccd"]["gain_adu_per_electron"])
    return slope


class SyntheticLFPipeline(nn.Module):
    """Compose cutoff, LF projection, camera noise, and H1 preprocessing."""

    def __init__(
        self,
        objective_filter: IncoherentObjectiveFilter,
        forward_model: nn.Module,
        *,
        forward_domain: ForwardDomain = "raw",
        target_domain: ForwardDomain = "cutoff",
        camera_config: CameraConfig | None = None,
        preprocess_config: LFPreprocessConfig | None = None,
        optical_gain: float = 1.0,
    ) -> None:
        super().__init__()
        if forward_domain not in ("raw", "cutoff") or target_domain not in (
            "raw",
            "cutoff",
        ):
            raise ValueError(
                "forward_domain and target_domain must be 'raw' or 'cutoff'"
            )
        declared_domain = getattr(forward_model, "input_domain", None)
        if declared_domain is not None and declared_domain != forward_domain:
            raise ValueError(
                f"Projector declares {declared_domain!r}, pipeline requests "
                f"{forward_domain!r}"
            )
        if optical_gain <= 0:
            raise ValueError("optical_gain must be positive")
        self.objective_filter = objective_filter
        self.forward_model = forward_model
        self.forward_domain = forward_domain
        self.target_domain = target_domain
        self.camera_config = camera_config or CameraConfig()
        self.preprocess_config = preprocess_config or LFPreprocessConfig()
        self.optical_gain = optical_gain

    def forward(
        self,
        raw: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, Any]:
        if raw.ndim == 3:
            raw = raw.unsqueeze(0)
        cutoff = self.objective_filter(raw)
        projector_input = raw if self.forward_domain == "raw" else cutoff
        target = raw if self.target_domain == "raw" else cutoff
        clean_lf = self.forward_model(projector_input) * self.optical_gain
        counts, camera_metadata = simulate_camera(
            clean_lf, self.camera_config, generator=generator
        )
        result = {
            "lf": preprocess_lf_counts(counts, self.preprocess_config, camera_config=self.camera_config),
            "gt": target,
            "lf_feedback": clean_lf,
            "raw": raw,
            "cutoff": cutoff,
            "lf_counts": counts,
            "camera": camera_metadata,
            "feedback_compatible": self.forward_domain == self.target_domain,
            "raw_feedback_compatible": self.forward_domain == "raw",
        }
        if self.camera_config.mode == "emccd":
            model = get_emccd_model(self.camera_config.emccd, str(clean_lf.device))
            result["lf_expected_counts"] = model.expected_counts(clean_lf, camera_metadata)
        return result
