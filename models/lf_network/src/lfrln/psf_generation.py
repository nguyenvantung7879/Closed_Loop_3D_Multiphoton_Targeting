from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from math import factorial
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
import torch
from torch.nn import functional as F


GridMagnification = Literal["objective", "effective"]
PSFNormalization = Literal["none", "peak", "sum"]
ViewSymmetry = Literal["auto", "force", "none"]


def osa_ansi_nm(index: int) -> tuple[int, int]:
    """Convert a zero-based OSA/ANSI Zernike index to radial orders ``(n,m)``."""
    if index < 0:
        raise ValueError("OSA/ANSI index must be non-negative")
    n = int(np.ceil((-3 + np.sqrt(9 + 8 * index)) / 2))
    m = 2 * index - n * (n + 2)
    return n, m


def zernike_osa_unit_rms(
    index: int, rho: torch.Tensor, phi: torch.Tensor
) -> torch.Tensor:
    """Return a unit-RMS OSA/ANSI Zernike mode on the unit pupil.

    This uses the same normalization and angular convention as
    ``aberration_estimation.ipynb`` in the measured-PSF project. Values
    outside ``rho <= 1`` are zero.
    """
    if rho.shape != phi.shape:
        raise ValueError("rho and phi must have the same shape")
    n, m = osa_ansi_nm(index)
    absolute_m = abs(m)
    radial = torch.zeros_like(rho)
    for k in range((n - absolute_m) // 2 + 1):
        coefficient = (
            (-1) ** k
            * factorial(n - k)
            / (
                factorial(k)
                * factorial((n + absolute_m) // 2 - k)
                * factorial((n - absolute_m) // 2 - k)
            )
        )
        radial = radial + coefficient * rho ** (n - 2 * k)
    normalization = (
        np.sqrt(2 * (n + 1)) if m != 0 else np.sqrt(n + 1)
    )
    if m > 0:
        mode = normalization * radial * torch.cos(absolute_m * phi)
    elif m < 0:
        mode = normalization * radial * torch.sin(absolute_m * phi)
    else:
        mode = normalization * radial
    return torch.where(rho <= 1, mode, torch.zeros_like(mode))


@dataclass(frozen=True)
class ZernikeAberration:
    """Measured pupil wavefront represented by unit-RMS OSA modes.

    Coefficients are stored as physical wavefront RMS in nanometres. They are
    converted to phase at the LF emission wavelength during PSF generation,
    so a fit performed at 520 nm can safely be used for the current 515 nm
    emission model without reusing wavelength-specific radians.
    """

    osa_indices: tuple[int, ...]
    coefficients_nm_rms: tuple[float, ...]
    source: str = ""
    coefficient_column: str = ""
    phase_sign: float = 1.0

    def __post_init__(self) -> None:
        if not self.osa_indices:
            raise ValueError("At least one Zernike coefficient is required")
        if len(self.osa_indices) != len(self.coefficients_nm_rms):
            raise ValueError("Zernike indices and coefficients must have equal length")
        if len(set(self.osa_indices)) != len(self.osa_indices):
            raise ValueError("Zernike indices must be unique")
        if any(index < 0 for index in self.osa_indices):
            raise ValueError("Zernike indices must be non-negative")
        if not all(np.isfinite(value) for value in self.coefficients_nm_rms):
            raise ValueError("Zernike coefficients must be finite")
        if self.phase_sign not in (-1.0, 1.0):
            raise ValueError("phase_sign must be +1 or -1")

    @property
    def breaks_xy_reflection_symmetry(self) -> bool:
        """Whether quarter-view mirroring would discard measured asymmetry."""
        return any(
            coefficient != 0 and osa_ansi_nm(index)[1] != 0
            for index, coefficient in zip(
                self.osa_indices, self.coefficients_nm_rms, strict=True
            )
        )

    def phase_coefficients_rad(self, wavelength_m: float) -> tuple[float, ...]:
        if wavelength_m <= 0:
            raise ValueError("wavelength_m must be positive")
        scale = self.phase_sign * 2 * np.pi * 1e-9 / wavelength_m
        return tuple(scale * value for value in self.coefficients_nm_rms)


def load_zernike_aberration_csv(
    path: str | Path,
    *,
    coefficient_column: str = "recommended_nm_RMS",
    include_zero: bool = False,
    phase_sign: float = 1.0,
) -> ZernikeAberration:
    """Load measured OSA coefficients from the aberration-estimation CSV.

    The curated ``simulation_coefficients.csv`` should normally be loaded via
    its ``recommended_nm_RMS`` column. ``mean_nm_RMS`` is also supported for
    the unfiltered tier-A mean table when explicitly requested.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames or []
        for required in ("OSA_j", coefficient_column):
            if required not in fieldnames:
                raise KeyError(f"{path} has no column {required!r}")
        values: list[tuple[int, float]] = []
        for row in reader:
            index = int(row["OSA_j"])
            coefficient = float(row[coefficient_column])
            if include_zero or coefficient != 0:
                values.append((index, coefficient))
    if not values:
        raise ValueError(f"No non-zero aberration coefficients found in {path}")
    return ZernikeAberration(
        osa_indices=tuple(index for index, _ in values),
        coefficients_nm_rms=tuple(value for _, value in values),
        source=str(path.resolve()),
        coefficient_column=coefficient_column,
        phase_sign=phase_sign,
    )


@dataclass(frozen=True)
class CleanLFMPSFConfig:
    """Clean (non-scattering) counterpart of ``sLF_PSF_sym20260321.m``.

    ``grid_magnification='objective'`` intentionally reproduces MATLAB line
    ``dx_CCD = d_CCD / M_obj``.  Select ``'effective'`` only after deciding
    that the 80/75 relay magnification belongs in the object-space sampling.
    The distinction is kept explicit because it changes the lateral scale by
    6.67% for the current microscope.
    """

    wavelength_m: float = 515e-9
    objective_reference_tube_focal_m: float = 165e-3
    tube_lens_image_focal_m: float = 165e-3 * 80.0 / 75.0
    objective_magnification: float = 40.0
    numerical_aperture: float = 1.0
    sample_refractive_index: float = 1.33
    objective_refractive_index: float = 1.33
    mla_focal_m: float = 2.1e-3
    mla_pitch_m: float = 100e-6
    mla_grid_pixels: int = 21
    sensor_pixels_per_lenslet: int = 21
    source_positions_per_lenslet: int = 21
    z_min_m: float = -25e-6
    z_max_m: float = 25e-6
    z_step_m: float = 0.5e-6
    fft_padding_factor: int = 1
    psf_tail_level: float = 0.01
    psf_area_reference_size: int = 1001
    n_ccd_override: int | None = None
    grid_magnification: GridMagnification = "objective"
    field_normalization: PSFNormalization = "peak"
    h_normalization: PSFNormalization = "none"
    aberration: ZernikeAberration | None = None
    view_symmetry: ViewSymmetry = "auto"

    def __post_init__(self) -> None:
        positive = (
            self.wavelength_m,
            self.objective_reference_tube_focal_m,
            self.tube_lens_image_focal_m,
            self.objective_magnification,
            self.numerical_aperture,
            self.sample_refractive_index,
            self.objective_refractive_index,
            self.mla_focal_m,
            self.mla_pitch_m,
            self.z_step_m,
        )
        if min(positive) <= 0:
            raise ValueError("All optical lengths and indices must be positive")
        if self.numerical_aperture > min(
            self.sample_refractive_index, self.objective_refractive_index
        ):
            raise ValueError("numerical_aperture cannot exceed the refractive index")
        for name, value in (
            ("mla_grid_pixels", self.mla_grid_pixels),
            ("sensor_pixels_per_lenslet", self.sensor_pixels_per_lenslet),
            ("source_positions_per_lenslet", self.source_positions_per_lenslet),
            ("fft_padding_factor", self.fft_padding_factor),
            ("psf_area_reference_size", self.psf_area_reference_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.mla_grid_pixels % 2 == 0:
            raise ValueError("mla_grid_pixels must be odd for a centered lenslet")
        if self.source_positions_per_lenslet % 2 == 0:
            raise ValueError("source_positions_per_lenslet must be odd")
        if self.source_positions_per_lenslet > self.mla_grid_pixels:
            raise ValueError("source positions cannot exceed the MLA calculation grid")
        ratio_uv = (self.mla_grid_pixels - 1) / (
            self.source_positions_per_lenslet - 1
        ) if self.source_positions_per_lenslet > 1 else 1.0
        if abs(ratio_uv - round(ratio_uv)) > 1e-10:
            raise ValueError(
                "This parity port currently requires (PN_MLA-1)/(N_Pos-1) "
                "to be an integer; use N_Pos=PN_MLA for the MATLAB fallback"
            )
        if self.z_max_m < self.z_min_m:
            raise ValueError("z_max_m must be at least z_min_m")
        if not 0 < self.psf_tail_level < 1:
            raise ValueError("psf_tail_level must be between zero and one")
        if self.grid_magnification not in ("objective", "effective"):
            raise ValueError("Unsupported grid_magnification")
        if self.field_normalization not in ("none", "peak", "sum"):
            raise ValueError("Unsupported field_normalization")
        if self.h_normalization not in ("none", "peak", "sum"):
            raise ValueError("Unsupported h_normalization")
        if self.view_symmetry not in ("auto", "force", "none"):
            raise ValueError("view_symmetry must be 'auto', 'force', or 'none'")
        if self.n_ccd_override is not None:
            if self.n_ccd_override <= 0:
                raise ValueError("n_ccd_override must be positive")
            if self.n_ccd_override % self.mla_grid_pixels:
                raise ValueError("n_ccd_override must contain whole lenslets")
            if (self.n_ccd_override // self.mla_grid_pixels) % 2 == 0:
                raise ValueError("n_ccd_override must contain an odd lenslet count")
        binned = self.sensor_pixels_per_lenslet * (
            self.n_ccd_override or self.mla_grid_pixels
        ) / self.mla_grid_pixels
        if self.n_ccd_override is not None and abs(binned - round(binned)) > 1e-10:
            raise ValueError("The binned sensor size must be an integer")

    @property
    def objective_focal_m(self) -> float:
        return self.objective_reference_tube_focal_m / self.objective_magnification

    @property
    def effective_magnification(self) -> float:
        return self.tube_lens_image_focal_m / self.objective_focal_m

    @property
    def mla_pixel_m(self) -> float:
        return self.mla_pitch_m / self.mla_grid_pixels

    @property
    def sample_pixel_m(self) -> float:
        magnification = (
            self.objective_magnification
            if self.grid_magnification == "objective"
            else self.effective_magnification
        )
        return self.mla_pixel_m / magnification

    @property
    def volume_pixel_m(self) -> float:
        """Object-space pixel after PN_MLA -> Nnum lateral area resampling."""
        return (
            self.sample_pixel_m
            * self.mla_grid_pixels
            / self.sensor_pixels_per_lenslet
        )

    @property
    def z_positions_m(self) -> np.ndarray:
        count = int(round((self.z_max_m - self.z_min_m) / self.z_step_m)) + 1
        return self.z_min_m + np.arange(count, dtype=np.float64) * self.z_step_m

    @property
    def use_view_symmetry(self) -> bool:
        if self.view_symmetry == "force":
            return True
        if self.view_symmetry == "none":
            return False
        return self.aberration is None or not self.aberration.breaks_xy_reflection_symmetry


@dataclass
class CleanLFMPSFResult:
    """Generated arrays in the same logical order as the MATLAB variables."""

    h: torch.Tensor  # [sensor_y, sensor_x, u, v, z]
    objective_field: torch.Tensor  # [y, x, z, polarization]
    objective_intensity: torch.Tensor  # [calculation_y, calculation_x, z]
    objective_intensity_binned: torch.Tensor  # [volume_y, volume_x, z]
    pupil_wavefront_m: torch.Tensor  # [pupil_y, pupil_x]
    pupil_phase_rad: torch.Tensor  # [pupil_y, pupil_x]
    z_positions_m: torch.Tensor
    sample_pixel_m: float
    volume_pixel_m: float
    mla_pixel_m: float
    effective_magnification: float


@dataclass
class _PreparedOptics:
    config: CleanLFMPSFConfig
    n_ccd: int
    binned_sensor_size: int
    source_indices: tuple[int, ...]
    objective_field: torch.Tensor
    objective_intensity: torch.Tensor
    objective_intensity_binned: torch.Tensor
    pupil_wavefront_m: torch.Tensor
    pupil_phase_rad: torch.Tensor
    mla_phase: torch.Tensor
    propagation_phase: torch.Tensor
    output_slice: slice
    z_positions: torch.Tensor
    use_view_symmetry: bool


def zero_padded_shift2d(
    image: torch.Tensor, shift_y: int, shift_x: int
) -> torch.Tensor:
    """Integer, non-circular shift matching MATLAB ``im_shift2``."""
    if not isinstance(shift_y, int) or not isinstance(shift_x, int):
        raise TypeError("shift_y and shift_x must be integers")
    height, width = image.shape[:2]
    output = torch.zeros_like(image)
    if abs(shift_y) >= height or abs(shift_x) >= width:
        return output

    source_y = slice(max(-shift_y, 0), min(height - shift_y, height))
    source_x = slice(max(-shift_x, 0), min(width - shift_x, width))
    target_y = slice(max(shift_y, 0), min(height + shift_y, height))
    target_x = slice(max(shift_x, 0), min(width + shift_x, width))
    output[target_y, target_x, ...] = image[source_y, source_x, ...]
    return output


def _coordinate_step(coordinates: torch.Tensor) -> torch.Tensor:
    if coordinates.numel() < 2:
        return coordinates.new_tensor(1.0)
    if not coordinates.is_floating_point():
        raise TypeError("CZT coordinates must be floating point")

    # Derive the represented affine grid from its endpoints. Checking every
    # adjacent subtraction is unreliable for long float32 physical grids:
    # cancellation makes their differences vary by several ulps even when the
    # coordinates came directly from arange * constant_step.
    count = coordinates.numel()
    span = coordinates[-1] - coordinates[0]
    step = span / (count - 1)
    if not torch.isfinite(step) or step == 0:
        raise ValueError("CZT coordinates must have a finite nonzero step")
    normalized = (coordinates - coordinates[0]) / span
    expected = torch.linspace(
        0.0,
        1.0,
        count,
        device=coordinates.device,
        dtype=coordinates.dtype,
    )
    normalized_tolerance = max(
        1e-12, 8 * torch.finfo(coordinates.dtype).eps
    )
    if torch.max(torch.abs(normalized - expected)) > normalized_tolerance:
        raise ValueError("CZT coordinates must be uniformly spaced")
    return step


def _czt_dimension(
    signal: torch.Tensor,
    input_coordinates: torch.Tensor,
    output_coordinates: torch.Tensor,
    dimension: int,
) -> torch.Tensor:
    moved = torch.movedim(signal, dimension, 0)
    if moved.shape[0] != input_coordinates.numel():
        raise ValueError("Signal size does not match the CZT input grid")
    kernel = torch.exp(
        -2j
        * torch.pi
        * output_coordinates[:, None]
        * input_coordinates[None, :]
    )
    transformed = kernel @ moved.reshape(moved.shape[0], -1)
    scale = torch.sqrt(
        torch.abs(
            _coordinate_step(input_coordinates)
            * _coordinate_step(output_coordinates)
        )
    )
    transformed = transformed * scale
    transformed = transformed.reshape(output_coordinates.numel(), *moved.shape[1:])
    return torch.movedim(transformed, 0, dimension)


def czt2_physical(
    signal: torch.Tensor,
    fy_in: torch.Tensor,
    fx_in: torch.Tensor,
    y_out: torch.Tensor,
    x_out: torch.Tensor,
) -> torch.Tensor:
    """Two-dimensional matrix CZT with ``czt_ndim_hw`` unitary scaling."""
    output = _czt_dimension(signal, fy_in, y_out, 0)
    return _czt_dimension(output, fx_in, x_out, 1)


def vectorial_sp_czt(
    ex_pupil: torch.Tensor,
    ey_pupil: torch.Tensor,
    *,
    numerical_aperture: float,
    refractive_index: float,
    wavelength_m: float,
    fx_vec: torch.Tensor,
    fy_vec: torch.Tensor,
    x_roi: torch.Tensor,
    y_roi: torch.Tensor,
    z_roi: torch.Tensor,
) -> torch.Tensor:
    """PyTorch port of ``VectorialSP_CZT.m``.

    Returns complex electric field in ``[y, x, z, polarization]`` order.
    """
    if ex_pupil.shape != ey_pupil.shape or ex_pupil.ndim != 2:
        raise ValueError("Pupil fields must have the same two-dimensional shape")
    real_dtype = ex_pupil.real.dtype
    device = ex_pupil.device
    for coordinate in (fx_vec, fy_vec, x_roi, y_roi, z_roi):
        if coordinate.device != device:
            raise ValueError("All fields and coordinates must use the same device")

    ky, kx = torch.meshgrid(
        fy_vec * (2 * torch.pi), fx_vec * (2 * torch.pi), indexing="ij"
    )
    k_n = 2 * torch.pi / wavelength_m * refractive_index
    kr_square = kx.square() + ky.square()
    sin_theta = torch.sqrt(kr_square) / k_n
    sin_theta = sin_theta.clamp(max=1)
    theta = torch.asin(sin_theta)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    phi = torch.atan2(ky, kx)
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    difference = cos_theta - 1
    matlab_eps = torch.tensor(
        np.finfo(np.float64).eps, device=device, dtype=real_dtype
    )
    active_pupil = (ex_pupil.abs() + ey_pupil.abs()) > 0
    prefactor = torch.where(
        active_pupil,
        1 / (torch.sqrt(cos_theta) + matlab_eps),
        torch.zeros_like(cos_theta),
    )

    m11 = 0.5 * ((cos_theta + 1) + difference * torch.cos(2 * phi))
    m12 = 0.5 * difference * torch.sin(2 * phi)
    m13 = -sin_theta * cos_phi
    m22 = 0.5 * ((cos_theta + 1) - difference * torch.cos(2 * phi))
    m23 = -sin_theta * sin_phi

    ax0 = prefactor * (m11 * ex_pupil + m12 * ey_pupil)
    ay0 = prefactor * (m12 * ex_pupil + m22 * ey_pupil)
    az0 = prefactor * (-m13 * ex_pupil - m23 * ey_pupil)
    kz = k_n * cos_theta
    propagator = torch.exp(-1j * kz[..., None] * z_roi[None, None, :])

    fields = []
    for component in (ax0, ay0, az0):
        fields.append(
            czt2_physical(
                component[..., None] * propagator,
                fy_vec,
                fx_vec,
                y_roi,
                x_roi,
            )
        )
    field = torch.stack(fields, dim=-1)

    dfx = torch.abs(_coordinate_step(fx_vec))
    dfy = torch.abs(_coordinate_step(fy_vec))
    dx = torch.abs(_coordinate_step(x_roi))
    dy = torch.abs(_coordinate_step(y_roi))
    pixels_per_slice = ex_pupil.numel()
    scale = (
        np.sqrt(pixels_per_slice)
        * torch.sqrt(dfx / dx)
        * torch.sqrt(dfy / dy)
    )
    return field / scale


def calculate_psf_area(
    n_reference: int,
    *,
    wavelength_m: float,
    numerical_aperture: float,
    refractive_index: float,
    sample_pixel_m: float,
    maximum_abs_z_m: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Port of ``calcPSFArea.m`` used to select the lateral PSF support."""
    count = 2 * n_reference
    df = 1 / sample_pixel_m / count
    indices = torch.arange(-n_reference, n_reference, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(indices, indices, indexing="ij")
    spatial_frequency = torch.sqrt(xx.square() + yy.square()) * df
    pupil = (spatial_frequency <= numerical_aperture / wavelength_m).to(dtype)
    pupil = pupil / torch.sqrt(pupil.sum())
    k_square = (2 * torch.pi * spatial_frequency).square()
    kz = torch.sqrt(
        torch.clamp(
            (2 * torch.pi / wavelength_m * refractive_index) ** 2 - k_square,
            min=0,
        )
    )
    angular_spectrum = (
        torch.exp(-1j * kz * maximum_abs_z_m)
        * pupil
        / np.sqrt(pupil.numel())
    )
    field = torch.fft.fftshift(torch.fft.fft2(angular_spectrum))
    start = n_reference - n_reference // 2
    return field.abs().square()[start : start + n_reference, start : start + n_reference]


def _select_n_ccd(config: CleanLFMPSFConfig, area: torch.Tensor) -> int:
    if config.n_ccd_override is not None:
        return config.n_ccd_override
    reference = config.psf_area_reference_size
    center = reference // 2
    search = area[: center + 1, center]
    index_one_based = int(torch.argmin(torch.abs(search - config.psf_tail_level))) + 1
    raw_size = ((center + 1) - index_one_based) * 2 - 1
    lenslet = config.mla_grid_pixels
    return lenslet * (int(np.floor(raw_size / (2 * lenslet))) * 2 + 1)


def _normalize_complex_field(
    field: torch.Tensor, normalization: PSFNormalization
) -> torch.Tensor:
    if normalization == "none":
        return field
    if normalization == "sum":
        return field / torch.sqrt(field.abs().square().sum()).clamp_min(
            torch.finfo(field.real.dtype).eps
        )
    flat = field.reshape(-1)
    peak = flat[torch.argmax(flat.abs())]
    return field / peak


def _normalization_scale(
    values: torch.Tensor, normalization: PSFNormalization
) -> torch.Tensor:
    if normalization == "none":
        return values.new_tensor(1.0)
    if normalization == "peak":
        return values.amax().clamp_min(torch.finfo(values.dtype).eps)
    return values.sum().clamp_min(torch.finfo(values.dtype).eps)


def _area_resample_matrix(
    input_size: int,
    output_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Box/area resize weights followed by MATLAB-style energy correction."""
    if input_size == output_size:
        return torch.eye(input_size, device=device, dtype=dtype)
    scale = input_size / output_size
    left = torch.arange(output_size, device=device, dtype=dtype) * scale
    right = left + scale
    pixel_left = torch.arange(input_size, device=device, dtype=dtype)
    pixel_right = pixel_left + 1
    overlap = torch.clamp(
        torch.minimum(right[:, None], pixel_right[None, :])
        - torch.maximum(left[:, None], pixel_left[None, :]),
        min=0,
    )
    # MATLAB's later energy correction makes sum weights equal one per input
    # lenslet. Keeping the resize as an average matches imresize(..., 'box').
    return overlap / scale


def lenslet_area_resize(
    image: torch.Tensor, input_pixels: int, output_pixels: int
) -> torch.Tensor:
    """Blockwise ``LFreshape`` with per-lenslet energy conservation."""
    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError("Expected a square two-dimensional sensor image")
    if image.shape[0] % input_pixels:
        raise ValueError("Sensor size must contain complete lenslets")
    if input_pixels == output_pixels:
        return image
    lenslets = image.shape[0] // input_pixels
    blocks = image.reshape(lenslets, input_pixels, lenslets, input_pixels).permute(
        0, 2, 1, 3
    )
    weights = _area_resample_matrix(
        input_pixels,
        output_pixels,
        device=image.device,
        dtype=image.dtype,
    )
    resized = torch.einsum("oi,abij,pj->abop", weights, blocks, weights)
    # The area weights partition every input pixel. Their column sums are
    # output_pixels/input_pixels, so this fixed factor conserves energy exactly.
    # A data-dependent energy_in/energy_out ratio with an absolute epsilon
    # suppresses faint lenslets and breaks optical superposition.
    resized = resized * (input_pixels / output_pixels) ** 2
    return resized.permute(0, 2, 1, 3).reshape(
        lenslets * output_pixels, lenslets * output_pixels
    )


def area_resize_psf_stack(
    intensity: torch.Tensor, output_size: int
) -> torch.Tensor:
    """Area-resample ``[Y,X,Z]`` intensity with per-plane energy conservation."""
    if intensity.ndim != 3 or intensity.shape[0] != intensity.shape[1]:
        raise ValueError("Expected a square objective intensity [Y, X, Z]")
    if output_size <= 0:
        raise ValueError("output_size must be positive")
    if intensity.shape[0] == output_size:
        return intensity
    planes = intensity.permute(2, 0, 1).unsqueeze(1)
    resized = F.interpolate(planes, size=(output_size, output_size), mode="area")
    energy_in = planes.sum(dim=(-2, -1), keepdim=True)
    energy_out = resized.sum(dim=(-2, -1), keepdim=True)
    resized = resized * energy_in / energy_out.clamp_min(
        torch.finfo(intensity.dtype).eps
    )
    return resized[:, 0].permute(1, 2, 0)


def _prepare_optics(
    config: CleanLFMPSFConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> _PreparedOptics:
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("PSF construction requires float32 or float64")
    if config.n_ccd_override is None:
        area = calculate_psf_area(
            config.psf_area_reference_size,
            wavelength_m=config.wavelength_m,
            numerical_aperture=config.numerical_aperture,
            refractive_index=config.sample_refractive_index,
            sample_pixel_m=config.sample_pixel_m,
            maximum_abs_z_m=max(abs(config.z_min_m), abs(config.z_max_m)),
            device=device,
            dtype=dtype,
        )
    else:
        area = torch.empty(0, device=device, dtype=dtype)
    n_ccd = _select_n_ccd(config, area)
    padded = config.fft_padding_factor * n_ccd - (
        (config.fft_padding_factor - 1) % 2
    ) * (n_ccd % 2)
    indices = torch.arange(1, padded + 1, device=device, dtype=dtype) - (
        padded // 2 + 1
    )
    z_positions = torch.as_tensor(config.z_positions_m, device=device, dtype=dtype)
    coordinate_x = indices * config.sample_pixel_m
    df = 1 / config.sample_pixel_m / (2 * (padded // 2))
    bfp_step = df * config.wavelength_m * config.objective_focal_m
    coordinate_bfp = indices * bfp_step
    bfp_y, bfp_x = torch.meshgrid(coordinate_bfp, coordinate_bfp, indexing="ij")
    pupil = (
        torch.sqrt(bfp_x.square() + bfp_y.square())
        <= config.numerical_aperture * config.objective_focal_m
    ).to(dtype)
    pupil = pupil / torch.sqrt(pupil.sum())
    pupil_radius = config.numerical_aperture * config.objective_focal_m
    rho = torch.sqrt(bfp_x.square() + bfp_y.square()) / pupil_radius
    phi = torch.atan2(bfp_y, bfp_x)
    pupil_wavefront_m = torch.zeros_like(pupil)
    if config.aberration is not None:
        for index, coefficient_nm in zip(
            config.aberration.osa_indices,
            config.aberration.coefficients_nm_rms,
            strict=True,
        ):
            pupil_wavefront_m = pupil_wavefront_m + (
                coefficient_nm
                * 1e-9
                * zernike_osa_unit_rms(index, rho, phi)
            )
    phase_sign = (
        config.aberration.phase_sign if config.aberration is not None else 1.0
    )
    pupil_phase_rad = (
        phase_sign * 2 * torch.pi * pupil_wavefront_m / config.wavelength_m
    )
    pupil_phasor = torch.exp(1j * pupil_phase_rad)
    complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    ex_pupil = (pupil * pupil_phasor / np.sqrt(2) * 1j).to(complex_dtype)
    ey_pupil = (pupil * pupil_phasor / np.sqrt(2)).to(complex_dtype)
    frequency_coordinates = coordinate_bfp / (
        config.wavelength_m * config.objective_focal_m
    )
    objective_field = vectorial_sp_czt(
        ex_pupil,
        ey_pupil,
        numerical_aperture=config.numerical_aperture,
        refractive_index=config.objective_refractive_index,
        wavelength_m=config.wavelength_m,
        fx_vec=frequency_coordinates,
        fy_vec=frequency_coordinates,
        x_roi=coordinate_x,
        y_roi=coordinate_x,
        z_roi=z_positions,
    )
    objective_field = _normalize_complex_field(
        objective_field, config.field_normalization
    )
    objective_intensity = objective_field.abs().square().sum(dim=-1)

    crop_start = int(np.ceil((padded + 1) / 2 - np.floor(n_ccd / 2))) - 1
    lenslet_coordinates = (
        torch.arange(1, config.mla_grid_pixels + 1, device=device, dtype=dtype)
        - int(np.ceil(config.mla_grid_pixels / 2))
    ) * config.mla_pixel_m
    lenslet_y, lenslet_x = torch.meshgrid(
        lenslet_coordinates, lenslet_coordinates, indexing="ij"
    )
    single_phase = -2 * torch.pi * (
        lenslet_x.square() + lenslet_y.square()
    ) / (2 * config.wavelength_m * config.mla_focal_m)
    lenslet_count = int(
        2
        * round(
            ((n_ccd / 2) - (config.mla_grid_pixels / 2))
            / config.mla_grid_pixels
        )
        + 1
    )
    if lenslet_count * config.mla_grid_pixels != n_ccd:
        raise RuntimeError("The selected PSF support does not tile into lenslets")
    phase_crop = single_phase.repeat(lenslet_count, lenslet_count)
    mla_phase = torch.zeros((padded, padded), device=device, dtype=dtype)
    mla_phase[
        crop_start : crop_start + n_ccd,
        crop_start : crop_start + n_ccd,
    ] = phase_crop

    yy, xx = torch.meshgrid(indices, indices, indexing="ij")
    spatial_frequency = torch.sqrt(xx.square() + yy.square()) * df
    k_square = (2 * torch.pi * spatial_frequency / config.effective_magnification).square()
    kz_mla = torch.sqrt(
        torch.clamp((2 * torch.pi / config.wavelength_m) ** 2 - k_square, min=0)
    )
    propagation_phase = kz_mla * config.mla_focal_m
    binned_size = lenslet_count * config.sensor_pixels_per_lenslet
    objective_intensity_binned = area_resize_psf_stack(
        objective_intensity, binned_size
    )
    output_start = int(np.ceil((padded + 1) / 2 - np.floor(n_ccd / 2))) - 1
    ratio_uv = int(
        round(
            (config.mla_grid_pixels - 1)
            / (config.source_positions_per_lenslet - 1)
        )
    ) if config.source_positions_per_lenslet > 1 else 1
    source_indices = (
        (int(np.ceil((config.mla_grid_pixels + 1) / 2)),)
        if config.source_positions_per_lenslet == 1
        else tuple(range(1, config.mla_grid_pixels + 1, ratio_uv))
    )
    return _PreparedOptics(
        config=config,
        n_ccd=n_ccd,
        binned_sensor_size=binned_size,
        source_indices=source_indices,
        objective_field=objective_field,
        objective_intensity=objective_intensity,
        objective_intensity_binned=objective_intensity_binned,
        pupil_wavefront_m=pupil_wavefront_m,
        pupil_phase_rad=pupil_phase_rad,
        mla_phase=mla_phase,
        propagation_phase=propagation_phase,
        output_slice=slice(output_start, output_start + n_ccd),
        z_positions=z_positions,
        use_view_symmetry=config.use_view_symmetry,
    )


def _source_psfs(prepared: _PreparedOptics) -> Iterator[tuple[int, int, int, torch.Tensor]]:
    """Yield calculated PSFs, using quarter symmetry only when physically valid."""
    config = prepared.config
    count = config.source_positions_per_lenslet
    calculated_count = (count + 1) // 2 if prepared.use_view_symmetry else count
    center_mla = int(np.ceil((config.mla_grid_pixels + 1) / 2))
    mla_transmittance = torch.exp(1j * prepared.mla_phase)
    propagation = torch.exp(1j * prepared.propagation_phase)[..., None]
    for z_index in range(prepared.z_positions.numel()):
        field = prepared.objective_field[:, :, z_index, :]
        for u in range(calculated_count):
            shift_y = prepared.source_indices[u] - center_mla
            for v in range(calculated_count):
                shift_x = prepared.source_indices[v] - center_mla
                shifted = zero_padded_shift2d(field, shift_y, shift_x)
                spectrum = torch.fft.fftshift(
                    torch.fft.fft2(
                        shifted * mla_transmittance[..., None], dim=(0, 1)
                    ),
                    dim=(0, 1),
                )
                sensor_field = torch.fft.ifft2(
                    torch.fft.ifftshift(spectrum * propagation, dim=(0, 1)),
                    dim=(0, 1),
                )
                sensor_field = zero_padded_shift2d(
                    sensor_field, -shift_y, -shift_x
                )
                cropped = sensor_field[
                    prepared.output_slice, prepared.output_slice, :
                ]
                intensity = cropped.abs().square().sum(dim=-1)
                intensity = lenslet_area_resize(
                    intensity,
                    config.mla_grid_pixels,
                    config.sensor_pixels_per_lenslet,
                )
                yield z_index, u, v, intensity.real


def _symmetry_targets(
    u: int, v: int, count: int
) -> tuple[tuple[int, int, bool, bool], ...]:
    u2 = count - 1 - u
    v2 = count - 1 - v
    targets = {
        (u, v, False, False),
        (u2, v, True, False),
        (u, v2, False, True),
        (u2, v2, True, True),
    }
    return tuple(targets)


def _psf_targets(
    u: int, v: int, count: int, use_view_symmetry: bool
) -> tuple[tuple[int, int, bool, bool], ...]:
    if not use_view_symmetry:
        return ((u, v, False, False),)
    return _symmetry_targets(u, v, count)


def generate_clean_lfm_psf(
    config: CleanLFMPSFConfig | None = None,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    maximum_h_bytes: int = 2 * 1024**3,
) -> CleanLFMPSFResult:
    """Generate clean ``H`` in memory for small or cropped configurations.

    Use :func:`write_clean_lfm_psf_h5` for a full PSF whose allocation would
    exceed ``maximum_h_bytes``.
    """
    config = config or CleanLFMPSFConfig()
    device = torch.device(device)
    prepared = _prepare_optics(config, device=device, dtype=dtype)
    count = config.source_positions_per_lenslet
    shape = (
        prepared.binned_sensor_size,
        prepared.binned_sensor_size,
        count,
        count,
        prepared.z_positions.numel(),
    )
    bytes_required = int(np.prod(shape)) * torch.empty((), dtype=dtype).element_size()
    if bytes_required > maximum_h_bytes:
        raise MemoryError(
            f"H shape {shape} needs {bytes_required / 1024**3:.2f} GiB; "
            "use write_clean_lfm_psf_h5 for streamed output"
        )
    h = torch.zeros(shape, device=device, dtype=dtype)
    for z_index, u, v, intensity in _source_psfs(prepared):
        for target_u, target_v, flip_y, flip_x in _psf_targets(
            u, v, count, prepared.use_view_symmetry
        ):
            value = intensity
            if flip_y:
                value = torch.flip(value, dims=(0,))
            if flip_x:
                value = torch.flip(value, dims=(1,))
            h[:, :, target_u, target_v, z_index] = value
    h = h / _normalization_scale(h, config.h_normalization)
    return CleanLFMPSFResult(
        h=h,
        objective_field=prepared.objective_field,
        objective_intensity=prepared.objective_intensity,
        objective_intensity_binned=prepared.objective_intensity_binned,
        pupil_wavefront_m=prepared.pupil_wavefront_m,
        pupil_phase_rad=prepared.pupil_phase_rad,
        z_positions_m=prepared.z_positions,
        sample_pixel_m=config.sample_pixel_m,
        volume_pixel_m=config.volume_pixel_m,
        mla_pixel_m=config.mla_pixel_m,
        effective_magnification=config.effective_magnification,
    )


def write_clean_lfm_psf_h5(
    output_path: str | Path,
    config: CleanLFMPSFConfig | None = None,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    compression: Literal["lzf", "gzip"] | None = "lzf",
    overwrite: bool = False,
    progress: bool = False,
) -> Path:
    """Stream a full non-scattering detection PSF without keeping ``H`` in RAM.

    The dataset ``H`` uses MATLAB logical order ``[Y, X, U, V, Z]``.  The
    file is regular HDF5 rather than a MATLAB ``-v7.3`` MAT container; MATLAB
    can read it with ``h5read`` and Python can use it without axis guessing.
    """
    import h5py

    config = config or CleanLFMPSFConfig()
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    prepared = _prepare_optics(config, device=device, dtype=dtype)
    count = config.source_positions_per_lenslet
    shape = (
        prepared.binned_sensor_size,
        prepared.binned_sensor_size,
        count,
        count,
        prepared.z_positions.numel(),
    )
    if config.h_normalization != "none":
        raise ValueError(
            "Streamed generation supports h_normalization='none'; normalize "
            "globally after generation so no slice gets a different scale"
        )
    with h5py.File(output_path, "w") as handle:
        h_dataset = handle.create_dataset(
            "H",
            shape=shape,
            dtype=np.float32 if dtype == torch.float32 else np.float64,
            chunks=(shape[0], shape[1], 1, 1, 1),
            compression=compression,
        )
        objective = prepared.objective_intensity.detach().cpu().numpy()
        handle.create_dataset("I_PSF_Vec", data=objective, compression=compression)
        handle.create_dataset(
            "I_PSF_Vec_binned",
            data=prepared.objective_intensity_binned.detach().cpu().numpy(),
            compression=compression,
        )
        handle.create_dataset(
            "pupil_wavefront_m",
            data=prepared.pupil_wavefront_m.detach().cpu().numpy(),
            compression=compression,
        )
        handle.create_dataset(
            "pupil_phase_rad",
            data=prepared.pupil_phase_rad.detach().cpu().numpy(),
            compression=compression,
        )
        handle.create_dataset(
            "Z", data=prepared.z_positions.detach().cpu().numpy()
        )
        handle.attrs["input_domain"] = "raw_emitter_intensity"
        handle.attrs["objective_operator"] = (
            "H and I_PSF_Vec include the configured detection-pupil aberration"
        )
        handle.attrs["matlab_reference"] = (
            "sLF_PSF_sym20260321.m clean branch with optional pupil aberration"
        )
        config_attributes = asdict(config)
        aberration_attributes = config_attributes.pop("aberration")
        for key, value in config_attributes.items():
            if value is not None:
                handle.attrs[key] = value
        handle.attrs["view_symmetry_used"] = prepared.use_view_symmetry
        if aberration_attributes is None:
            handle.attrs["aberration_model"] = "none"
        else:
            handle.attrs["aberration_model"] = "OSA_ANSI_unit_RMS_Zernike"
            handle.attrs["aberration_osa_indices"] = np.asarray(
                aberration_attributes["osa_indices"], dtype=np.int32
            )
            handle.attrs["aberration_coefficients_nm_rms"] = np.asarray(
                aberration_attributes["coefficients_nm_rms"], dtype=np.float64
            )
            handle.attrs["aberration_source"] = aberration_attributes["source"]
            handle.attrs["aberration_coefficient_column"] = aberration_attributes[
                "coefficient_column"
            ]
            handle.attrs["aberration_phase_sign"] = aberration_attributes[
                "phase_sign"
            ]
        handle.attrs["n_ccd"] = prepared.n_ccd
        handle.attrs["sample_pixel_m"] = config.sample_pixel_m
        handle.attrs["volume_pixel_m"] = config.volume_pixel_m
        handle.attrs["effective_magnification"] = config.effective_magnification

        source_psfs = _source_psfs(prepared)
        if progress:
            from tqdm.auto import tqdm

            calculated_count = (
                (count + 1) // 2 if prepared.use_view_symmetry else count
            )
            source_psfs = tqdm(
                source_psfs,
                total=(
                    prepared.z_positions.numel()
                    * calculated_count
                    * calculated_count
                ),
                desc="generate non-scattering detection LF PSF",
            )
        for z_index, u, v, intensity in source_psfs:
            base = intensity.detach().cpu().numpy()
            for target_u, target_v, flip_y, flip_x in _psf_targets(
                u, v, count, prepared.use_view_symmetry
            ):
                value = base
                if flip_y:
                    value = np.flip(value, axis=0)
                if flip_x:
                    value = np.flip(value, axis=1)
                h_dataset[:, :, target_u, target_v, z_index] = value
        handle.flush()
    return output_path


def psf_parity_metrics(
    candidate: torch.Tensor, reference: torch.Tensor
) -> dict[str, float]:
    """Scale-aligned metrics for comparing Python and MATLAB PSF arrays."""
    if candidate.shape != reference.shape:
        raise ValueError(
            f"PSF shapes differ: candidate {candidate.shape}, reference {reference.shape}"
        )
    candidate = candidate.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    reference = reference.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    gain = torch.dot(candidate, reference) / torch.dot(candidate, candidate).clamp_min(
        torch.finfo(torch.float64).eps
    )
    aligned = candidate * gain
    relative_l2 = torch.linalg.vector_norm(aligned - reference) / torch.linalg.vector_norm(
        reference
    ).clamp_min(torch.finfo(torch.float64).eps)
    centered_candidate = aligned - aligned.mean()
    centered_reference = reference - reference.mean()
    correlation = torch.dot(centered_candidate, centered_reference) / (
        torch.linalg.vector_norm(centered_candidate)
        * torch.linalg.vector_norm(centered_reference)
    ).clamp_min(torch.finfo(torch.float64).eps)
    return {
        "gain": float(gain),
        "relative_l2": float(relative_l2),
        "correlation": float(correlation),
    }
