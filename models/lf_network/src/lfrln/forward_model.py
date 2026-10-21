from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .physics_rearrange import space_to_view_depth, view_depth_to_space


ForwardInputDomain = Literal["raw", "cutoff"]
PSFH5AxisOrder = Literal["auto", "yxuvz", "zvuxy"]


@dataclass(frozen=True)
class LFMForwardConfig:
    """Shape and acquisition settings for the fixed LF projector.

    The PSF weights describe the detection path. ``apply_axial_profile`` is a
    legacy convenience for applying the MATLAB ``LF_z_disV`` excitation-side
    envelope to the emitter before detection; it is not part of the LF PSF.
    """

    depth: int = 91
    nnum: int = 11
    kernel_size: int = 21
    depth_step_um: float = 0.5
    axial_fwhm_um: float = 40.62
    apply_axial_profile: bool = False
    # The exported H contains objective + MLA propagation and therefore maps
    # raw emitter intensity to the sensor. Applying it to GT_emPSF would apply
    # the objective response twice.
    input_domain: ForwardInputDomain = "raw"

    def __post_init__(self) -> None:
        if self.depth <= 0 or self.nnum <= 0 or self.kernel_size <= 0:
            raise ValueError("depth, nnum, and kernel_size must be positive")
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for MATLAB 'same' padding")
        if self.depth_step_um <= 0 or self.axial_fwhm_um <= 0:
            raise ValueError("Axial sampling values must be positive")
        if self.input_domain not in ("raw", "cutoff"):
            raise ValueError("input_domain must be 'raw' or 'cutoff'")

    @property
    def num_views(self) -> int:
        return self.nnum**2

    @property
    def input_channels(self) -> int:
        return self.depth * self.num_views

    @property
    def weight_shape(self) -> tuple[int, int, int, int]:
        return (
            self.num_views,
            self.input_channels,
            self.kernel_size,
            self.kernel_size,
        )


def lf_axial_profile(
    depth: int,
    *,
    depth_step_um: float = 0.5,
    fwhm_um: float = 40.62,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the legacy excitation-side ``LF_z_disV`` depth profile."""
    center = (depth - 1) / 2.0
    dz = (torch.arange(depth, device=device, dtype=dtype) - center) * depth_step_um
    sqrt_three = torch.sqrt(torch.tensor(3.0, device=device, dtype=dtype))
    profile = 1.0 / torch.sqrt(1.0 + (2.0 * sqrt_three * dz / fwhm_um) ** 2)
    return profile.reshape(1, depth, 1, 1)


class PhaseSpaceLFMForward(nn.Module):
    """
    Exact differentiable form of MATLAB ``convL1 -> convL2 -> convL3``.

    The one-hot first and third layers are implemented as tensor rearrangements.
    The fixed PSF layer is detection-only. A legacy excitation-side axial
    envelope can optionally be applied before it; that envelope remains a
    separate factor and is never embedded in the PSF weights. The weights are
    deliberately a non-persistent buffer so multi-gigabyte physics weights are
    not duplicated in H1 checkpoints.
    """

    def __init__(self, weight: torch.Tensor, config: LFMForwardConfig) -> None:
        super().__init__()
        if tuple(weight.shape) != config.weight_shape:
            raise ValueError(
                f"Expected forward weight {config.weight_shape}, got "
                f"{tuple(weight.shape)}"
            )
        if not weight.is_floating_point():
            raise TypeError("Forward weights must be floating point")
        self.config = config
        self.input_domain: ForwardInputDomain = config.input_domain
        self.register_buffer("weight", weight.detach(), persistent=False)
        self.register_buffer(
            "axial_profile",
            lf_axial_profile(
                config.depth,
                depth_step_um=config.depth_step_um,
                fwhm_um=config.axial_fwhm_um,
            ),
            persistent=True,
        )

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        if volume.ndim != 4 or volume.shape[1] != self.config.depth:
            raise ValueError(
                f"Expected volume [B, {self.config.depth}, H, W], got "
                f"{tuple(volume.shape)}"
            )
        if volume.device != self.weight.device:
            raise ValueError(
                f"Volume is on {volume.device}, but forward weights are on "
                f"{self.weight.device}"
            )
        if volume.shape[-2] % self.config.nnum or volume.shape[-1] % self.config.nnum:
            raise ValueError(
                f"Volume size {tuple(volume.shape[-2:])} must be divisible by "
                f"nnum={self.config.nnum}"
            )

        output_dtype = volume.dtype
        compute_volume = volume.to(dtype=self.weight.dtype)
        if self.config.apply_axial_profile:
            compute_volume = compute_volume * self.axial_profile.to(
                device=volume.device, dtype=self.weight.dtype
            )

        phase_depth = space_to_view_depth(compute_volume, self.config.nnum)
        projected_views = F.conv2d(
            phase_depth,
            self.weight,
            padding=self.config.kernel_size // 2,
        )
        lf = view_depth_to_space(
            projected_views,
            channels=1,
            block_size=self.config.nnum,
        )
        return lf.to(dtype=output_dtype)


def load_lfm_forward_binary(
    path: str | Path,
    *,
    config: LFMForwardConfig | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float16,
) -> PhaseSpaceLFMForward:
    """Load the raw float32 file written by ``export_lf_forward.m``."""
    config = config or LFMForwardConfig()
    path = Path(path)
    expected_values = int(np.prod(config.weight_shape))
    expected_bytes = expected_values * np.dtype("<f4").itemsize
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Forward-weight file has {actual_bytes} bytes; expected "
            f"{expected_bytes} for shape {config.weight_shape}"
        )

    # Copy-on-write mapping avoids loading a second host copy before transfer.
    mapped = np.memmap(
        path,
        mode="c",
        dtype="<f4",
        shape=config.weight_shape,
        order="C",
    )
    weight = torch.from_numpy(mapped).to(device=device, dtype=dtype)
    return PhaseSpaceLFMForward(weight, config)


def _phase_space_indices(
    sensor_size: int, nnum: int
) -> np.ndarray:
    if sensor_size % nnum:
        raise ValueError(
            f"LF PSF sensor size {sensor_size} is not divisible by nnum={nnum}"
        )
    kernel_size = sensor_size // nnum
    matlab_offset = int(np.ceil(((sensor_size + 1) % (2 * nnum)) / 2))
    indices = np.empty((nnum, nnum, kernel_size), dtype=np.int64)
    steps = nnum * np.arange(kernel_size)
    for source_view in range(nnum):
        for output_view in range(nnum):
            # Direct zero-based form of Hu/Hv in saveLF_PSF2CNN_20241203.m.
            start_one_based = output_view - source_view + matlab_offset
            values = start_one_based + steps
            values[values <= 0] += sensor_size
            values[values > sensor_size] -= sensor_size
            indices[source_view, output_view] = values[::-1] - 1
    return indices


def lf_psf_to_phase_space_weight(h: torch.Tensor) -> torch.Tensor:
    """Convert clean ``H[Y,X,U,V,Z]`` to the fixed convL2 tensor.

    This ports the indexing loop in ``saveLF_PSF2CNN_20241203.m`` and returns
    PyTorch order ``[output_view, input_view_depth, kernel_y, kernel_x]``.
    It is intended for parity tests and small PSFs; use the streamed exporter
    for a full-size HDF5 PSF.
    """
    if h.ndim != 5 or h.shape[0] != h.shape[1] or h.shape[2] != h.shape[3]:
        raise ValueError(f"Expected square H[Y,X,U,V,Z], got {tuple(h.shape)}")
    sensor_size, _, nnum, _, depth = h.shape
    indices = _phase_space_indices(sensor_size, nnum)
    kernel_size = sensor_size // nnum
    h_numpy = h.detach().to(device="cpu", dtype=torch.float32).numpy()
    weight = np.empty(
        (nnum**2, nnum**2 * depth, kernel_size, kernel_size),
        dtype=np.float32,
    )
    for u in range(nnum):
        rows = indices[u]
        for v in range(nnum):
            columns = indices[v]
            input_start = (u * nnum + v) * depth
            for z in range(depth):
                sampled = h_numpy[:, :, u, v, z][
                    rows[:, None, :, None], columns[None, :, None, :]
                ]
                weight[:, input_start + z] = sampled.reshape(
                    nnum**2, kernel_size, kernel_size
                )
    return torch.from_numpy(weight).to(dtype=h.dtype, device=h.device)


def lf_psf_view_energy_prior(h: torch.Tensor) -> torch.Tensor:
    """Return Equation-13-style phase-space energies from ``H[Y,X,U,V,Z]``.

    ``H`` is first sampled into the same phase-space convolution used by the
    fixed LF forward projector. L1 energy is then summed over every input
    source view, reconstruction depth, and spatial kernel location for each
    output component ``k``. The returned ``[U,V]`` array uses the same
    row-major flattening as :class:`~lfrln.physics_rearrange.LFToViews`.
    """
    if h.ndim != 5 or h.shape[0] != h.shape[1] or h.shape[2] != h.shape[3]:
        raise ValueError(f"Expected square H[Y,X,U,V,Z], got {tuple(h.shape)}")
    phase_space_weight = lf_psf_to_phase_space_weight(
        h.detach().to(device="cpu", dtype=torch.float64)
    )
    energy = phase_space_weight.abs().sum(dim=(1, 2, 3)).reshape(
        h.shape[2], h.shape[3]
    )
    total = energy.sum()
    if not torch.isfinite(total) or total <= 0:
        raise ValueError("LF PSF must have positive finite L1 energy")
    return energy / total


def export_lf_psf_view_energy_prior_from_h5(
    psf_path: str | Path,
    output_path: str | Path,
    *,
    dataset_name: str = "H",
    depth: int = 91,
    axis_order: PSFH5AxisOrder = "auto",
    overwrite: bool = False,
    progress: bool = True,
) -> np.ndarray:
    """Stream a full LF PSF and save its small 2-D angular-energy prior.

    ``yxuvz`` is the explicit layout written by the Python PSF generator.
    MATLAB ``-v7.3`` files expose the same logical array to h5py with reversed
    axes, ``zvuxy``. The JSON output stores both sum-one probabilities and the
    corresponding mean-one multiplicative weights used before H2 BP.
    """
    import h5py

    psf_path = Path(psf_path)
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if depth <= 0:
        raise ValueError("depth must be positive")
    if axis_order not in ("auto", "yxuvz", "zvuxy"):
        raise ValueError("axis_order must be 'auto', 'yxuvz', or 'zvuxy'")

    with h5py.File(psf_path, "r") as handle:
        if dataset_name not in handle:
            raise KeyError(f"{psf_path} has no dataset {dataset_name!r}")
        dataset = handle[dataset_name]
        if len(dataset.shape) != 5:
            raise ValueError(f"Expected a five-dimensional PSF, got {dataset.shape}")
        shape = tuple(int(value) for value in dataset.shape)
        resolved_order = axis_order
        if resolved_order == "auto":
            yxuvz = shape[0] == shape[1] and shape[2] == shape[3]
            zvuxy = shape[1] == shape[2] and shape[3] == shape[4]
            if yxuvz == zvuxy:
                raise ValueError(
                    f"Cannot infer PSF axis order unambiguously from {shape}; "
                    "pass axis_order explicitly"
                )
            resolved_order = "yxuvz" if yxuvz else "zvuxy"

        if resolved_order == "yxuvz":
            sensor_y, sensor_x, nnum_u, nnum_v, available_depth = shape
        else:
            available_depth, nnum_v, nnum_u, sensor_x, sensor_y = shape
        if sensor_y != sensor_x or nnum_u != nnum_v:
            raise ValueError(f"Expected square sensor/view axes, got {shape}")
        if depth > available_depth:
            raise ValueError(
                f"Requested depth={depth}, but PSF contains {available_depth} planes"
            )
        z_start = (available_depth - depth) // 2
        z_stop = z_start + depth
        indices = _phase_space_indices(sensor_y, nnum_u)
        energy = np.zeros((nnum_u, nnum_v), dtype=np.float64)

        if resolved_order == "yxuvz":
            iterator = range(nnum_u * nnum_v * depth)
            if progress:
                from tqdm.auto import tqdm

                iterator = tqdm(iterator, desc="PSF view energy")
            for linear_index in iterator:
                source_view, z_offset = divmod(linear_index, depth)
                u, v = divmod(source_view, nnum_v)
                plane = np.asarray(
                    dataset[:, :, u, v, z_start + z_offset],
                    dtype=np.float64,
                )
                rows = indices[u]
                columns = indices[v]
                sampled = plane[
                    rows[:, None, :, None], columns[None, :, None, :]
                ]
                energy += np.abs(sampled).sum(axis=(2, 3), dtype=np.float64)
        else:
            # MATLAB v7.3 chunks this file as [Z,V,U,1,1]. Reading one X
            # coordinate and all Y values at a time visits every compressed
            # chunk once while retaining only an 11x11 accumulator.
            # For each source view, map every logical sensor coordinate to
            # the output phase-space component selected by the exact MATLAB
            # indexing loop. Across output-view and kernel indices this is a
            # permutation of the complete sensor axis.
            coordinate_to_output = np.empty(
                (nnum_u, sensor_y), dtype=np.int64
            )
            for source_view in range(nnum_u):
                for output_view in range(nnum_u):
                    coordinate_to_output[
                        source_view, indices[source_view, output_view]
                    ] = output_view

            iterator = range(sensor_x)
            if progress:
                from tqdm.auto import tqdm

                iterator = tqdm(iterator, desc="PSF view energy")
            for x in iterator:
                slab = np.asarray(
                    dataset[z_start:z_stop, :, :, x, :],
                    dtype=np.float64,
                )
                # slab is [Z,source-V,source-U,Y]. Sum depth, then assign each
                # logical X/Y coordinate to its output phase-space V/U.
                source_energy = np.abs(slab).sum(axis=0, dtype=np.float64)
                for source_u in range(nnum_u):
                    output_u = coordinate_to_output[source_u]
                    for source_v in range(nnum_v):
                        output_v = coordinate_to_output[source_v, x]
                        energy[:, output_v] += np.bincount(
                            output_u,
                            weights=source_energy[source_v, source_u],
                            minlength=nnum_u,
                        )

    total = float(energy.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("LF PSF must have positive finite L1 energy")
    probabilities = energy / total
    mean_one = probabilities * probabilities.size
    payload = {
        "schema": "lfrln.psf_view_energy_prior.v2",
        "source_psf": str(psf_path.resolve()),
        "dataset": dataset_name,
        "source_h5_axis_order": resolved_order,
        "logical_view_axis_order": "UV",
        "energy_definition": (
            "L1 energy per output phase-space component after MATLAB "
            "LF-PSF rearrangement, summed over input view, depth, and kernel"
        ),
        "flatten_order": "row-major: view_id = u * nnum + v",
        "depth": int(depth),
        "depth_start_zero_based": int(z_start),
        "nnum": int(nnum_u),
        "probabilities_sum_one": probabilities.tolist(),
        "probabilities_flat": probabilities.reshape(-1).tolist(),
        "multiplicative_weights_mean_one": mean_one.tolist(),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return probabilities


def export_lfm_forward_from_h5(
    psf_path: str | Path,
    output_path: str | Path,
    *,
    dataset_name: str = "H",
    depth: int = 91,
    overwrite: bool = False,
    progress: bool = True,
) -> LFMForwardConfig:
    """Stream a Python-generated clean HDF5 PSF into convL2 binary order."""
    import h5py

    psf_path = Path(psf_path)
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(psf_path, "r") as handle:
        if dataset_name not in handle:
            raise KeyError(f"{psf_path} has no dataset {dataset_name!r}")
        h_dataset = handle[dataset_name]
        if len(h_dataset.shape) != 5:
            raise ValueError(f"Expected H[Y,X,U,V,Z], got {h_dataset.shape}")
        sensor_y, sensor_x, nnum_u, nnum_v, available_depth = h_dataset.shape
        if sensor_y != sensor_x or nnum_u != nnum_v:
            raise ValueError(f"Expected square sensor/view axes, got {h_dataset.shape}")
        if depth <= 0 or depth > available_depth:
            raise ValueError(
                f"Requested depth={depth}, but H contains {available_depth} planes"
            )
        kernel_size = sensor_y // nnum_u
        config = LFMForwardConfig(
            depth=depth,
            nnum=nnum_u,
            kernel_size=kernel_size,
            input_domain="raw",
        )
        indices = _phase_space_indices(sensor_y, nnum_u)
        z_start = (available_depth - depth) // 2
        weight = np.memmap(
            output_path,
            mode="w+",
            dtype="<f4",
            shape=config.weight_shape,
            order="C",
        )

        iterator = range(nnum_u * nnum_u * depth)
        if progress:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, desc="export detection LF projector")
        for linear_index in iterator:
            source_view, z = divmod(linear_index, depth)
            u, v = divmod(source_view, nnum_u)
            plane = np.asarray(
                h_dataset[:, :, u, v, z_start + z], dtype=np.float32
            )
            rows = indices[u]
            columns = indices[v]
            sampled = plane[
                rows[:, None, :, None], columns[None, :, None, :]
            ]
            weight[:, linear_index] = sampled.reshape(
                nnum_u**2, kernel_size, kernel_size
            )
        weight.flush()
        del weight

        psf_metadata: dict[str, object] = {}
        for key in (
            "aberration_model",
            "aberration_osa_indices",
            "aberration_coefficients_nm_rms",
            "aberration_source",
            "aberration_coefficient_column",
            "aberration_phase_sign",
            "view_symmetry_used",
            "wavelength_m",
            "sample_pixel_m",
            "volume_pixel_m",
        ):
            if key not in handle.attrs:
                continue
            value = handle.attrs[key]
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, np.generic):
                value = value.item()
            psf_metadata[key] = value

    metadata = {
        "sourcePath": str(psf_path.resolve()),
        "sourceDataset": dataset_name,
        "sourceDomain": "raw_emitter_intensity",
        "weightShapePyTorch": list(config.weight_shape),
        "dtype": "float32 little-endian",
        "depthSliceStartZeroBased": z_start,
        "scattering": False,
        "axialProfileEmbedded": False,
        "psfCalibration": psf_metadata,
    }
    output_path.with_suffix(output_path.suffix + ".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return config
