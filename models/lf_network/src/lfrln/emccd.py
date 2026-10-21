"""Intensity-dependent EMCCD noise using the measured MATLAB blank maps.

The blank calibration describes an effective Gaussian background core, not an
independently identified read-noise/dark-current decomposition. We add the EM
statistics of the *signal* to it; fitting another background from kappa3 would
double count part of the measured blank noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import scipy.io
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class EMCCDConfig:
    """G is effective ADU/input-electron, NOT the camera's EM500/800 setting.

    ``registered`` keeps the 517x517 projector's clean mean exactly, and uses
    a bilinearly lifted sensor intensity to approximate its shot covariance.
    ``sensor`` samples directly on the calibration grid with no such lift.
    No per-image normalization, quantization or ADC clipping is performed.

    Optional empirical_blank_path selects registered measured blank frames.
    This replaces Gaussian blank/common noise; signal Poisson--Gamma noise
    remains separate. The bank hash is required for reproducible profiles.
    """

    calibration_path: str
    gain_adu_per_electron: float
    input_domain: Literal["sensor", "registered"] = "registered"
    correlated: bool = True
    hot_pixels: bool = True
    sigma_scale: float = 1.0
    subtract_calibrated_offset: bool = True
    extra_background_electrons: float = 0.0
    excess_noise: Literal["gamma", "none"] = "gamma"
    empirical_blank_path: str | None = None
    empirical_blank_sha256: str | None = None
    registration_path: str | None = None
    registration_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "calibration_path", str(Path(self.calibration_path).resolve()))
        if self.registration_path is not None:
            object.__setattr__(self, 'registration_path', str(Path(self.registration_path).resolve()))
            if self.input_domain != 'registered' or not self.registration_sha256:
                raise ValueError('Explicit registration requires registered input and a hash')
        if self.empirical_blank_path is not None:
            object.__setattr__(self, "empirical_blank_path", str(Path(self.empirical_blank_path).resolve()))
            if self.input_domain != 'registered' or not self.empirical_blank_sha256 or self.sigma_scale != 1:
                raise ValueError('Empirical blanks require registered domain, a hash, and sigma_scale=1')
        if self.input_domain not in ("sensor", "registered"):
            raise ValueError("EMCCD input_domain must be 'sensor' or 'registered'")
        if self.excess_noise not in ("gamma", "none"):
            raise ValueError("excess_noise must be 'gamma' or 'none'")
        for name in ("gain_adu_per_electron", "sigma_scale"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (not math.isfinite(self.extra_background_electrons)
                or self.extra_background_electrons < 0):
            raise ValueError("extra_background_electrons must be finite and nonnegative")


def _resize_weights(n_in: int, n_out: int, kernel: str = 'bicubic') -> tuple[torch.Tensor, torch.Tensor]:
    """MATLAB bicubic resize with antialiasing and symmetric extension."""
    scale = n_out / n_in
    antialias = min(scale, 1.0)
    if kernel not in ('bilinear', 'bicubic'):
        raise ValueError('Resize kernel must be bilinear or bicubic')
    width = (2.0 if kernel == 'bilinear' else 4.0) / antialias
    centres = (torch.arange(n_out, dtype=torch.float64) + 0.5) / scale - 0.5
    left = torch.floor(centres - width / 2).long()
    indices = left[:, None] + torch.arange(math.ceil(width) + 2)[None, :]
    x = ((centres[:, None] - indices) * antialias).abs()
    weights = torch.where(
        x <= 1, 1.5 * x**3 - 2.5 * x**2 + 1,
        torch.where(x <= 2, -0.5 * x**3 + 2.5 * x**2 - 4 * x + 2, 0),
    ) * antialias
    if kernel == 'bilinear':
        weights = (1 - x).clamp_min(0) * antialias
    weights /= weights.sum(dim=1, keepdim=True)
    indices = indices.remainder(2 * n_in)
    indices = torch.where(indices < n_in, indices, 2 * n_in - 1 - indices)
    return indices, weights.float()


class LFNoiseRegistration(nn.Module):
    """Linear geometry of LF_preprocess, without clipping or normalization.

    Raw tensors use MATLAB's [1004,1002] matrix order. The rotation centre,
    resize rounding and asymmetric crop follow the supplied MATLAB helper.
    """

    sensor_shape = (1004, 1002)
    output_shape = (517, 517)

    def __init__(self, *, raw_affine_row_xy=None, resize_kernel='bicubic') -> None:
        super().__init__()
        indices, weights = _resize_weights(1002, 529, resize_kernel)
        self.register_buffer("indices", indices, persistent=False)
        self.register_buffer("weights", weights, persistent=False)
        forward, inverse = self._grid(-0.2), self._grid(0.2)
        if raw_affine_row_xy is not None:
            matrix = torch.as_tensor(raw_affine_row_xy, dtype=torch.float64)
            if matrix.shape != (3,3) or not torch.isfinite(matrix).all() or not torch.equal(matrix[:,2], torch.tensor([0.,0.,1.],dtype=torch.float64)):
                raise ValueError('Expected finite MATLAB 3x3 row affine matrix')
            forward = self._affine_grid(torch.linalg.inv(matrix))
            inverse = self._affine_grid(matrix)
        self.register_buffer("rotation_grid", forward, persistent=False)
        self.register_buffer("inverse_grid", inverse, persistent=False)
        # imwarp fills a sample outside the first/last pixel centre with zero.
        # grid_sample otherwise blends a partial border pixel into that sample.
        bound = 1 - 1/1002
        valid = (forward.abs() <= bound).all(-1) if raw_affine_row_xy is not None else None
        self.register_buffer('rotation_valid',valid,persistent=False)

    @staticmethod
    def _affine_grid(matrix):
        yy, xx = torch.meshgrid(torch.arange(1,1003,dtype=torch.float64),
                                torch.arange(1,1003,dtype=torch.float64),indexing='ij')
        q = torch.stack((xx,yy,torch.ones_like(xx)),dim=-1) @ matrix
        return (2*(q[...,:2]-.5)/1002-1).float()[None]

    @staticmethod
    def _grid(theta: float) -> torch.Tensor:
        # MATLAB world coordinates are one-based, with centre cols/2, rows/2.
        yy, xx = torch.meshgrid(torch.arange(1, 1003, dtype=torch.float64),
                                torch.arange(1, 1003, dtype=torch.float64), indexing="ij")
        angle = math.radians(theta)
        xin = math.cos(angle) * (xx - 501) - math.sin(angle) * (yy - 501) + 501
        yin = math.sin(angle) * (xx - 501) + math.cos(angle) * (yy - 501) + 501
        return torch.stack((2 * (xin - 0.5) / 1002 - 1,
                            2 * (yin - 0.5) / 1002 - 1), dim=-1).float()[None]

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-2:] != self.sensor_shape:
            raise ValueError(f"Expected sensor shape {self.sensor_shape}, got {value.shape[-2:]}")
        shape = value.shape[:-2]
        v = value.float().reshape(-1, 1, *self.sensor_shape)
        v = torch.rot90(v[..., 1:1003, :].flip(-1), 1, (-2, -1))
        v = F.grid_sample(v, self.rotation_grid.expand(v.shape[0], -1, -1, -1),
                          mode="bilinear", padding_mode="zeros", align_corners=False)
        if self.rotation_valid is not None:
            v = v * self.rotation_valid[:,None]
        v = (v[:, :, self.indices, :] * self.weights[None, None, :, :, None]).sum(dim=3)
        v = (v[..., self.indices] * self.weights[None, None, None, :, :]).sum(dim=-1)
        return v[..., 2:-10, 5:-7].reshape(*shape, *self.output_shape)

    def lift(self, value: torch.Tensor) -> torch.Tensor:
        """Approximate sensor mean; NOT an inverse of the optical projector."""
        if value.shape[-2:] != self.output_shape:
            raise ValueError(f"Expected registered shape {self.output_shape}, got {value.shape[-2:]}")
        shape = value.shape[:-2]
        v = value.float().reshape(-1, 1, *self.output_shape)
        v = F.pad(v, (5, 7, 2, 10), mode="replicate")
        v = F.interpolate(v, size=(1002, 1002), mode="bilinear", align_corners=False)
        v = F.grid_sample(v, self.inverse_grid.expand(v.shape[0], -1, -1, -1),
                          mode="bilinear", padding_mode="border", align_corners=False)
        v = torch.rot90(v, -1, (-2, -1)).flip(-1)
        return F.pad(v, (0, 0, 1, 1), mode="replicate").reshape(*shape, *self.sensor_shape)


def _load_calibration(path: str) -> dict[str, np.ndarray]:
    fields = ("offset", "sigmaRead", "sigmaTotal", "sigmaCommon", "isHot", "acf1", "acf2")
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as handle:
            cal = handle["cal"]
            return {key: np.asarray(cal[key]).T.copy() for key in fields if key in cal}
    cal = scipy.io.loadmat(path, simplify_cells=True)["cal"]
    return {key: np.asarray(cal[key]) for key in fields if key in cal}


def _colour_filter(acf: np.ndarray, n: int) -> torch.Tensor:
    a = torch.as_tensor(acf.reshape(-1), dtype=torch.float32)
    if not torch.isfinite(a).all() or a.numel() == 0 or a[0] <= 0:
        raise ValueError("Invalid calibration autocorrelation")
    a = a / a[0]
    length = min(a.numel() - 1, (n - 1) // 2)
    r = torch.zeros(n)
    r[0] = 1
    if length:
        r[1:length + 1] = a[1:length + 1]
        r[-length:] = a[1:length + 1].flip(0)
    power = torch.fft.fft(r).real.clamp_min(0)
    return (power / power.mean()).sqrt()


class EMCCDNoiseModel(nn.Module):
    """Measured spatial background plus Poisson--Gamma signal noise.

    Common-mode samples are independent across images. This models single
    frames, not the measured drift trajectories of a time sequence. Hot-pixel
    std substitution reproduces variance, not rare-event tail distributions.
    """

    def __init__(self, config: EMCCDConfig) -> None:
        super().__init__()
        self.config = config
        cal = _load_calibration(config.calibration_path)
        with open(config.calibration_path, "rb") as source:
            self.calibration_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
        required = {"offset", "sigmaRead", "sigmaCommon"}
        if config.hot_pixels:
            required |= {"isHot", "sigmaTotal"}
        if config.correlated:
            required |= {"acf1", "acf2"}
        missing = required - cal.keys()
        if missing:
            raise ValueError(f"Calibration missing {sorted(missing)}")
        offset = np.asarray(cal["offset"], dtype=np.float32)
        sigma = np.asarray(cal["sigmaRead"], dtype=np.float32).copy()
        if offset.ndim != 2 or sigma.shape != offset.shape:
            raise ValueError("Calibration offset and sigmaRead must be matching 2D maps")
        if config.hot_pixels:
            if any(cal[k].shape != offset.shape for k in ("isHot", "sigmaTotal")):
                raise ValueError("Hot-pixel maps must match calibration shape")
            hot = cal["isHot"].astype(bool)
            sigma[hot] = cal["sigmaTotal"][hot]
        common = float(np.asarray(cal["sigmaCommon"]).squeeze())
        if (not np.isfinite(offset).all() or not np.isfinite(sigma).all()
                or np.any(sigma < 0) or not math.isfinite(common) or common < 0):
            raise ValueError("Calibration maps must be finite with nonnegative noise std")
        self.register_buffer("offset", torch.from_numpy(offset), persistent=False)
        self.register_buffer("sigma", torch.from_numpy(sigma) * config.sigma_scale, persistent=False)
        self.sigma_common = common
        geometry = {}
        if config.registration_path:
            with open(config.registration_path, 'rb') as source:
                if hashlib.file_digest(source,'sha256').hexdigest() != config.registration_sha256:
                    raise ValueError('Noise registration changed')
            geometry = json.loads(Path(config.registration_path).read_text(encoding='utf-8'))
            geometry = {k:geometry[k] for k in ('raw_affine_row_xy','resize_kernel')}
        self.registration = LFNoiseRegistration(**geometry) if config.input_domain == "registered" else None
        if self.registration is not None and tuple(offset.shape) != self.registration.sensor_shape:
            raise ValueError("Registered EMCCD simulation requires a 1004x1002 calibration")
        h, w = offset.shape
        for axis, size in ((1, h), (2, w)):
            filt = _colour_filter(cal[f"acf{axis}"], size) if config.correlated else torch.ones(size)
            self.register_buffer(f"filter{axis}", filt, persistent=False)
        baseline = self.registration(self.offset) if self.registration else self.offset.clone()
        self.register_buffer("baseline", baseline, persistent=False)
        flat = torch.ones_like(self.offset)
        flat = self.registration(flat) if self.registration else flat
        self.register_buffer("flat_response", flat, persistent=False)
        self.empirical_blanks = None
        if config.empirical_blank_path:
            with open(config.empirical_blank_path, 'rb') as stream:
                bank_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
            if bank_hash != config.empirical_blank_sha256:
                raise ValueError('Empirical blank bank changed')
            bank = np.load(config.empirical_blank_path, mmap_mode='r', allow_pickle=False)
            if bank.dtype != np.float32 or bank.ndim != 3 or tuple(bank.shape[1:]) != tuple(baseline.shape) or len(bank) < 2 or not np.isfinite(bank).all():
                raise ValueError('Invalid registered empirical blank bank')
            self.empirical_blanks = bank
            self.register_buffer('empirical_blank_mean', torch.from_numpy(np.asarray(bank).mean(0).copy()), persistent=False)

    def forward(
        self, clean_lf: torch.Tensor, *, electrons_per_lf_unit: float,
        signal_variation: tuple[float, float] = (1, 1),
        background_variation: tuple[float, float] = (1, 1),
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if clean_lf.ndim < 2 or not clean_lf.is_floating_point():
            raise TypeError("EMCCD input must be a floating-point image or image batch")
        expected_shape = self.baseline.shape
        if clean_lf.shape[-2:] != expected_shape:
            raise ValueError(f"EMCCD {self.config.input_domain} input requires {tuple(expected_shape)}; got {tuple(clean_lf.shape[-2:])}")
        if not torch.isfinite(clean_lf).all() or torch.any(clean_lf < 0):
            raise ValueError("Clean LF must be finite and nonnegative")
        if not math.isfinite(electrons_per_lf_unit) or electrons_per_lf_unit < 0:
            raise ValueError("electrons_per_lf_unit must be finite and nonnegative")
        original_shape = clean_lf.shape
        clean = clean_lf.float().reshape(-1, *expected_shape)
        n = clean.shape[0]

        def variation(interval: tuple[float, float]) -> torch.Tensor:
            low, high = interval
            if not (math.isfinite(low) and math.isfinite(high) and 0 < low <= high):
                raise ValueError("Variation intervals must be finite, positive and ordered")
            if low == high:
                return clean.new_full((n, 1, 1), low)
            return low + (high - low) * torch.rand((n, 1, 1), device=clean.device, generator=generator)

        signal_factor = variation(signal_variation)
        background_factor = variation(background_variation)
        signal = clean * electrons_per_lf_unit * signal_factor
        sensor_signal = self.registration.lift(signal) if self.registration else signal
        background = self.config.extra_background_electrons * background_factor
        rate = sensor_signal + background
        electrons = torch.poisson(rate, generator=generator)
        # torch.distributions.Gamma does not accept a Generator. Its underlying
        # PyTorch sampler does, so deterministic dataset workers remain local.
        if self.config.excess_noise == "gamma":
            multiplied = torch._standard_gamma(electrons, generator=generator)
            multiplied = torch.where(electrons > 0, multiplied, 0)
        else:
            multiplied = electrons
        multiplied = multiplied * self.config.gain_adu_per_electron
        residual = multiplied - rate * self.config.gain_adu_per_electron
        if self.empirical_blanks is None:
            white = torch.randn(rate.shape, device=clean.device, generator=generator)
            if self.config.correlated:
                white = torch.fft.ifft2(torch.fft.fft2(white)
                                        * self.filter1[:, None] * self.filter2[None, :]).real
            residual = residual + white * self.sigma
            common = torch.randn((n, 1, 1), device=clean.device, generator=generator) * self.sigma_common
            residual = residual + common
        if self.registration:
            residual = self.registration(residual)
        mean = (self.baseline + self.config.gain_adu_per_electron
                * (signal + background * self.flat_response))
        blank_indices = None
        if self.empirical_blanks is not None:
            blank_indices = torch.randint(len(self.empirical_blanks), (n,), device=clean.device, generator=generator).cpu().tolist()
            # Whole registered blank frames retain skewness, hot events, 2D
            # covariance and common offsets. Do not add Gaussian blank noise again.
            blank = torch.as_tensor(np.asarray(self.empirical_blanks[blank_indices]).copy(), device=clean.device)
            residual = residual + blank - self.baseline
        observed = (mean + residual).reshape(original_shape)
        metadata = {
            "mode": "emccd", "calibration_path": self.config.calibration_path,
            "calibration_sha256": self.calibration_sha256,
            "gain_adu_per_electron": self.config.gain_adu_per_electron,
            "excess_noise": self.config.excess_noise,
            "excess_noise_factor_squared": 2. if self.config.excess_noise == "gamma" else 1.,
            "shot_variance_slope_adu": self.config.gain_adu_per_electron * (2 if self.config.excess_noise == "gamma" else 1),
            "electrons_per_lf_unit": electrons_per_lf_unit,
            "gain_factor": signal_factor.flatten().tolist(),
            "background_factor": background_factor.flatten().tolist(),
            "input_domain": self.config.input_domain,
            "shot_noise_model": ("registered_mean_preserving_sensor_lift" if self.registration
                                 else f"sensor_poisson_{self.config.excess_noise}"),
            "background_model": "empirical_registered_blank_frames" if self.empirical_blanks is not None else "calibrated_gaussian_blank_core",
            "empirical_blank_indices": blank_indices,
            "empirical_blank_sha256": self.config.empirical_blank_sha256,
            "registration_sha256": self.config.registration_sha256,
            "common_mode_model": "independent_per_frame",
        }
        return observed, metadata

    def expected_counts(self, clean_lf: torch.Tensor, metadata: dict[str, Any]) -> torch.Tensor:
        factors = clean_lf.new_tensor(metadata["gain_factor"]).reshape(-1, 1, 1)
        bg_factors = clean_lf.new_tensor(metadata["background_factor"]).reshape(-1, 1, 1)
        mean = (self.baseline + self.config.gain_adu_per_electron * (
            clean_lf.float().reshape(-1, *self.baseline.shape)
            * metadata["electrons_per_lf_unit"] * factors
            + self.config.extra_background_electrons * bg_factors * self.flat_response))
        if self.empirical_blanks is not None:
            mean = mean + self.empirical_blank_mean - self.baseline
        return mean.reshape(clean_lf.shape)


@lru_cache(maxsize=6)
def get_emccd_model(config: EMCCDConfig, device: str = "cpu") -> EMCCDNoiseModel:
    """Cache calibration maps/geometry, never random samples or RNG state."""
    return EMCCDNoiseModel(config).to(device)
