from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .data import load_lf_tensor, load_volume_tensor
from .emccd import get_emccd_model
from .synthetic import (
    PHANTOM_KINDS,
    CameraConfig,
    LFPreprocessConfig,
    PhantomConfig,
    PhantomKind,
    generate_phantom,
    preprocess_lf_counts,
    simulate_camera,
)
from .synthetic_io import load_synthetic_sample


class PairedMatDataset(Dataset[dict[str, torch.Tensor | str | bool]]):
    """Paired ``LF/<name>.mat`` and ``GT/<name>.mat`` H1 dataset."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        split: str,
        lf_variable: str = "LF",
        gt_variable: str = "GT_emPSF",
        raw_directory: str | Path | None = None,
        raw_variable: str = "GT",
        raw_feedback_compatible: bool | None = None,
        augment: bool = False,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.lf_variable = lf_variable
        self.gt_variable = gt_variable
        self.raw_variable = raw_variable
        self.raw_feedback_compatible = raw_feedback_compatible
        self.augment = augment

        lf_directory = self.dataset_root / split / "LF"
        gt_directory = self.dataset_root / split / "GT"
        if not lf_directory.is_dir() or not gt_directory.is_dir():
            raise FileNotFoundError(
                f"Expected paired directories {lf_directory} and {gt_directory}"
            )

        lf_files = {path.name: path for path in lf_directory.glob("*.mat")}
        gt_files = {path.name: path for path in gt_directory.glob("*.mat")}
        missing_gt = sorted(set(lf_files) - set(gt_files))
        missing_lf = sorted(set(gt_files) - set(lf_files))
        if missing_gt or missing_lf:
            raise ValueError(
                "LF/GT filenames do not match. "
                f"Missing GT={missing_gt}; missing LF={missing_lf}"
            )
        if not lf_files:
            raise ValueError(f"No .mat files found in {lf_directory}")

        raw_files: dict[str, Path] = {}
        if raw_directory is not None:
            raw_path = Path(raw_directory)
            if not raw_path.is_dir():
                raise FileNotFoundError(f"Raw GT directory does not exist: {raw_path}")
            raw_files = {path.name: path for path in raw_path.glob("*.mat")}
            missing_raw = sorted(set(lf_files) - set(raw_files))
            if missing_raw:
                raise ValueError(
                    f"Raw GT directory is missing paired files: {missing_raw}"
                )
        self.pairs = [
            (lf_files[name], gt_files[name], raw_files.get(name))
            for name in sorted(lf_files)
        ]

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _augment_tensors(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # Uniform sample from the eight rotations/reflections of a square.
        transform = int(torch.randint(0, 8, size=()).item())
        rotations = transform % 4
        output = []
        for tensor in tensors:
            if rotations:
                tensor = torch.rot90(tensor, rotations, dims=(-2, -1))
            if transform >= 4:
                tensor = torch.flip(tensor, dims=(-1,))
            output.append(tensor.contiguous())
        return tuple(output)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | bool]:
        lf_path, gt_path, raw_path = self.pairs[index]
        lf = load_lf_tensor(lf_path, self.lf_variable)
        gt = load_volume_tensor(gt_path, self.gt_variable)
        raw = (
            load_volume_tensor(raw_path, self.raw_variable)
            if raw_path is not None
            else None
        )
        if self.augment:
            augmented = self._augment_tensors(
                *((lf, gt) if raw is None else (lf, gt, raw))
            )
            lf, gt = augmented[:2]
            if raw is not None:
                raw = augmented[2]
        result: dict[str, torch.Tensor | str | bool] = {
            "lf": lf,
            "gt": gt,
            "name": lf_path.name,
        }
        if raw is not None:
            result["raw"] = raw
            result["cutoff"] = gt
            result["lf_feedback"] = lf
        if self.raw_feedback_compatible is not None:
            result["raw_feedback_compatible"] = self.raw_feedback_compatible
        return result


class PhantomDataset(Dataset[dict[str, torch.Tensor | str | dict]]):
    """Deterministic, balanced raw phantoms generated directly in Python."""

    def __init__(
        self,
        length: int,
        *,
        config: PhantomConfig | None = None,
        base_seed: int = 20260217,
        kinds: tuple[PhantomKind, ...] = PHANTOM_KINDS,
    ) -> None:
        if length <= 0:
            raise ValueError("length must be positive")
        if not kinds:
            raise ValueError("At least one phantom kind is required")
        self.length = length
        self.config = config or PhantomConfig()
        self.base_seed = base_seed
        self.kinds = kinds

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | dict]:
        if index < 0:
            index += self.length
        if index < 0 or index >= self.length:
            raise IndexError(index)
        kind = self.kinds[index % len(self.kinds)]
        raw, metadata = generate_phantom(
            self.config,
            seed=self.base_seed + index,
            kind=kind,
        )
        return {
            "raw": raw,
            "name": f"python_{index:06d}_{kind}",
            "metadata": metadata,
        }


class SyntheticPairDataset(Dataset[dict[str, torch.Tensor | str | dict]]):
    """Load portable HDF5 or legacy PyTorch synthetic LF/volume pairs.

    A split must contain only one sample format. Each file is opened and
    closed inside ``__getitem__``, which is safe with multi-worker PyTorch
    data loaders and avoids retaining one HDF5 handle per worker.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        split: str,
        augment: bool = False,
        dtype: torch.dtype = torch.float32,
        include_metadata: bool = True,
        sample_format: str = "auto",
        lf_scale: float = 1.0,
    ) -> None:
        self.directory = Path(dataset_root) / split
        if not self.directory.is_dir():
            raise FileNotFoundError(f"Synthetic split does not exist: {self.directory}")
        if sample_format not in ("auto", "h5", "pt"):
            raise ValueError("sample_format must be 'auto', 'h5', or 'pt'")
        if not math.isfinite(lf_scale) or lf_scale <= 0:
            raise ValueError("lf_scale must be finite and positive")
        files_by_format = {
            "h5": sorted(
                (*self.directory.glob("*.h5"), *self.directory.glob("*.hdf5"))
            ),
            "pt": sorted(self.directory.glob("*.pt")),
        }
        if sample_format == "auto":
            available = [name for name, files in files_by_format.items() if files]
            if len(available) > 1:
                raise ValueError(
                    f"Mixed HDF5 and .pt samples in {self.directory}; select one "
                    "format explicitly or keep each split format-homogeneous"
                )
            selected_format = available[0] if available else "h5"
        else:
            selected_format = sample_format
        self.files = files_by_format[selected_format]
        if not self.files:
            raise ValueError(
                f"No {selected_format} samples found in {self.directory}"
            )
        self.sample_format = selected_format
        self.augment = augment
        self.dtype = dtype
        self.include_metadata = include_metadata
        self.lf_scale = float(lf_scale)

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def _apply_transform(tensor: torch.Tensor, transform: int) -> torch.Tensor:
        rotations = transform % 4
        if rotations:
            tensor = torch.rot90(tensor, rotations, dims=(-2, -1))
        if transform >= 4:
            tensor = torch.flip(tensor, dims=(-1,))
        return tensor.contiguous()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | dict]:
        path = self.files[index]
        payload = load_synthetic_sample(
            path, include_metadata=self.include_metadata
        )
        required = {"lf", "gt", "lf_feedback"}
        missing = required - set(payload)
        if missing:
            raise ValueError(f"{path} is missing required tensors {sorted(missing)}")
        result: dict[str, torch.Tensor | str | dict] = {
            "name": payload.get("name", path.stem),
            "synthetic_split": self.directory.name,
        }
        if self.include_metadata:
            result["metadata"] = payload.get("metadata", {})
        if "feedback_compatible" in payload:
            result["feedback_compatible"] = bool(payload["feedback_compatible"])
        if "raw_feedback_compatible" in payload:
            result["raw_feedback_compatible"] = bool(
                payload["raw_feedback_compatible"]
            )
        transform = int(torch.randint(0, 8, size=()).item()) if self.augment else 0
        result["synthetic_transform"] = transform
        tensor_keys = required | ({"raw", "cutoff", "lf_counts", "lf_expected_counts"} & set(payload))
        for key in tensor_keys:
            tensor = payload[key].to(dtype=self.dtype)
            if key == "lf" and self.lf_scale != 1.0:
                tensor = tensor * self.lf_scale
            result[key] = self._apply_transform(tensor, transform)
        return result


class MixedLFNoiseDataset(Dataset[dict[str, Any]]):
    """Apply count-domain camera noise to a configurable fraction of LF inputs.

    The wrapped dataset remains responsible for loading and geometric
    augmentation. Noise is applied afterwards to ``lf_feedback`` when that
    clean linear measurement is available, otherwise to ``lf``. Supervised
    targets are never changed.

    Augmentation deliberately happens BEFORE the camera model. The sensor's
    per-pixel calibration and its anisotropic readout correlation are fixed in
    sensor coordinates, so noising the already-rotated light field leaves that
    structure on the same axis the network will see at inference. Measured in
    the registered domain the anisotropy is small in any case -- lag-1 0.677
    along rows against 0.662 along columns -- because the bicubic 1002->529
    downsample dominates and is isotropic.

    With ``deterministic=True``, both the noisy subset and camera realization are fixed by ``seed + index``; this is
    intended for reproducible validation and testing.

    ``peak_signal_adu_range`` optionally samples log-uniform synthetic signal
    peaks in EMCCD mode. It changes illumination and the known count divisor
    together, preserving expected input units before clamping. This is based
    on the clean phantom, never the maximum of an observed noisy frame.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        noise_probability: float,
        camera_config: CameraConfig,
        preprocess_config: LFPreprocessConfig,
        deterministic: bool = False,
        seed: int = 0,
        clean_lf_key: str = "lf_feedback",
        peak_signal_adu_range: tuple[float, float] | None = None,
    ) -> None:
        if not 0.0 <= noise_probability <= 1.0:
            raise ValueError("noise_probability must be between zero and one")
        self.dataset = dataset
        self.noise_probability = float(noise_probability)
        self.camera_config = camera_config
        self.preprocess_config = preprocess_config
        self.deterministic = deterministic
        self.seed = int(seed)
        self.clean_lf_key = clean_lf_key
        self.peak_signal_adu_range = peak_signal_adu_range
        if peak_signal_adu_range is not None:
            lo, hi = peak_signal_adu_range
            if not (math.isfinite(lo) and math.isfinite(hi) and 0 < lo <= hi):
                raise ValueError("peak_signal_adu_range must be finite, positive and ordered")
            if (camera_config.mode != "emccd" or camera_config.signal_adu_per_lf_unit is None
                    or tuple(camera_config.gain_variation) != (1., 1.)
                    or preprocess_config.subtract_background != 0
                    or not camera_config.emccd.subtract_calibrated_offset):
                raise ValueError("Peak ADU sampling requires fixed-gain EMCCD ADU mode and calibrated offset subtraction")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.dataset[index])
        if "lf" not in sample:
            raise ValueError("Wrapped LF dataset sample is missing 'lf'")
        clean_lf = sample.get(self.clean_lf_key, sample["lf"])
        if not isinstance(clean_lf, torch.Tensor) or not clean_lf.is_floating_point():
            raise TypeError("Clean LF input must be a floating-point tensor")

        generator = None
        if self.deterministic:
            if clean_lf.device.type != "cpu":
                raise ValueError("Deterministic dataset noise expects CPU tensors")
            generator = torch.Generator().manual_seed(self.seed + index)

        if self.noise_probability == 0.0:
            apply_noise = False
        elif self.noise_probability == 1.0:
            apply_noise = True
        else:
            apply_noise = bool(
                torch.rand((), generator=generator, device=clean_lf.device)
                < self.noise_probability
            )

        camera = self.camera_config
        preprocessing = self.preprocess_config
        if self.peak_signal_adu_range is not None:
            # A controlled SYNTHETIC illumination draw, derived from the known
            # clean LF. Never divide a measured/noisy image by its observed peak.
            # Scale S and the count divisor together: E[input] and all clean
            # supervision retain their original units, while SNR changes.
            peak = float(clean_lf.max())
            if not math.isfinite(peak) or peak < 0:
                raise ValueError("Clean LF peak must be finite and nonnegative")
            desired_peak = 0.
            if apply_noise and peak > 0:
                lo, hi = self.peak_signal_adu_range
                u = float(torch.rand((), generator=generator, device=clean_lf.device)) if hi > lo else 0.
                desired_peak = math.exp(math.log(lo) + u * math.log(hi / lo))
                signal_scale = desired_peak / peak
                preprocessing = replace(preprocessing, count_scale=preprocessing.count_scale
                    * signal_scale / camera.signal_adu_per_lf_unit)
                camera = replace(camera, signal_adu_per_lf_unit=signal_scale,
                    photons_per_lf_unit=signal_scale / camera.emccd.gain_adu_per_electron)
            sample["lf_noise_peak_signal_adu"] = desired_peak
            sample["lf_noise_signal_adu_per_unit"] = camera.signal_adu_per_lf_unit
            sample["lf_noise_count_scale"] = preprocessing.count_scale
        if apply_noise:
            counts, camera_metadata = simulate_camera(
                clean_lf,
                camera,
                generator=generator,
            )
            sample["lf"] = preprocess_lf_counts(
                counts, preprocessing, camera_config=camera,
            )
            # If diagnostics were loaded, keep them paired with the *new*
            # observation. Never leave stale counts attached to an augmented LF.
            if "lf_counts" in sample:
                sample["lf_counts"] = counts
            if "lf_expected_counts" in sample:
                if camera.mode == "emccd":
                    model = get_emccd_model(camera.emccd, str(clean_lf.device))
                    sample["lf_expected_counts"] = model.expected_counts(clean_lf, camera_metadata)
                else:
                    sample["lf_expected_counts"] = (
                        clean_lf.clamp_min(0) * camera.photons_per_lf_unit
                        * camera_metadata["gain_factor"] + camera_metadata["background_mean"]
                    )
            if "metadata" in sample:
                if self.peak_signal_adu_range is not None:
                    camera_metadata = {**camera_metadata,
                        "peak_signal_adu_range": self.peak_signal_adu_range,
                        "sampled_peak_signal_adu": desired_peak,
                        "input_count_scale": preprocessing.count_scale,
                        "brightness_mode": "known_clean_peak_illumination; preserves expected input units"}
                sample["metadata"] = {**sample["metadata"], "camera": camera_metadata}
            # Keep lf_feedback in its clean optical units. Camera exposure and
            # gain belong to the observation model, not that supervision target.
        sample["lf_noise_applied"] = apply_noise
        return sample


class SyntheticPtDataset(SyntheticPairDataset):
    """Backward-compatible name; now reads both HDF5 and legacy ``.pt``."""
