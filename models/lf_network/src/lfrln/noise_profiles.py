"""Portable camera/preprocessing profiles shared by generation and training."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

from .emccd import EMCCDConfig
from .synthetic import CameraConfig, LFPreprocessConfig


def load_noise_profile(path: str | Path) -> tuple[CameraConfig, LFPreprocessConfig, dict]:
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "lfrln.noise_profile.v1":
        raise ValueError(f"Unsupported noise profile schema: {path}")
    camera = dict(payload["camera_config"])
    emccd = dict(camera["emccd"])
    calibration = Path(emccd["calibration_path"])
    if not calibration.is_absolute():
        calibration = path.parent / calibration
    emccd["calibration_path"] = str(calibration.resolve())
    if emccd.get('registration_path'):
        registration = Path(emccd['registration_path'])
        emccd['registration_path'] = str((path.parent / registration).resolve() if not registration.is_absolute() else registration.resolve())
    if emccd.get('empirical_blank_path'):
        bank = Path(emccd['empirical_blank_path'])
        emccd['empirical_blank_path'] = str((path.parent / bank).resolve() if not bank.is_absolute() else bank.resolve())
    with calibration.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != payload["calibration_sha256"]:
        raise ValueError(f"Calibration changed since this noise profile was prepared: {calibration}")
    camera["emccd"] = EMCCDConfig(**emccd)
    camera = CameraConfig(**camera)
    preprocessing = LFPreprocessConfig(**payload["preprocess_config"])
    if not camera.emccd.subtract_calibrated_offset or preprocessing.subtract_background != 0:
        raise ValueError("Noise profiles require calibrated-offset subtraction with no second baseline")
    return camera, preprocessing, {**payload, "profile_path": str(path)}


def with_signal_adu_scale(camera: CameraConfig, preprocessing: LFPreprocessConfig,
                          adu_per_lf_unit: float) -> tuple[CameraConfig, LFPreprocessConfig]:
    """Change acquisition brightness while retaining the optical network units."""
    camera = replace(camera, signal_adu_per_lf_unit=adu_per_lf_unit)
    return camera, replace(preprocessing, count_scale=adu_per_lf_unit)
