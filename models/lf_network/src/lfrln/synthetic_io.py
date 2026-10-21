from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Mapping

import h5py
import numpy as np
import torch


SyntheticSampleFormat = Literal["h5", "pt"]
H5Compression = Literal["none", "gzip"]
StorageDType = Literal["float16", "float32"]

H5_SAMPLE_SCHEMA = "lfrln.synthetic.sample.v1"
SYNTHETIC_TENSOR_KEYS = (
    "lf",
    "gt",
    "lf_feedback",
    "raw",
    "cutoff",
    "lf_counts",
    "lf_expected_counts",
)
REQUIRED_SYNTHETIC_TENSORS = frozenset(("lf", "gt", "lf_feedback"))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot encode {type(value).__name__} as JSON")


def _decode_h5_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _axis_order(key: str, ndim: int) -> str:
    if ndim == 3 and key in ("lf", "lf_feedback", "lf_counts", "lf_expected_counts"):
        return "CYX"
    if ndim == 3:
        return "ZYX"
    return "".join(f"D{index}" for index in range(ndim))


def _as_storage_array(value: torch.Tensor, storage_dtype: StorageDType) -> np.ndarray:
    array = value.detach().cpu().numpy()
    if np.issubdtype(array.dtype, np.floating):
        dtype = np.float32 if storage_dtype == "float32" else np.float16
        array = array.astype(dtype, copy=False)
    return np.ascontiguousarray(array)


def save_synthetic_sample_h5(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    storage_dtype: StorageDType = "float32",
    compression: H5Compression = "none",
    compression_level: int = 1,
) -> None:
    """Write one portable synthetic LF/volume pair to HDF5.

    Tensor dimensions are stored in Python order and labeled explicitly as
    ``CYX`` (LF images) or ``ZYX`` (volumes). Uncompressed storage is the
    training-oriented default; GZIP remains available for portable archival.
    """
    destination = Path(path)
    if destination.suffix.lower() not in (".h5", ".hdf5"):
        raise ValueError("Portable synthetic samples must use .h5 or .hdf5")
    if storage_dtype not in ("float16", "float32"):
        raise ValueError("storage_dtype must be 'float16' or 'float32'")
    if compression not in ("none", "gzip"):
        raise ValueError("compression must be 'none' or 'gzip'")
    if not 0 <= compression_level <= 9:
        raise ValueError("compression_level must be between 0 and 9")

    missing = REQUIRED_SYNTHETIC_TENSORS - set(payload)
    if missing:
        raise ValueError(f"Synthetic sample is missing tensors {sorted(missing)}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(destination, "w") as handle:
        handle.attrs["schema"] = H5_SAMPLE_SCHEMA
        handle.attrs["name"] = str(payload.get("name", destination.stem))
        handle.attrs["storage_dtype"] = storage_dtype
        handle.attrs["tensor_dimension_convention"] = "Python/C order"
        metadata = payload.get("metadata", {})
        handle.attrs["metadata_json"] = json.dumps(
            metadata,
            ensure_ascii=False,
            separators=(",", ":"),
            default=_json_default,
        )
        for key in ("feedback_compatible", "raw_feedback_compatible"):
            if key in payload:
                handle.attrs[key] = np.uint8(bool(payload[key]))

        for key in SYNTHETIC_TENSOR_KEYS:
            value = payload.get(key)
            if value is None:
                continue
            if not torch.is_tensor(value):
                raise TypeError(f"payload[{key!r}] must be a torch.Tensor")
            # ADU diagnostics can exceed float16's range or lose sub-count
            # residuals. Keep counts in float32 even with half-precision volumes.
            array = _as_storage_array(
                value, "float32" if key in ("lf_counts", "lf_expected_counts") else storage_dtype,
            )
            options: dict[str, Any] = {}
            if compression == "gzip":
                options.update(
                    compression="gzip",
                    compression_opts=compression_level,
                    shuffle=True,
                )
            dataset = handle.create_dataset(key, data=array, **options)
            dataset.attrs["axis_order"] = _axis_order(key, array.ndim)


def load_synthetic_sample_h5(
    path: str | Path,
    *,
    include_metadata: bool = True,
) -> dict[str, Any]:
    """Read a sample written by :func:`save_synthetic_sample_h5`."""
    source = Path(path)
    with h5py.File(source, "r") as handle:
        schema = _decode_h5_text(handle.attrs.get("schema", ""))
        if schema != H5_SAMPLE_SCHEMA:
            raise ValueError(
                f"Unsupported HDF5 sample schema {schema!r} in {source}"
            )
        missing = REQUIRED_SYNTHETIC_TENSORS - set(handle.keys())
        if missing:
            raise ValueError(f"{source} is missing datasets {sorted(missing)}")
        payload: dict[str, Any] = {
            "name": _decode_h5_text(handle.attrs.get("name", source.stem)),
        }
        if include_metadata:
            metadata_json = _decode_h5_text(
                handle.attrs.get("metadata_json", "{}")
            )
            payload["metadata"] = json.loads(metadata_json)
        for key in ("feedback_compatible", "raw_feedback_compatible"):
            if key in handle.attrs:
                payload[key] = bool(handle.attrs[key])
        for key in SYNTHETIC_TENSOR_KEYS:
            if key in handle:
                payload[key] = torch.from_numpy(handle[key][...])
    return payload


def save_synthetic_sample_pt(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    storage_dtype: StorageDType = "float16",
) -> None:
    """Write the legacy PyTorch-specific sample format."""
    if storage_dtype not in ("float16", "float32"):
        raise ValueError("storage_dtype must be 'float16' or 'float32'")
    output = dict(payload)
    torch_dtype = torch.float32 if storage_dtype == "float32" else torch.float16
    for key in SYNTHETIC_TENSOR_KEYS:
        if key in output and torch.is_tensor(output[key]):
            tensor = output[key].detach().cpu()
            dtype = torch.float32 if key in ("lf_counts", "lf_expected_counts") else torch_dtype
            output[key] = tensor.to(dtype) if tensor.is_floating_point() else tensor
    torch.save(output, Path(path))


def load_synthetic_sample_pt(
    path: str | Path,
    *,
    include_metadata: bool = True,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a sample mapping")
    if not include_metadata:
        payload.pop("metadata", None)
    return payload


def load_synthetic_sample(
    path: str | Path,
    *,
    include_metadata: bool = True,
) -> dict[str, Any]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in (".h5", ".hdf5"):
        return load_synthetic_sample_h5(
            source, include_metadata=include_metadata
        )
    if suffix == ".pt":
        return load_synthetic_sample_pt(
            source, include_metadata=include_metadata
        )
    raise ValueError(f"Unsupported synthetic sample extension {source.suffix!r}")
