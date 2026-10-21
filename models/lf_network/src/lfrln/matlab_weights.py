from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import scipy.io
import torch
from torch import nn

from .h1_model import H1RLN

if TYPE_CHECKING:
    from .h1_lite_model import H1Lite


@dataclass(frozen=True)
class MatlabLoadReport:
    loaded: tuple[str, ...]
    missing: tuple[str, ...]
    unused: tuple[str, ...]


def _h1_layer_map(model: H1RLN) -> dict[str, str]:
    mapping = {
        "initialLayer": "initial",
        "H1_convL1": "forward_project",
        "H1_convL1t": "observed_backproject",
        "H1_convL2_1_extend": "fp_extend",
        "H1_convL2_2": "fp_refine.conv",
        "H1_convL2_2_norm": "fp_refine.norm",
        "H1_convL2_V": "fp_to_views.conv",
        "H1_convL2_V_norm": "fp_to_views.norm",
        "H1_dv_norm": "ratio_norm",
        "H1_va_squeeze": "view_attention.squeeze",
        "H1_va_excite": "view_attention.excite",
        "H1_convL2t_1_extend": "bp_extend",
        "H1_convL2t_2": "bp_refine.conv",
        "H1_convL2t_2_norm": "bp_refine.norm",
        "H1_convL2t_V": "bp_to_depth_views.conv",
        "H1_convL2t_V_norm": "bp_to_depth_views.norm",
        "H1_bp_CM2_conv": "global_context.mask",
        "H1_bp_tran2_conv1": "global_context.transform1",
        "H1_bp_tran2_LN": "global_context.norm",
        "H1_bp_tran2_conv2": "global_context.transform2",
        "H1_convL3t": "volume_backproject",
        "H1_bp_upConv": "up_refine",
        "H1_bp_upConv_norm": "up_norm",
        "H1_mu_norm": "output_norm",
    }

    for layer_index in range(model.config.fp_layers):
        block_index, conv_index = divmod(layer_index, 2)
        mapping[f"H1_convL2_1_{layer_index + 1}"] = (
            f"fp_blocks.{block_index}.conv{conv_index + 1}"
        )

    for layer_index in range(model.config.bp_layers):
        block_index, conv_index = divmod(layer_index, 2)
        mapping[f"H1_convL2t_1_{layer_index + 1}"] = (
            f"bp_blocks.{block_index}.conv{conv_index + 1}"
        )

    return mapping


def _h1_lite_shared_layer_map(model: H1Lite) -> dict[str, str]:
    """Layers whose shapes and meanings are unchanged in H1Lite."""
    mapping = {
        "initialLayer": "initial",
        "H1_convL2_1_extend": "fp_extend",
        "H1_convL2_2": "fp_refine.conv",
        "H1_convL2_2_norm": "fp_refine.norm",
        "H1_convL2_V": "fp_to_views.conv",
        "H1_convL2_V_norm": "fp_to_views.norm",
        "H1_dv_norm": "ratio_norm",
        "H1_va_squeeze": "view_attention.squeeze",
        "H1_va_excite": "view_attention.excite",
    }
    if isinstance(getattr(model, "output_norm", None), nn.InstanceNorm2d):
        mapping["H1_mu_norm"] = "output_norm"
    for layer_index in range(model.config.fp_layers):
        block_index, conv_index = divmod(layer_index, 2)
        mapping[f"H1_convL2_1_{layer_index + 1}"] = (
            f"fp_blocks.{block_index}.conv{conv_index + 1}"
        )
    return mapping


def _read_exported_parameters(path: Path) -> dict[str, np.ndarray]:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {key: np.asarray(archive[key]) for key in archive.files}

    try:
        mat = scipy.io.loadmat(path, squeeze_me=False, struct_as_record=False)
    except NotImplementedError as error:
        raise ValueError(
            f"{path} is MATLAB v7.3. Run scripts/export_h1_checkpoint.m "
            "to produce the numeric v7 interchange file."
        ) from error

    if "parameters" not in mat:
        return {
            key: np.asarray(value)
            for key, value in mat.items()
            if not key.startswith("__")
        }

    parameter_struct = mat["parameters"]
    if parameter_struct.size != 1:
        raise ValueError("Expected a scalar MATLAB 'parameters' struct")
    parameter_struct = parameter_struct.flat[0]
    field_names = getattr(parameter_struct, "_fieldnames", None)
    if field_names is None:
        raise ValueError("MATLAB 'parameters' is not a struct")
    return {
        field_name: np.asarray(getattr(parameter_struct, field_name))
        for field_name in field_names
    }


def _matlab_weight_to_torch(value: np.ndarray, module: nn.Module) -> torch.Tensor:
    array = np.asarray(value)
    while array.ndim < 4:
        array = np.expand_dims(array, axis=-1)

    if array.ndim != 4:
        raise ValueError(f"Expected a 4D MATLAB weight array, got {array.shape}")
    if not isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        raise TypeError(f"Weights are unsupported for {type(module).__name__}")

    # MATLAB: [height, width, input, output] for Conv2D and
    # [height, width, output, input] for transposed Conv2D.
    # PyTorch stores both corresponding layouts as [dim4, dim3, H, W].
    return torch.from_numpy(np.ascontiguousarray(array.transpose(3, 2, 0, 1)))


def _copy_parameter(
    module: nn.Module, matlab_parameter: str, value: np.ndarray
) -> None:
    if matlab_parameter == "Weights":
        target_name = "weight"
        tensor = _matlab_weight_to_torch(value, module)
    elif matlab_parameter == "Bias":
        target_name = "bias"
        tensor = torch.from_numpy(np.ascontiguousarray(np.asarray(value).reshape(-1)))
    elif matlab_parameter == "Scale":
        target_name = "weight"
        tensor = torch.from_numpy(np.ascontiguousarray(np.asarray(value).reshape(-1)))
    elif matlab_parameter == "Offset":
        target_name = "bias"
        tensor = torch.from_numpy(np.ascontiguousarray(np.asarray(value).reshape(-1)))
    else:
        raise ValueError(f"Unsupported MATLAB parameter: {matlab_parameter}")

    target = getattr(module, target_name, None)
    if target is None:
        raise ValueError(
            f"{type(module).__name__} does not have parameter '{target_name}'"
        )
    if tuple(tensor.shape) != tuple(target.shape):
        raise ValueError(
            f"Shape mismatch for {type(module).__name__}.{target_name}: "
            f"MATLAB converts to {tuple(tensor.shape)}, "
            f"PyTorch expects {tuple(target.shape)}"
        )

    with torch.no_grad():
        target.copy_(tensor.to(device=target.device, dtype=target.dtype))


def _load_mapped_parameters(
    model: nn.Module,
    path: str | Path,
    layer_map: dict[str, str],
    *,
    reject_missing: bool,
    reject_unused: bool,
) -> MatlabLoadReport:
    parameters = _read_exported_parameters(Path(path))
    loaded: list[str] = []
    expected: set[str] = set()
    used: set[str] = set()

    for matlab_layer, module_path in layer_map.items():
        module = model.get_submodule(module_path)
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            matlab_parameters = ("Weights", "Bias")
        elif isinstance(module, nn.InstanceNorm2d):
            matlab_parameters = ("Scale", "Offset")
        else:
            raise TypeError(f"Unsupported mapped module: {type(module).__name__}")

        for matlab_parameter in matlab_parameters:
            key = f"{matlab_layer}__{matlab_parameter}"
            expected.add(key)
            if key not in parameters:
                continue
            _copy_parameter(module, matlab_parameter, parameters[key])
            loaded.append(key)
            used.add(key)

    missing = sorted(expected - used)
    unused = sorted(set(parameters) - used)
    if (reject_missing and missing) or (reject_unused and unused):
        raise ValueError(
            "MATLAB export did not match the requested mapping. "
            f"Missing={missing}; unused={unused}"
        )
    return MatlabLoadReport(
        loaded=tuple(sorted(loaded)),
        missing=tuple(missing),
        unused=tuple(unused),
    )


def load_h1_matlab_export(
    model: H1RLN,
    path: str | Path,
    *,
    strict: bool = True,
) -> MatlabLoadReport:
    """
    Load numeric parameters exported by ``export_h1_checkpoint.m``.

    The original MATLAB ``dlnetwork`` object is not a stable cross-language
    interchange format. The exporter flattens its learnables into numeric
    arrays named ``Layer__Parameter``.
    """
    layer_map = _h1_layer_map(model)
    report = _load_mapped_parameters(
        model,
        path,
        layer_map,
        reject_missing=strict,
        reject_unused=strict,
    )

    optical_keys = {
        "H1_convL1__Weights",
        "H1_convL1__Bias",
        "H1_convL1t__Weights",
        "H1_convL1t__Bias",
        "H1_convL3t__Weights",
        "H1_convL3t__Bias",
    }
    if optical_keys.issubset(report.loaded):
        model.mark_optics_loaded()
    return report


def load_h1_lite_shared_from_matlab(
    model: H1Lite,
    path: str | Path,
    *,
    strict_shared: bool = True,
) -> MatlabLoadReport:
    """
    Transfer the unchanged H1 forward/ratio layers into an H1Lite student.

    Backprojection, latent attention, and separable output-refinement weights
    are intentionally not transferred because their shapes/operations changed.
    Unused MATLAB tensors are therefore expected and are reported, not rejected.
    """
    return _load_mapped_parameters(
        model,
        path,
        _h1_lite_shared_layer_map(model),
        reject_missing=strict_shared,
        reject_unused=False,
    )
