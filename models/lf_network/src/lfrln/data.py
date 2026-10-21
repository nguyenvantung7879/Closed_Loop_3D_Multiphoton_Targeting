from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .io import is_mat_v73, load_mat_variable


def load_lf_tensor(
    path: str | Path, variable_name: str = "LF", *, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Load a MATLAB LF image as ``[1, H, W]``."""
    array = np.asarray(load_mat_variable(path, variable_name))
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D LF image, got {array.shape}")

    # h5py exposes MATLAB v7.3 arrays with reversed dimension order.
    if is_mat_v73(path):
        array = array.T
    return torch.as_tensor(np.ascontiguousarray(array), dtype=dtype).unsqueeze(0)


def load_volume_tensor(
    path: str | Path, variable_name: str = "GT", *, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Load a MATLAB ``[H, W, D]`` volume as PyTorch ``[D, H, W]``."""
    array = np.asarray(load_mat_variable(path, variable_name))
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got {array.shape}")

    if is_mat_v73(path):
        # h5py order is [D, W, H].
        array = array.transpose(0, 2, 1)
    else:
        # scipy order is the original MATLAB [H, W, D].
        array = array.transpose(2, 0, 1)
    return torch.as_tensor(np.ascontiguousarray(array), dtype=dtype)
