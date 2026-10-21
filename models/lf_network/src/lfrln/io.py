from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.io


def is_mat_v73(path: str | Path) -> bool:
    """
    Check whether a .mat file is MATLAB v7.3/HDF5 format.
    """
    path = Path(path)
    try:
        with h5py.File(path, "r"):
            return True
    except OSError:
        return False


def list_mat_variables(path: str | Path) -> list[dict[str, Any]]:
    """
    List variable names, shapes, and dtypes inside a MATLAB .mat file.

    Works for both:
    - MATLAB v7.3 files through h5py
    - older MATLAB files through scipy.io.loadmat

    Returns
    -------
    info:
        A list of dictionaries with fields:
        name, shape, dtype, backend
    """
    path = Path(path)

    if is_mat_v73(path):
        info = []
        with h5py.File(path, "r") as f:
            def visitor(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset):
                    info.append(
                        {
                            "name": name,
                            "shape": tuple(obj.shape),
                            "dtype": str(obj.dtype),
                            "backend": "h5py/v7.3",
                        }
                    )

            f.visititems(visitor)
        return info

    mat = scipy.io.loadmat(path, squeeze_me=False, struct_as_record=False)
    info = []

    for key, value in mat.items():
        if key.startswith("__"):
            continue

        if isinstance(value, np.ndarray):
            shape = value.shape
            dtype = str(value.dtype)
        else:
            shape = None
            dtype = type(value).__name__

        info.append(
            {
                "name": key,
                "shape": shape,
                "dtype": dtype,
                "backend": "scipy",
            }
        )

    return info


def load_mat_variable(path: str | Path, variable_name: str) -> np.ndarray:
    """
    Load one variable from a MATLAB .mat file.

    Important
    ---------
    MATLAB and NumPy both display arrays differently from PyTorch.
    This function only loads the raw NumPy array. We will handle
    dimension permutation separately after inspecting the shapes.
    """
    path = Path(path)

    if is_mat_v73(path):
        with h5py.File(path, "r") as f:
            if variable_name not in f:
                raise KeyError(
                    f"Variable '{variable_name}' not found in {path}. "
                    f"Available top-level keys: {list(f.keys())}"
                )
            arr = np.array(f[variable_name])
        return arr

    mat = scipy.io.loadmat(path, squeeze_me=False, struct_as_record=False)

    if variable_name not in mat:
        available = [k for k in mat.keys() if not k.startswith("__")]
        raise KeyError(
            f"Variable '{variable_name}' not found in {path}. "
            f"Available variables: {available}"
        )

    return mat[variable_name]


def print_mat_summary(path: str | Path) -> None:
    """
    Print a readable summary of variables inside a .mat file.
    """
    path = Path(path)
    print(f"\nFile: {path}")
    print(f"MATLAB v7.3 / HDF5: {is_mat_v73(path)}")

    info = list_mat_variables(path)

    if not info:
        print("No variables found.")
        return

    name_width = max(len(item["name"]) for item in info)
    shape_width = max(len(str(item["shape"])) for item in info)

    print("-" * 80)
    print(f"{'name'.ljust(name_width)}  {'shape'.ljust(shape_width)}  dtype  backend")
    print("-" * 80)

    for item in info:
        print(
            f"{item['name'].ljust(name_width)}  "
            f"{str(item['shape']).ljust(shape_width)}  "
            f"{item['dtype']}  "
            f"{item['backend']}"
        )