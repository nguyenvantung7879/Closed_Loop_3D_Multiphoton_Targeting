"""Minimal 3D-target to SLM-pattern example."""
from pathlib import Path
import sys

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
DEEPCGH_DIR = ROOT / "models" / "deepcgh"
sys.path.insert(0, str(DEEPCGH_DIR))

from DeepCGH256 import hologram, make_pattern_buffer, warmup_hologram  # noqa: E402


def main():
    # Columns: x_um, y_um, z_um, relative intensity.
    targets = np.array(
        [
            [-10.0, 0.0, -4.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [10.0, 0.0, 4.0, 1.0],
        ],
        dtype=np.float32,
    )

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    buffer = make_pattern_buffer()
    warmup_hologram(targets, n=5, out=buffer)
    pattern = hologram(targets, out=buffer)

    output_path = output_dir / "deepcgh_pattern.tif"
    tifffile.imwrite(output_path, pattern)
    print(f"Saved {pattern.shape} {pattern.dtype} pattern to {output_path}")


if __name__ == "__main__":
    main()
