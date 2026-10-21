"""Minimal LF-image to 3D-volume example."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LF_DIR = ROOT / "models" / "lf_network"
sys.path.insert(0, str(LF_DIR))

from infer_beads import LFBeadReconstructor  # noqa: E402


def main():
    input_path = ROOT / "examples" / "registered_lf" / "registered_bead_01.npz"
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    with np.load(input_path, allow_pickle=False) as payload:
        registered_counts_yx = payload["LF"]

    runner = LFBeadReconstructor(LF_DIR / "model.pt", device="auto", branch="final")
    volume_zyx = runner.reconstruct(
        registered_counts_yx,
        background=97.5573501586914,
    )

    output_path = output_dir / "lf_volume_zyx.npy"
    np.save(output_path, volume_zyx.astype(np.float32))
    print(f"Saved {volume_zyx.shape} volume to {output_path}")


if __name__ == "__main__":
    main()
