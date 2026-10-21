# Closed-loop 3D multiphoton targeting

This repository contains the two trained networks used in the work, small example datasets, example SLM patterns, and the final manuscript figures.

## Contents

```text
models/
  lf_network/          LF image to 3D reconstruction
  deepcgh/             3D target coordinates to SLM pattern
examples/
  raw_lf/              Representative raw LF camera images
  registered_lf/       Registered 517 × 517 LF inputs for the LF network
  patterns/            Example 1024 × 1024 SLM patterns
usage/
  run_lf_network.py
  run_deepcgh.py
figures/
  fig1/ ... fig5/      Final paper figures; files will be added by the authors
```

Figure- and video-rendering scripts are intentionally not included. The repository demonstrates the core pipeline:

```text
registered LF image -> LF network -> 3D volume
3D target coordinates -> DeepCGH network -> SLM pattern
```

## Installation

Python 3.11 and a CUDA-enabled PyTorch installation are recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch numpy scipy tifffile
```

Install the PyTorch build appropriate for the local CUDA version if GPU support is required.

## LF reconstruction network

The LF network accepts a registered `517 × 517` LF image and returns a `91 × 517 × 517` volume in `Z,Y,X` order. The included raw camera examples are provided for inspection; network inference requires the registered examples in `examples/registered_lf`.

```powershell
python usage/run_lf_network.py
```

The example writes `outputs/lf_volume_zyx.npy`. The network is loaded once and should remain resident for repeated inference.

## DeepCGH network

DeepCGH accepts target rows `[x_um, y_um, z_um, intensity]` and returns a `1024 × 1024` `uint8` SLM pattern.

```powershell
python usage/run_deepcgh.py
```

The example warms up the network, generates a three-target pattern, and writes `outputs/deepcgh_pattern.tif`.

## Example data

- `examples/raw_lf`: static and dynamic before/after camera frames.
- `examples/registered_lf`: three network-ready LF examples.
- `examples/patterns`: one static and two dynamic experimentally generated SLM patterns.

These examples demonstrate file formats and inference. They are not a complete archive of the experimental sessions.

## Models

- LF reconstruction: selected H1 → H2 → learned-fusion checkpoint, with metadata in `models/lf_network/bundle.json`.
- DeepCGH: validated 256 × 256 model using `models/deepcgh/DeepCGH_Models256/best.pt`, converted to a 1024 × 1024 SLM gray-level pattern.

## License

See [LICENSE](LICENSE).
