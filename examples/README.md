# Examples

## Raw LF camera images

The four arrays in `raw_lf/` are unmodified `uint16` camera frames:

- `static_before.npy` and `static_after.npy`: `StaticQuench_20261017_222041`.
- `dynamic_before.npy` and `dynamic_after.npy`: `DynamicQuench_20261018_004137`.

These frames preserve the binned camera geometry. They must be registered to the `517 × 517` network grid before LF inference.

## Registered LF inputs

The three NPZ files in `registered_lf/` contain a network-ready array named `LF`:

| File | Background (ADU) |
|---|---:|
| `registered_bead_01.npz` | 97.5573501586914 |
| `registered_bead_02.npz` | 97.0403823852539 |
| `registered_bead_03.npz` | 97.22154998779297 |

The first sample is used by `usage/run_lf_network.py`.

## SLM patterns

The TIFF files in `patterns/` are experimental `1024 × 1024` gray-level patterns:

- `static_quench_pattern.tif`: first pulse from the selected static run.
- `dynamic_cycle01_pattern.tif`: first pulse from the selected dynamic run.
- `dynamic_cycle02_pattern.tif`: corrective pulse from the selected dynamic run.
