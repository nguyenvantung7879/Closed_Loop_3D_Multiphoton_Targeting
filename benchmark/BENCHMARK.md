# Image-to-pattern benchmark

One recorded raw LF frame with four targets, repeated for 11 trials on an NVIDIA RTX 3090 (networks warmed up before timing; the first trial is discarded, so **n = 10**).

| Stage | Time, mean ± SD (ms) | Paper |
|---|---:|---:|
| LF acquisition + transfer (one 60 fps frame) | 16.67 | < 17 ms |
| 3D reconstruction | 18.29 ± 0.68 | < 20 ms |
| Target detection + mapping | 7.53 ± 1.06 | < 10 ms |
| DeepCGH pattern generation + GDD | 6.93 ± 0.89 | < 10 ms |
| SLM pattern transfer (allowance, not measured) | 3.00 | < 3 ms |
| **Imaging-to-pattern** | **52.42 ± 1.72** | **< 60 ms** |

Imaging-to-pattern = camera period (16.67 ms) + measured computation (32.75 ± 1.72 ms) + SLM allowance (3 ms); every trial is below 60 ms (50.2–55.3 ms). The SLM was not driven in this benchmark.

## Test environment

- GPU: NVIDIA GeForce RTX 3090
- Operating system: Windows 11 (build 10.0.26200)
- Python: 3.11.9
- PyTorch: 2.12.0+cu126
- CUDA runtime: 12.6, cuDNN 9.10.02
- LF inference: AMP, CUDA Graph, final H1/H2/fusion output, GPU-resident 3D volume
- 3D detection and DeepCGH + GDD: GPU-resident
- DeepCGH: 256 × 256 inference followed by 1024 × 1024 gray-level pattern generation
- CPU: Intel Core i7-6700 @ 3.40 GHz
- System RAM: 32 GB
