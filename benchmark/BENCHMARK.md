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

Test setup: Windows, Python 3.11, PyTorch 2.12 (CUDA 12.6), LF network with AMP + CUDA Graph, DeepCGH 256 × 256 followed by 1024 × 1024 pattern generation with GDD.
