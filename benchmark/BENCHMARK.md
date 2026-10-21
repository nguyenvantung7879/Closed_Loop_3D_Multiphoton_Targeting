# Image-to-pattern benchmark

The benchmark used one recorded raw LF frame with four detected targets. Both neural networks were warmed up before timing. Eleven trials were recorded; the first measured cold-start trial was excluded, leaving **n = 10** trials in the summary.

| Pipeline stage | Time, mean ± SD (ms) |
|---|---:|
| LF reconstruction | 18.37 ± 0.66 |
| 3D detection and target mapping | 10.80 ± 2.36 |
| DeepCGH pattern generation with GDD | 11.19 ± 1.17 |
| Total GPU/CPU computation | 40.36 ± 3.32 |
| Estimated image-to-pattern latency | 60.03 ± 3.32 |
| Estimated pipeline rate | 16.70 ± 0.89 fps |

The estimated image-to-pattern latency is the measured computation time plus one `60 fps` camera period (`16.67 ms`) and a `3.00 ms` SLM upload allowance. SLM transfer was not physically performed during this benchmark.

## Test environment

- GPU: NVIDIA GeForce RTX 3090
- Operating system: Windows
- Python: 3.11.15
- PyTorch: 2.12.0
- CUDA runtime: 12.6
- LF inference: AMP, CUDA Graph, final H1/H2/fusion output, GPU-resident 3D volume
- DeepCGH: 256 × 256 inference followed by 1024 × 1024 gray-level pattern generation

CPU model and installed system RAM were not recorded with this benchmark. They can be added later if needed; the timed neural-network stages were GPU accelerated.
