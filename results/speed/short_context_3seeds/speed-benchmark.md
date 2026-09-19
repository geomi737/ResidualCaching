# Repeated GPU inference speed benchmark

FP32 weights, BF16 autocast, SDPA, batch 1; all three models resident during a seed; prefill and greedy decode timed separately; CUDA synchronization around wall-clock timings; rotated run order; no training.

Seeds [17, 29, 43]; contexts [256, 1024]; 9 repetitions; 128 generated tokens per decode. Two warmup sequences per model/context. Longer context measures speed only, not quality. Throughput uses total tokens / total time; individual timings are preserved. Models share device residency, so this benchmark does not measure isolated VRAM.

| Seed | Context | Mode | Baseline tok/s | Sliding tok/s | vs baseline | Sliding + R tok/s | vs baseline |
|---|---:|---|---:|---:|---:|---:|---:|
| 17 | 256 | prefill | 41391.8 | 39699.2 | -4.09% | 38038.7 | -8.10% |
| 17 | 256 | decode | 171.3 | 168.7 | -1.50% | 169.3 | -1.18% |
| 17 | 1024 | prefill | 160863.7 | 158271.5 | -1.61% | 156840.5 | -2.50% |
| 17 | 1024 | decode | 168.0 | 165.7 | -1.40% | 165.2 | -1.67% |
| 29 | 256 | prefill | 40517.2 | 39401.5 | -2.75% | 39652.4 | -2.13% |
| 29 | 256 | decode | 169.2 | 165.9 | -1.99% | 166.1 | -1.83% |
| 29 | 1024 | prefill | 164973.0 | 152582.0 | -7.51% | 159849.8 | -3.11% |
| 29 | 1024 | decode | 168.6 | 164.3 | -2.54% | 165.4 | -1.91% |
| 43 | 256 | prefill | 40947.9 | 40171.9 | -1.90% | 39084.3 | -4.55% |
| 43 | 256 | decode | 168.0 | 167.2 | -0.47% | 168.1 | +0.10% |
| 43 | 1024 | prefill | 160628.3 | 155999.0 | -2.88% | 157837.5 | -1.74% |
| 43 | 1024 | decode | 168.0 | 166.9 | -0.69% | 165.5 | -1.54% |

These repeats reduce short-measurement noise but do not guarantee an idle device or fixed GPU clocks. Wall time includes Python dispatch and greedy argmax. The earlier training-adjacent one-shot measurements remain unchanged.
