# Repeated GPU inference speed benchmark

FP32 weights, BF16 autocast, SDPA, batch 1; all three models resident during a seed; prefill and greedy decode timed separately; CUDA synchronization around wall-clock timings; rotated run order; no training.

Seeds [17]; contexts [2048, 4096, 8192, 16384]; 6 repetitions; 128 generated tokens per decode. Two warmup sequences per model/context. Longer context measures speed only, not quality. Throughput uses total tokens / total time; individual timings are preserved. Models share device residency, so this benchmark does not measure isolated VRAM.

| Seed | Context | Mode | Baseline tok/s | Sliding tok/s | vs baseline | Sliding + R tok/s | vs baseline |
|---|---:|---|---:|---:|---:|---:|---:|
| 17 | 2048 | prefill | 296414.8 | 319466.8 | +7.78% | 317255.3 | +7.03% |
| 17 | 2048 | decode | 166.1 | 165.8 | -0.19% | 165.5 | -0.36% |
| 17 | 4096 | prefill | 302910.4 | 415836.1 | +37.28% | 411117.4 | +35.72% |
| 17 | 4096 | decode | 163.3 | 163.7 | +0.25% | 159.7 | -2.19% |
| 17 | 8192 | prefill | 240384.4 | 350575.3 | +45.84% | 345992.9 | +43.93% |
| 17 | 8192 | decode | 163.2 | 165.8 | +1.60% | 165.1 | +1.16% |
| 17 | 16384 | prefill | 158808.8 | 238071.2 | +49.91% | 238113.8 | +49.94% |
| 17 | 16384 | decode | 163.7 | 164.2 | +0.27% | 165.5 | +1.10% |

These repeats reduce short-measurement noise but do not guarantee an idle device or fixed GPU clocks. Wall time includes Python dispatch and greedy argmax. The earlier training-adjacent one-shot measurements remain unchanged.
