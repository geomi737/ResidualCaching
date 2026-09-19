# Repeated GPU inference speed benchmark

FP32 weights, BF16 autocast, SDPA, batch 1; all three models resident during a seed; prefill and greedy decode timed separately; CUDA synchronization around wall-clock timings; rotated run order; no training.

Seeds [17]; contexts [131072]; 6 repetitions; 128 generated tokens per decode. Two warmup sequences per model/context. Longer context measures speed only, not quality. Throughput uses total tokens / total time; individual timings are preserved. Models share device residency, so this benchmark does not measure isolated VRAM.

| Seed | Context | Mode | Baseline tok/s | Sliding tok/s | vs baseline | Sliding + R tok/s | vs baseline |
|---|---:|---|---:|---:|---:|---:|---:|
| 17 | 131072 | prefill | 32457.3 | 51068.8 | +57.34% | 50727.4 | +56.29% |
| 17 | 131072 | decode | 33.4 | 43.8 | +31.44% | 43.6 | +30.67% |

These repeats reduce short-measurement noise but do not guarantee an idle device or fixed GPU clocks. Wall time includes Python dispatch and greedy argmax. The earlier training-adjacent one-shot measurements remain unchanged.
