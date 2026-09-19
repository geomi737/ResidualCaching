# Repeated GPU inference speed benchmark

FP32 weights, BF16 autocast, SDPA, batch 1; all three models resident during a seed; prefill and greedy decode timed separately; CUDA synchronization around wall-clock timings; rotated run order; no training.

Seeds [17]; contexts [65536]; 6 repetitions; 128 generated tokens per decode. Two warmup sequences per model/context. Longer context measures speed only, not quality. Throughput uses total tokens / total time; individual timings are preserved. Models share device residency, so this benchmark does not measure isolated VRAM.

| Seed | Context | Mode | Baseline tok/s | Sliding tok/s | vs baseline | Sliding + R tok/s | vs baseline |
|---|---:|---|---:|---:|---:|---:|---:|
| 17 | 65536 | prefill | 59665.4 | 91691.2 | +53.68% | 91502.5 | +53.36% |
| 17 | 65536 | decode | 63.8 | 82.9 | +30.02% | 83.2 | +30.50% |

These repeats reduce short-measurement noise but do not guarantee an idle device or fixed GPU clocks. Wall time includes Python dispatch and greedy argmax. The earlier training-adjacent one-shot measurements remain unchanged.
