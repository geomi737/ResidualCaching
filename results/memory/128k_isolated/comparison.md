# Isolated 128K inference: speed and GPU memory

Context 131072; seed 17; 6 repeats; 128 generated tokens. Each variant runs in a fresh process. FP32 weights, BF16 autocast, SDPA, batch 1. Two warmup sequences; allocator stays warm.

| Variant | Parameters MiB | Weights-only allocated MiB | Prefill s | Prefill peak allocated / reserved MiB | Decode tok/s | Decode peak allocated / reserved MiB | Final KV MiB |
|---|---:|---:|---:|---|---:|---|---:|
| baseline | 84.39 | 84.40 | 3.9813 | 2702.53 / 3426.00 | 33.85 | 1727.24 / 3426.00 | 1537.50 |
| sliding | 84.39 | 84.40 | 2.4634 | 1934.53 / 2854.00 | 46.22 | 1679.19 / 2854.00 | 1153.12 |
| sliding_residual | 84.39 | 84.40 | 2.4174 | 1934.53 / 2854.00 | 46.81 | 1679.19 / 2854.00 | 1153.12 |

## Differences from baseline

| Variant | Prefill speed | Decode speed | Prefill allocated peak | Decode allocated peak | Final KV |
|---|---:|---:|---:|---:|---:|
| sliding | +61.62% | +36.54% | -28.42% | -2.78% | -25.00% |
| sliding_residual | +64.70% | +38.29% | -28.42% | -2.78% | -25.00% |

Memory peaks include weights and live tensors, not only attention or KV. Reserved memory includes allocator caching and is not added to allocated. CUDA context, driver, and other processes are excluded. Peaks are maxima over repeats; speed is total tokens / total time. Memory and speed are measured together, without mixing shared-residency timings from earlier tests. Long-context quality is not evaluated; one seed and sequential variant order limit generalization.
