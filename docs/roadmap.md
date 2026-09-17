# Research beta and continuing experiments

This project is a fully AI-assisted / vibe-coded research beta directed by geomi737. Gemini and ChatGPT/Codex helped develop the project-specific implementation and documentation. Andrej Karpathy's nanoGPT remains the credited upstream foundation. The author is releasing early to record provenance and invite reproducible scrutiny. Development and experimentation will continue.

## What is implemented

The supported model uses the requested learned weighted merge plus constituent sum, one-way compression without unmerging, and causal window-boundary loss. Completed windows are compressed once before the deep blocks. The residual sum is added once at that boundary; it is not a persistent decoding cache.

Singleton windows use `x+x` with residuals enabled. The training recipe also includes 10% singleton-only batches. These are explicit implementation choices, not additional measured properties of the original formula. `residual_tail=False` retains the older pass-through tail convention; `unmerged_prob=0.0` disables singleton-only batches while retaining mixed contexts with tails.

## Continuing work

- Implement and verify persistent KV caching, including replacement of singleton tails when windows become complete and a defined cropping/position policy.
- Reduce singleton-only batch memory through smaller microbatches and gradient accumulation, then rerun paired quality/throughput comparisons.
- Compare equal supervised-prediction and wall-clock budgets in addition to equal input-token budgets.
- Add controls for scaled averaging, constrained weights, and singleton scaling to identify the residual mechanism.
- Test more datasets, seeds, model widths, context lengths, and GPUs, with held-out task accuracy and confidence estimates.
- Profile generation kernels before claiming an inference speedup; evaluate compilation and distributed training separately.

## Current release limits

The measured residual configuration improves plain merging on the tested dataset, while classical nanoGPT retains better final-prefix quality. Training input throughput improves, but supervised predictions per second decrease. Pure compressed batches reduce allocated peak memory; mixed training retains the full-size singleton peak. Generation acceleration is not demonstrated. See the [complete experiment report](experiments.md).

The code, API, and findings may change. Contributions should preserve attribution and include reproducible commands and raw measurements for empirical claims. No dates or outcomes are promised for future work.
