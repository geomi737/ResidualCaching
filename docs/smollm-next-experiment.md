# Historical SmolLM2 feasibility and adaptation protocol

**Historical report.** This page records the earlier pretrained feasibility
measurements and adaptation proposal. The active experiment now trains **from
scratch**; see [the sliding GPU pilot](sliding-merging-experiment.md).
Legacy training scripts are retained under `experiments/smollm/legacy/`.
Pretrained and scratch results must not be mixed.

## Why pretrained weights

[SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) provides a
134,515,008-parameter starting point. Its model card reports two trillion
pretraining tokens. Training a similarly sized random model briefly on a desktop
would not reproduce those capabilities. This experiment should adapt pretrained
weights and measure how much quality compression retains. Compression itself is
not evidence of improved intelligence. Credit for the pretrained model belongs
to Hugging Face and the SmolLM2 authors; its license is Apache-2.0.

## Measured local feasibility

On 2026-09-17, the unmodified pretrained model completed five full-parameter
AdamW updates per configuration on the local RTX 4060. Parameters and optimizer
states were float32, computation used BF16 autocast and SDPA, batch size was one,
and gradient checkpointing was disabled. Each case ran in a fresh process.

| Context | Peak allocated MiB | Peak reserved MiB | Input tokens/s |
| --- | ---: | ---: | ---: |
| 512 | 2618.50 | 2804 | 4680 |
| 1024 | 3449.59 | 3576 | 8526 |

These are short synthetic-input feasibility probes, not quality measurements or
stable performance benchmarks. Throughput excludes the first warmup update.
Memory peaks include the warmup and optimizer-state initialization, but exclude
other applications and CUDA allocations outside PyTorch's allocator. No adapted
weights were saved. Random-input losses must not be interpreted as validation
losses. Raw results are in `results/smollm135m_feasibility_*.json`.

Reproduce from the repository root (reuse a working CUDA PyTorch installation):

```bash
python -m venv --system-site-packages .venv-smollm
.venv-smollm/bin/python -m pip install -r requirements-smollm.txt
hf download HuggingFaceTB/SmolLM2-135M --revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2 --local-dir models/SmolLM2-135M
.venv-smollm/bin/python experiments/smollm/probe_smollm.py --length 512 --output results/smollm135m_feasibility_512.json
.venv-smollm/bin/python experiments/smollm/probe_smollm.py --length 1024 --output results/smollm135m_feasibility_1024.json
```

## Historical proposed adaptation and evaluation

1. Load identical pretrained weights for the conventional model, weighted merging
   without residual aggregation, and weighted merging plus the constituent sum.
   Keep the original formula, singleton convention, and boundary-only supervision
   explicit. Do not unmerge. A late merge boundary should be the initial trial;
   earlier boundaries can then test the quality/compute tradeoff.
2. Implement Llama-specific integration and test it: RMSNorm, grouped-query
   attention, and rotary positions differ from nanoGPT. Use original window-end
   positions for compressed tokens and recompute rotary embeddings accordingly.
   Verify baseline equivalence, future-token independence, gradients, and tails.
3. Adapt on a bounded general-text corpus with document-level training/validation/
   test separation and recorded dataset revisions. Start with microbatch one,
   context 512, gradient accumulation, and a small learning-rate pilot. Select
   settings on validation only. Prior pretraining contamination cannot generally
   be excluded: call these data held out from adaptation, not necessarily unseen
   throughout the model's lifetime.
4. Compare identical validation/test targets with prefix next-token loss and
   accuracy, plus task accuracy such as multiple-choice reasoning. Score every
   answer token from its prefix rather than silently omitting tokens inside
   merge windows. Record both equal input-token and equal supervised-prediction
   budgets, training wall time, and at least three seeds for the final comparison.
5. Measure training and inference memory separately. Report prefill separately
   from autoregressive decode. Include the conventional model's native KV-cached
   generation as a practical baseline; a no-cache comparison alone cannot support
   a deployment speedup claim. Persistent compressed KV caching needs separate
   correctness work for incomplete windows and position/cropping semantics.

The current measurements support starting with 135M locally. A 360M trial is a
later option, subject to its own memory probe. Neither a quality improvement nor
an inference speedup has been demonstrated for compressed SmolLM2.

## Cloud resources

There is no connected Colab runtime in this workspace. Google's
[Colab FAQ](https://research.google.com/colaboratory/faq.html) states that free
resources and GPU availability are not guaranteed. Colab can be an optional
user-launched environment, but is not required for the measured 135M setup.
