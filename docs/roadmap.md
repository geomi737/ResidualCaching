# Research direction and remaining work

This independent, fully AI-assisted research beta is directed by geomi737.
Gemini and ChatGPT/Codex assisted implementation and documentation. nanoGPT,
SmolLM2, Transformers, and WikiText retain their own upstream credits.

## Selected direction

Train **Sliding and Sliding + residual from scratch**, with dense causal overlapping windows
during training and disjoint compressed context at inference. No pretrained
weight adaptation or unmerge is part of the active training path. Baseline, Sliding, and
Sliding + R are the default runs. Sliding + R achieved the
best merged boundary accuracy in the [single-seed pilot](sliding-merging-experiment.md).
It does not beat baseline quality; Sliding without residual has better
final-prefix loss. Accuracy is one metric, not a complete quality measure.

## Priorities

- Repeat baseline, Sliding, and Sliding + R across several seeds; use fresh confirmation
  targets after the exploratory test-based direction choice.
- Measure context lengths 256/1024/4096/8192, recording live weights, KV bytes,
  temporary peaks, prefill and decode separately. Repeated, rotated timing runs
  should precede stronger speed claims.
- Evaluate distant-fact retrieval, relationships, numbers, and generated text,
  alongside matched next-token accuracy and cross-entropy.
- Study merge depth and the overlapping/disjoint-history mismatch. Compare both active sliding variants
  to isolate the contribution of the constituent residual sum.
- Extend R=2 caching to larger windows and padded/variable-length batches only
  after cache/full-prefix equivalence and causality checks.
- Keep equal-input, equal-supervision and equal-wall-time budgets distinct.

## Historical investigations

Conventional disjoint merging and Disjoint + R training are **legacy**.
Conventional merging receives roughly half as many predictions per input batch
at R=2; it is not demonstrated to slow learning by hundreds of times. It showed
30.79% lower training allocated peak in the scratch pilot. Those controls remain
in raw evidence but are not repeated by default.

Historical nanoGPT protocols, pretrained adaptation and unmerge prototypes are
separately labeled. Experimental conclusions may change; no global novelty,
lossless compression, or production-readiness claim is made.

## Latest evidence and next measurements

The [three-seed result](reports/training.md) supports a boundary-quality benefit
from R while retaining a final-prefix-loss tradeoff. [Repeated timings](reports/speed.md)
show large-context prefill and 64K/128K decode gains; [isolated 128K memory](reports/memory.md)
shows 25% smaller persistent KV but only 2.78% lower decode allocated peak.
Profile temporary singleton-cache allocations, confirm long-context timings across
seeds, and evaluate long-context quality before extending performance claims.
