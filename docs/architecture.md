# Architecture and open questions

## One-way causal compression

Let the original sequence have length `T`, hidden width `D`, and merge ratio `R`. The first `merge_layer` blocks use causal attention at full length. They produce contextual vectors `x[t]` which contain information only from positions `<=t`.

Non-overlapping completed windows are aggregated with learned weights shared across windows:

```text
a = softmax(w)
C[k] = Σᵢ a[i] x[kR+i] + Σᵢ x[kR+i]
```

The residual sum is a sum of **contextual hidden states**, not a separate sum of raw token embeddings. Unfinished windows are kept as separate singleton positions. With residuals enabled and `residual_tail=True`, each singleton is `2x`; with residuals disabled it is `x`. The shorter sequence length is:

```text
S = floor(T/R) + (T mod R)
```

Deep blocks use ordinary causal attention over this sequence. There is no unmerge, no full-resolution output block, and no extra positional embedding after merging. Original positional information is carried by the contextual vectors.

For `R=2`:

```text
Input positions:       0  1  2  3  4
Deep representations:  [C₀] [C₁] [singleton₄]
Last visible position:  1    3       4
Next-token target:      2    4       5
Shifted Y index:        1    3       4
```

## Why boundary loss is causal

A completed window ending at original position `e` aggregates only vectors from positions `<=e`. All earlier windows also end before `e`. Causal attention cannot read later windows or tails. Its prediction can therefore target the next token at `e+1` without future-token exposure.

When `Y[t]` is already shifted to the next token, its correct selected index is **`e`**, not `e+1`. The implementation collects `R−1,2R−1,…` for full windows and each original tail position, then computes mean cross-entropy over those predictions. `-1` targets are ignored, matching upstream nanoGPT's padding convention.

This provides approximately `T/R` signals instead of one final-prefix signal. It still provides fewer signals than the baseline's `T`. It does not claim that every original position is predicted in every compressed forward pass.

The historical hourglass prototype copies a complete window back onto its earlier positions. That construction lets an earlier prediction see a later token within the same window. The supported implementation avoids that operation entirely.

## Training mixed representations

The length sampler covers every remainder by choosing from `max(1,B−R+1)…B`. For `R=2` and even `B`, this alternates even and odd lengths. For larger `R`, it trains longer singleton tails as well.

Additionally, a fraction `unmerged_prob` of training batches use `merge_tokens=False`. The early blocks and merge interface remain present, but every position becomes a singleton representation; deep blocks receive an uncompressed sequence. Targets then cover all original positions. This is a batch-level mixture, not arbitrary selective merging inside the history.

This addresses training coverage: tensor compatibility alone does not imply that deep layers have learned to predict from every representation type. `merge_ratio=1` is a separate baseline path and does not apply the singleton residual scaling.

## Forward and checkpoint behavior

- `model(idx, targets)` returns all valid prediction logits and boundary loss.
- `model(idx)` returns only the last prediction, shaped `(B,1,V)`, and `None` loss.
- `model(idx, targets, merge_tokens=False)` returns singleton-only predictions in the merged architecture.
- `merge_layer` means the number of early blocks, and must satisfy `0<=merge_layer<n_layer` when merging is active. Invalid settings fail instead of silently moving the compression boundary.
- Training checkpoints persist all architecture flags. Resume and sampling preserve the old pass-through singleton convention for checkpoints without `residual_tail`.
- The training entry point supports merged models from scratch and resume. Upstream pretrained GPT-2 import remains baseline-only.
- DDP enables unused-parameter detection for merged models because singleton-only batches do not use the merge weights. GPU/DDP execution has not been validated in the CPU-only release environment.

## What residual caching does and does not mean

The implementation adds the sum once at compression. It does not retain each constituent vector, perform content-dependent routing, or retain a cache across generation steps. Calling it an implicit memory slot describes the aggregated residual path.

If `a[i]=1/R`, then:

```text
C = (R+1) * mean(x₀,…,xᵣ₋₁)
```

For pairs it is three times the mean. At initialization, the extra sum therefore introduces no new direction in the compressed vector. Learned nonuniform weights change the coefficients to `1+a[i]`, but the result remains a single linear aggregate, not lossless memory. Pre-layer normalization can remove much of a positive global scale difference from an attention/MLP branch; the residual stream can still evolve differently because the branch outputs and skip path have different relative scales.

The useful effect, if any, must be measured. Existing tests establish alignment and causality, not preservation of all fine details or better downstream accuracy. Future ablations should include a simple scaled-average control and multiple seeds.

## Performance expectations and remaining work

Only deeper blocks process `S` rather than `T` positions. Their projection and MLP work scales roughly with `S`, and the dense attention pair count scales with `S²`. This does not imply an overall twofold speedup: early blocks, vocabulary projections, runtime overhead, and hardware utilization also contribute. Flash attention changes how attention memory is materialized.

Generation currently recomputes the cropped prefix at every step. Implementing a persistent KV cache requires explicit handling of the transition from singleton tails to a completed window: temporary tail states must be removed or replaced consistently in every deep layer. Prefix cropping also resets original positional indices and window grouping. That is future work, not an implemented feature in this release.

The benchmark reports matched boundary loss and final-prefix loss over uniformly sampled prefix lengths. Neither is labeled as full-sequence perplexity. Training compute comparisons must also account for different supervised-prediction counts, singleton-batch proportions, warmup, and validation overhead.
