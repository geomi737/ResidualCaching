# Architecture and open questions

## Active sliding-training protocol

The current scratch experiment uses dense causal overlapping pairs during
training and disjoint compressed windows during inference. Sliding + R is the
selected direction; baseline and Sliding + R are the default new comparisons.
Each dense training position uses only the previous/current contextual states
and predicts its next token. Training length is preserved; inference compresses
completed windows. See the [full five-variant protocol](sliding-merging-experiment.md).

The sections below describe the shared **disjoint** compression path and the
earlier nanoGPT implementation. Their boundary-only training loss does not
apply to dense sliding training. Standard Transformer residual connections,
additional constituent-sum aggregation, and persistent KV caches are separate
mechanisms. The current scratch adapter keeps only completed-window K/V in the
persistent deep cache; singleton computations use temporary cache records.


## One-way causal compression

Let the original sequence have length `T`, hidden width `D`, and merge ratio `R`. The first `merge_layer` blocks use causal attention at full length. They produce contextual vectors `x[t]` which contain information only from positions `<=t`.

Non-overlapping completed windows are aggregated with learned weights shared across windows:

```text
a = softmax(w)
C[k] = Σᵢ a[i] x[kR+i] + Σᵢ x[kR+i]
```

The residual sum is a sum of **contextual hidden states**, not a separate sum of raw token embeddings. Unfinished windows are kept as separate singleton positions. Each incomplete tail position is passed through unchanged, with or without residual aggregation. Compression is always applied to complete windows from left to right. The shorter sequence length is:

```text
S = floor(T/R) + (T mod R)
```

Deep blocks use ordinary causal attention over this sequence. There is no unmerge, no full-resolution output block, and no full-length reconstruction. nanoGPT carries its learned original positions in the contextual vectors. The SmolLM2 adapter instead applies RoPE using original window-end positions in deep blocks, preserving the original position of an untouched tail.

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

The former 10% singleton-only bypass has been removed. Every merged forward compresses all complete windows, even on odd-length inputs. The new paired SmolLM2 runner uses the same input sequences and batch sizes as the baseline. Equal input batches remain valid for ordinary backpropagation; only the output positions and target selection change.

This addresses training coverage: tensor compatibility alone does not imply that deep layers have learned to predict from every representation type. `merge_ratio=1` is a separate baseline path and does not apply the singleton residual scaling.

## Forward and checkpoint behavior

- `model(idx, targets)` returns all valid prediction logits and boundary loss.
- `model(idx)` returns only the last prediction, shaped `(B,1,V)`, and `None` loss.
- `merge_tokens=False`, `residual_tail=True`, and nonzero `unmerged_prob` are rejected in the current nanoGPT entry points. The SmolLM2 adapter has no such bypass or scaling options.
- `merge_layer` means the number of early blocks, and must satisfy `0<=merge_layer<n_layer` when merging is active. Invalid settings fail instead of silently moving the compression boundary.
- Training checkpoints persist architecture flags. Old doubled-tail checkpoints must be used with historical revision `3edb3c5`.
- The training entry point supports merged models from scratch and resume. Upstream pretrained GPT-2 import remains baseline-only.
- DDP enables unused-parameter detection for merged models because sequences shorter than a complete window do not use the merge weights. Single-GPU execution and float32/BF16 correctness have been validated on RTX 4060; compilation and multi-GPU DDP have not.

## What residual caching does and does not mean

The implementation adds the sum once at compression. It does not retain each constituent vector, perform content-dependent routing, or itself retain a cache across generation steps. Calling it an implicit memory slot describes the aggregated residual path. The SmolLM2 adapter additionally has a conventional persistent KV cache, which is a separate mechanism.

If `a[i]=1/R`, then:

```text
C = (R+1) * mean(x₀,…,xᵣ₋₁)
```

For pairs it is three times the mean. At initialization, the extra sum therefore introduces no new direction in the compressed vector. Learned nonuniform weights change the coefficients to `1+a[i]`, but the result remains a single linear aggregate, not lossless memory. Pre-layer normalization can remove much of a positive global scale difference from an attention/MLP branch; the residual stream can still evolve differently because the branch outputs and skip path have different relative scales.

For any learned weights, define `q[i]=(1+a[i])/(R+1)`: the operator is exactly `(R+1)*sum(q[i]*x[i])`, with normalized positive weights. [Explicit proofs and collision examples](proofs.md) establish its limits. [Three-seed GPU experiments](experiments.md) find better held-out quality than plain merging in the tested setup, while final-prefix quality remains below baseline. Those runs do not establish lossless preservation or isolate scale from weighting/optimization; a mechanism control and broader datasets are still needed.

## Performance expectations and remaining work

Only deeper blocks process `S` rather than `T` positions. Their projection and MLP work scales roughly with `S`, and the dense attention pair count scales with `S²`. This does not imply an overall twofold speedup: early blocks, vocabulary projections, runtime overhead, and hardware utilization also contribute. Flash attention changes how attention memory is materialized.

nanoGPT generation recomputes the cropped prefix at every step. The SmolLM2 adapter caches original-resolution early-layer keys/values and compressed deep-layer keys/values. An incomplete tail is provisional: when its partner arrives, every deep layer removes its tail cache entry and appends the completed merged window. Its pre-merge hidden state is retained until then. On context cropping, the adapter rebuilds the cache, resetting relative positions and window grouping consistently. Tests compare this path against full-prefix recomputation. Cached streaming currently supports R=2 and unpadded equal-length batches only.

The benchmark reports matched held-out boundary loss/accuracy and final-prefix loss/accuracy with balanced near-block-size lengths covering all remainders. Neither is labeled as full-sequence perplexity. It measures training and generation separately, checks paired initial weights/data plans, and records allocated/reserved memory. Training compute comparisons must also account for different supervised-prediction counts, singleton-batch proportions, warmup, and validation overhead.
