# Algebraic properties and runtime evidence

This document separates properties that follow from the implemented equations from empirical hypotheses. Numerical witnesses are reproducible with:

```bash
python experiments/nanogpt/verify_properties.py --output=results/legacy/nanogpt/property_evidence.json
python experiments/nanogpt/verify_properties.py --output=results/legacy/nanogpt/property_evidence.json
```

The [recorded evidence](../results/legacy/nanogpt/property_evidence.json) uses float64 for the algebraic examples and a small CPU model for runtime tracing. The regression suite also checks GPU causality and backward in float32 and BF16.

## 1. The residual aggregate cannot recover arbitrary individual vectors

For a window of size `R`, let `a=softmax(w)` and `b_i=1+a_i`. The implemented operator is:

```text
C(x_0,...,x_(R-1)) = sum_i b_i x_i.
```

It maps `R*D` scalar coordinates to `D` coordinates. With the weights fixed, its matrix is the block row `[b_0 I_D ... b_(R-1) I_D]`. Its rank is `D` and its nullspace has dimension `(R-1)*D` for `R>1`.

An explicit pair collision is stronger than counting dimensions. For any nonzero vector `v`, set:

```text
x'_0 = x_0 + b_1 v
x'_1 = x_1 - b_0 v.
```

Then `C(x'_0,x'_1)=C(x_0,x_1)` although the inputs differ. The recorded collision has input-difference norm 2.427 and output difference below `1e-16`. No decoder receiving only this aggregate can distinguish those arbitrary inputs.

This proves that the **aggregation operator** is not lossless on arbitrary hidden states. It does not prove that every constructed collision is reachable from real text, that earlier blocks cannot encode useful information redundantly, or that compression must harm every language-model task. Those questions require downstream evaluation.

The residual sum also does not guarantee nonzero magnitude: with uniform weights and `x_1=-x_0`, the compressed vector is zero while both input vectors are nonzero. “Preserves magnitude” therefore describes an intended amplification effect, not a universal norm guarantee.

## 2. Uniform weights produce a scaled mean; nonuniform weights produce a scaled weighted average

At initialization `a_i=1/R`:

```text
C = (1+1/R) sum_i x_i = (R+1) mean_i x_i.
```

For pairs this is exactly three times the mean. The float64 witness has zero error.

More generally, define `q_i=(1+a_i)/(R+1)`. The `q_i` are positive and sum to one, so:

```text
C = (R+1) sum_i q_i x_i.
```

This holds for learned nonuniform weights too; the recorded error is below `1e-15`. For pairs, each `q_i` lies between `1/3` and `2/3`. The residual sum changes overall scale and constrains the relative constituent weights. It does not introduce an additional independent output channel.

A quality advantage over the no-residual model would establish an advantage for the implemented residual configuration under the experiment's conditions. The requested three-way comparison cannot separately attribute that advantage to scale, constrained weighting, singleton scaling, or optimization. A scaled-average control is needed for a causal explanation of the mechanism.

## 3. nanoGPT generation recomputes the prefix

This section describes nanoGPT only. The SmolLM2 adapter has a separately tested incremental KV cache; see [the new experiment](smollm-next-experiment.md).

`GPT.generate` invokes `forward(idx_cond)` for each newly generated token. `forward` recalculates embeddings and all Transformer blocks; attention accepts only `x` and has no retained key/value argument or persistent decoding state.

The runtime witness starts with 5 tokens, generates 6 more, and uses a block size of 8. Its early block receives lengths:

```text
[5, 6, 7, 8, 8, 8]
```

It processes 42 positions across six calls. The first forward processes the prompt; each subsequent call still processes the whole available prefix, rather than only the one newly appended position. Deep lengths are `[3,3,4,4,4,4]` because the prefix is compressed on every call.

This confirms that “residual caching” currently denotes the sum added at the compression boundary, not a persistent KV cache across decoding steps.

## 4. Boundary loss has fewer supervised predictions

The number of deep positions and targets is exactly:

```text
S(T,R) = floor(T/R) + (T mod R).
```

Full windows select shifted-target indices `R-1,2R-1,...`; each trailing singleton selects its own original position. For `T=8,R=2`, indices are `[1,3,5,7]`: four predictions instead of the baseline's eight. For `T=7,R=2`, indices are `[1,3,5,6]`: four instead of seven.

Singleton-only batches supply `T` predictions. The experiment records exact counts for the full training run and for timed steps. Thus equal input-token budgets are not equal supervised-prediction budgets. Comparisons between the two compressed variants are paired on both budgets; their comparison with the classical baseline uses equal inputs and steps.

The validity of each selected prediction follows from causal early blocks and causal attention over completed windows and singleton tails. The regression suite compares every selected prediction against its exact corresponding prefix and checks zero gradients into future input positions.

## 5. Cropping resets positions and window alignment

After the context exceeds `block_size`, generation supplies the last `block_size` IDs. `forward` always creates positional indices `0,...,T-1` and merges windows from offset zero of that supplied sequence.

In the runtime witness, the absolute offsets of supplied prefixes are `[0,0,0,0,1,2]`. The first full window therefore changes from absolute positions `[0,1]` to `[1,2]` and then `[2,3]`, while the local position indices remain `0,...,7`.

This is confirmed against the generated sequence itself, not inferred from printed buffer bookkeeping. It is the current behavior, not a proof that this convention is optimal. Persistent caching would need an explicit policy for positions, window alignment, and replacing temporary singleton states.

## 6. Speed, memory, and accuracy are empirical

Linear projection/MLP work in deep layers scales with compressed length; the dense attention pair count scales with its square. Overall measurements also include early blocks, vocabulary projection, the selected SDPA kernels, optimizer state, launch overhead, and sampling. These facts do not imply an overall twofold speedup or lower maximum training memory when singleton-only batches remain.

See [the experiment report](experiments.md) for actual GPU measurements, paired held-out accuracy, and limitations. The report distinguishes training from generation, full-run peaks from compressed-mode peaks, and trained-model quality from fresh-model tensor-shape benchmarks.
