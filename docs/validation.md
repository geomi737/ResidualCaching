# Validation and reproduction

## Current correctness suite

```bash
python -m unittest discover -s tests -v
```

The current suite has **26 test cases**. After repository organization, 22 of the
23 original/adapter cases passed on CPU and one CUDA-only case skipped under
restricted execution. Three new artifact-integrity cases passed separately.
The skipped CUDA causality/backward case then passed on RTX 4060 outside the
restricted environment. Thus all 26 cases were exercised across CPU and GPU.
No remote CI run is claimed.

Coverage includes:

- nanoGPT shifted-target selection for R=1/2/3, short/multi-position tails,
  aggregation formulas, loss and padding, no future gradients, generation,
  checkpoint resume and sampling.
- Llama baseline equivalence to native Hugging Face logits/loss, contextual
  pair formulas, gradients, causal prefix equivalence, and persistent caching.
- Sliding weighted/residual pair formulas without recursive aggregation,
  future independence, dense causal-prefix agreement, and mode dispatch.
- Cached/full-prefix agreement for both parities, singleton-only prefixes,
  multiple merge boundaries, generation and context cropping. Current deep
  caches persist completed windows only.
- Preserved GPU artifact checksums, identical initialization/data plans,
  matching dataset token hashes, equal-input/different-supervision budgets,
  matched evaluation target counts, and actual per-layer KV cache lengths.

GitHub Actions runs CPU correctness on Python 3.11 with Transformers 4.57.6.
The repaired adapter import now runs SmolLM tests rather than accidentally
skipping them through an obsolete module path. Tests require no corpus download.
Compilation, padded/variable-length cache batches, ratios beyond R=2 for the
current Llama cache, and multi-GPU training remain outside the verified scope.

## Current scratch evidence

See the [full protocol](sliding-merging-experiment.md) and
[result index](../results/README.md). Exact evidence is preserved in
[seed11.json](../results/training/seed11_pilot/seed11.json), with checksums in
[provenance.json](../results/training/seed11_pilot/provenance.json).
Five models were trained on actual WikiText-2 text from random initialization
on RTX 4060, 1,500 updates each. Initial weights and data/input plans are paired.
Test evaluates 65,536 common boundary targets and 512 final-prefix targets.

Recreate English metric tables and the three PNG/SVG figure sets:

```bash
python experiments/reporting/export_sliding_analysis.py
MPLCONFIGDIR=/tmp/residual-matplotlib python experiments/reporting/plot_sliding_results.py
```

Figures were visually inspected for readable labels, correct variant ordering,
units and source values. Raw evidence was copied byte-for-byte, and SHA-256
checks verify it. Published Markdown links resolve locally. Corpus text, token
binaries, downloaded model assets and checkpoints remain excluded from Git.

## Earlier evidence

The [nanoGPT report](experiments.md) contains separately labeled historical
measurements: nine trained runs across three seeds, 69 random-input shape
cases, and numerical witnesses. Its retired compression-bypass/doubled-tail
protocol does not describe the current scratch pilot. The
[pretrained SmolLM2 page](smollm-next-experiment.md) records earlier feasibility
and adaptation work, not new scratch quality.

`results/legacy/nanogpt/smoke_cpu.json` and local synthetic GPU smoke runs verify execution
only. Random-input loss must not be interpreted as language quality.

## Limits

One seed and short sequential inference timers do not establish significance
of small variant differences. Top-1 accuracy is not a reasoning or preference
benchmark; cross-entropy and final-prefix behavior are reported alongside it.
Evaluation windows can overlap. The post-test choice of Sliding + R is an
exploratory research direction to confirm on fresh targets. No lossless
compression, universal quality win, or production-readiness claim is made.

## Publication integrity

The reorganized evidence passed 26 existing CPU/GPU checks and four additional
publication checks: SHA-256 integrity, three paired fresh training seeds, matching
prompts and timing-repeat counts, and isolated 128K KV lengths/peak-memory semantics.
All repository/report Markdown links were checked after the moves.
