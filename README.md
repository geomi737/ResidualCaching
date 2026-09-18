# ResidualCaching

**Research beta · Fully AI-assisted / vibe-coded project · Experiments ongoing**

ResidualCaching explores one-way context compression for autoregressive language
models. Early Transformer blocks read individual tokens; deeper blocks read
merged contextual representations. Project idea and architectural direction:
**[geomi737](https://github.com/geomi737)**. Gemini and ChatGPT/Codex assisted
implementation and documentation. The original nanoGPT foundation is by
**Andrej Karpathy**. This is an early research project, not a production library
or an independent audit. Publication records project provenance, not global
novelty or priority.

## Current direction: Sliding + residual, trained from scratch

The current experiment trains an eight-layer, 22.1M-parameter Llama model on
WikiText-2 from **random initialization**. SmolLM2 supplies the configuration and
tokenizer, not pretrained weights. For pairs, training keeps all positions:

```text
a = softmax(learned_weights)
z[0] = x[0]
z[t] = a[0] * x[t-1] + a[1] * x[t] + x[t-1] + x[t]    (t >= 1)

training:  [x0, x1, x2, x3] -> [x0, C(x0,x1), C(x1,x2), C(x2,x3)]
inference: [x0, x1, x2, x3] -> [C(x0,x1), C(x2,x3)]
```

The windows use original contextual states simultaneously, not recursive sums.
Every training position predicts its next token without future-token exposure.
Inference uses non-overlapping windows and an unchanged singleton tail. There
is **no unmerge**. Standard Transformer residual connections remain intact.
The extra constituent sum is added once at the merge boundary; it does not
store separate copies of the original states. The KV cache is a separate
mechanism: early layers retain all token K/V, while deep layers persist only
completed windows. Singleton tails participate in prediction without persisting
in the deep cache.

The author selected **Sliding + R** as the next direction based on its best
compressed boundary accuracy among the four merged variants. Conventional
merging remains a recorded control, not the default direction for new training.
Accuracy here means top-1 next-token accuracy on specified targets, not a
complete measure of language quality or reasoning.

## Latest GPU pilot

Five variants, one seed, identical random base weights and input batches,
1,500 updates each, WikiText-2, RTX 4060. Inference context: 256 tokens.

| Variant | Test boundary CE ↓ | Boundary accuracy ↑ | Cached decode tokens/s ↑ | Training allocated peak MiB ↓ |
|---|---:|---:|---:|---:|
| Baseline | 5.2743 | 24.71% | 137.65 | 1197.1 |
| Conventional merge | 5.4673 | 22.99% | 175.96 | 828.6 |
| Sliding | 5.3758 | 23.33% | 180.16 | 1197.3 |
| Conventional merge + R | 5.4697 | 23.07% | 181.39 | 828.6 |
| Sliding + R | **5.3573** | **23.71%** | 183.39 | 1197.3 |

Sliding + R has **1.57% higher boundary loss**, **0.99 percentage points lower
boundary accuracy**, and **33.22% higher measured cached decode throughput**
than baseline. Its persistent KV bytes are **25% lower**. Sliding keeps dense
training positions, so its training allocated peak is effectively unchanged.
Conventional merging reduces training allocated peak by **30.79%**, but supplies
**49.90% fewer supervised predictions** at the same input budget.

Sliding without R has the better merged final-prefix loss. All merged variants
have the same inference structure; their small timing differences are not
established architecture effects. These are short measurements from one seed,
not a universal winner or a statistically established speedup.

![Five-variant GPU overview](results/sliding_scratch/figures/overview.png)

[Full protocol and analysis](docs/sliding-merging-experiment.md) ·
[Every percentage metric](results/sliding_scratch/percentage_analysis.md) ·
[Exact raw measurements](results/sliding_scratch/seed11.json) ·
[Artifact provenance](results/sliding_scratch/provenance.json)

## Run the current experiment

Training requires a CUDA GPU. Use an existing CUDA PyTorch environment or create
one before installing the requirements. Tests and report generation also work
on CPU. Run commands from the repository root.

```bash
python -m pip install -r requirements-smollm.txt -r requirements-experiments.txt
hf download HuggingFaceTB/SmolLM2-135M config.json tokenizer.json special_tokens_map.json merges.txt vocab.json --revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2 --local-dir models/SmolLM2-135M
hf download Salesforce/wikitext README.md wikitext-2-raw-v1/train-00000-of-00001.parquet wikitext-2-raw-v1/validation-00000-of-00001.parquet wikitext-2-raw-v1/test-00000-of-00001.parquet --repo-type dataset --revision b08601e04326c79dfdd32d625aee71d232d685c3 --local-dir data/smollm/wikitext
python experiments/smollm/prepare_smollm.py
python experiments/smollm/train_sliding.py --device cuda
```

Default runs compare baseline and Sliding + R, both from scratch. To reproduce
the completed five-variant pilot, use the command in the
[experiment protocol](docs/sliding-merging-experiment.md#reproduction).

In another terminal, show progress updated every two seconds:

```bash
python experiments/smollm/serve_smollm.py --output out-sliding-scratch-smollm --port 8767
```

Open [the local dashboard](http://127.0.0.1:8767). Reports and checkpoints are
written to `out-sliding-scratch-smollm/`, which is ignored by Git. Published
results contain measurements, not corpus text or checkpoints.

## Validate and regenerate published figures

```bash
python -m unittest discover -s tests -v
python experiments/reporting/export_sliding_analysis.py
MPLCONFIGDIR=/tmp/residual-matplotlib python experiments/reporting/plot_sliding_results.py
```

## Repository guide

| Location | Purpose |
|---|---|
| `experiments/smollm/` | Active scratch runner, merging adapters, data preparation, and dashboard |
| `experiments/smollm/legacy/` | Earlier tiny, pretrained adaptation, and unmerge experiments |
| `experiments/nanogpt/` | Historical nanoGPT benchmarks and numerical witnesses |
| `experiments/reporting/` | Evidence-derived reports and reproducible plotting |
| `results/sliding_scratch/` | Latest raw results, percentages, source metadata, PNG/SVG figures |
| `docs/` | Current protocol, architecture, validation, and historical experiment reports |
| `docs/archive/` | Previous proposals retained for research provenance |
| `archive/prototypes/` | Non-causal or leaking exploratory prototypes; not quality evidence |
| `nanoGPT/` | Attributed upstream implementation and the earlier causal model |
| `tests/` | Causality, formulas, loss alignment, cache agreement, evidence, and workflows |

See the [experiment index](experiments/README.md), [results index](results/README.md),
[documentation index](docs/README.md), and [roadmap](docs/roadmap.md).

## Earlier work

The [nanoGPT report](docs/experiments.md) preserves the three-seed Shakespeare
experiment and explicitly labels its retired bypass/doubled-tail protocol.
The [earlier SmolLM2 report](docs/smollm-next-experiment.md) describes pretrained
feasibility and adaptation separately. Those measurements are not results of
the new from-scratch sliding protocol. See [architecture](docs/architecture.md)
and [algebraic limits](docs/proofs.md) for distinctions between weighted sums,
residual scaling, compression, and persistent caching.

## Acknowledgments

Thank you to **Andrej Karpathy** for nanoGPT and his educational work, the
**SmolLM2 authors and Hugging Face** for the compact model family, tokenizer and
Transformers tooling, and **Stephen Merity, Caiming Xiong, James Bradbury,
Richard Socher, Salesforce Research, and Wikipedia contributors** for WikiText.
The current experiment could not be reproduced without these foundations.
See [credits](AUTHORS.md) and [pinned source provenance](UPSTREAM.md).

## License and attribution

Project-specific contributions are [Apache-2.0](LICENSE); upstream nanoGPT
retains its [MIT license](nanoGPT/LICENSE). Preserve applicable notices,
including [NOTICE](NOTICE). Model/tokenizer and dataset terms remain separate;
see [UPSTREAM.md](UPSTREAM.md). AI assistance does not imply upstream endorsement.
Please cite [CITATION.cff](CITATION.cff) when discussing this project. The citation
request does not add conditions to Apache-2.0 or claim exclusive ownership of
mathematical ideas.
