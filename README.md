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

The active priorities are **Sliding** and **Sliding + R**, trained from scratch
and compared against baseline. Sliding + R had the best compressed boundary
accuracy among the four merged variants in this pilot; Sliding had the better
final-prefix loss. Conventional disjoint merging and its residual variant are
**legacy**: their results remain available for historical comparison.
Accuracy here means top-1 next-token accuracy on specified targets, not a
complete measure of language quality or reasoning.

## Latest results

Nine fresh GPU trainings: Baseline, Sliding, and Sliding + R for seeds 17/29/43,
1500 updates each on WikiText-2. Mean ± sample standard deviation:

| Compressed inference quality | Baseline | Sliding | Sliding + R |
|---|---:|---:|---:|
| Boundary accuracy | **24.74 ± 0.12%** | 23.07 ± 0.05% | 23.40 ± 0.30% |
| Boundary cross-entropy ↓ | **5.256 ± 0.010** | 5.402 ± 0.008 | 5.375 ± 0.020 |
| Final-prefix cross-entropy ↓ | **5.224 ± 0.031** | 5.413 ± 0.056 | 5.542 ± 0.053 |

Sliding + R improves compressed boundary accuracy over Sliding in all three
seeds, averaging **+0.34 percentage points**, but has worse final-prefix loss in
all three seeds. Both remain active priorities; baseline retains better quality.
Dense sliding training keeps the number of positions and does not save training
allocated memory. Conventional merging training is **legacy**.

![Three-seed quality and training overview](results/training/three_seeds/figures/overview.png)

Repeated checkpoint-only timing finds no speed advantage at 256/1024 tokens.
Prefill improves by 2048 tokens. A substantial decode improvement appears at
64K. Long-context quality has not been evaluated.

**128K speed and memory measured together**, seed 17, six warmed repeats,
RTX 4060, batch one, FP32 weights/BF16 autocast/SDPA, one model per fresh process:

| Metric | Baseline | Sliding | Sliding + R |
|---|---:|---:|---:|
| Prefill time ↓ | 3.981 s | 2.463 s | 2.417 s |
| Prefill throughput difference | reference | **+61.62%** | **+64.70%** |
| Decode tokens/s ↑ | 33.85 | 46.22 | 46.81 |
| Decode throughput difference | reference | **+36.54%** | **+38.29%** |
| Prefill allocated peak, MiB ↓ | 2702.5 | **1934.5 (−28.42%)** | **1934.5 (−28.42%)** |
| Decode allocated peak, MiB ↓ | 1727.2 | **1679.2 (−2.78%)** | **1679.2 (−2.78%)** |
| Persistent KV, MiB ↓ | 1537.5 | **1153.1 (−25%)** | **1153.1 (−25%)** |

Memory is PyTorch allocator memory, excluding CUDA context, driver and other
applications. KV savings do not imply equal whole-model peak savings. Long-context
benchmarks use a 22M-parameter model trained on 255/256-token sequences; they
measure performance, not usable long-context quality. Speed results depend on
this implementation and hardware. Shared-residency sweep timings and isolated
memory/speed timings are separate experiments.

![Repeated context speed sweep](results/speed/figures/context_sweep.png)

[Training report](docs/reports/training.md) · [Speed report](docs/reports/speed.md) ·
[Memory report](docs/reports/memory.md) · [Raw results and checksums](results/README.md) ·
[Original five-variant pilot](docs/sliding-merging-experiment.md)

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
python experiments/reporting/plot_published_benchmarks.py
MPLCONFIGDIR=/tmp/residual-matplotlib python experiments/reporting/plot_sliding_results.py
```

## Repository guide

| Location | Purpose |
|---|---|
| `experiments/smollm/` | Active scratch runner, merging adapters, data preparation, and dashboard |
| `experiments/smollm/legacy/` | Earlier tiny, pretrained adaptation, and unmerge experiments |
| `experiments/nanogpt/` | Historical nanoGPT benchmarks and numerical witnesses |
| `experiments/reporting/` | Evidence-derived reports and reproducible plotting |
| `results/training/` | Three-seed quality/training and original pilot evidence |
| `results/speed/` | Repeated speed measurements from 256 to 128K |
| `results/memory/` | Isolated joint memory/speed measurements |
| `results/legacy/` | Earlier nanoGPT and pretrained SmolLM evidence |
| `docs/reports/` | English training, speed, and memory interpretation |
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
