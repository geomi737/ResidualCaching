# ResidualCaching

**Research beta · Fully AI-assisted / vibe-coded project · Experiments ongoing**

This is an early research beta, not a production-ready optimization library. The project-specific implementation and documentation were developed through a fully AI-assisted, "vibe-coded" workflow with Gemini and ChatGPT/Codex under geomi737's direction, on top of Andrej Karpathy's independently authored nanoGPT. Tests and reported experiments provide bounded evidence; they are not a comprehensive independent audit.

The author is publishing early to document the idea's provenance and reduce the risk of it being presented without attribution. This records this project's development, not a verified global priority claim. **The experiment will continue**: implementation, measurements, and conclusions may change. See the [research roadmap](docs/roadmap.md).

**Causal Token Merging with Residual Caching** is an experimental, one-way context compression architecture for autoregressive language models, built on [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT).

The idea: process tokens at full resolution in the first few Transformer blocks, merge completed windows, and let deeper blocks operate on fewer positions. Add the sum of the original contextual token vectors as a residual path. Predict directly from the compressed sequence: **no unmerging**.

Project idea and architectural direction: **[geomi737](https://github.com/geomi737)**. Development was assisted by **Gemini** and **ChatGPT (including Codex)**. See [credits and provenance](AUTHORS.md).

[Architecture](docs/architecture.md) · [Proofs](docs/proofs.md) · [GPU experiments](docs/experiments.md) · [Validation](docs/validation.md) · [Original nanoGPT README](nanoGPT/README.md)

## Architecture

For a completed window of size `R`, using contextual representations after the early blocks:

```text
C(x₀, …, xᵣ₋₁) = Σ softmax(w)ᵢ xᵢ + Σ xᵢ

[x₀, x₁, x₂, x₃, x₄] → [C(x₀,x₁), C(x₂,x₃), C(x₄)]
                                      singleton C(x₄) = 2x₄

full-length early blocks → compression → shorter deep blocks → LM head
```

Singletons use the same residual formula. With `use_residual_cache=False`, a singleton is `x`; `residual_tail=False` reproduces the earlier pass-through tail convention. `merge_ratio=1` is the ordinary nanoGPT baseline with no extra residual scaling.

Each compressed window predicts the token immediately after its last original position. Given shifted targets `Y[t] = X[t+1]`, completed windows select indices `R−1, 2R−1, …`; singleton tails select their original positions. This **window-boundary loss** supplies approximately `T/R` predictions in one pass without exposing future tokens to the prediction.

## Quick start

Requires Python 3.10+ and PyTorch 2.0+.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
cd nanoGPT
python data/shakespeare_char/prepare.py
python train.py config/train_residual_shakespeare.py
python sample.py --out_dir=out-residual-shakespeare
```

For a small CPU experiment:

```bash
cd nanoGPT
python train.py config/train_residual_shakespeare.py --device=cpu --dtype=float32 --compile=False --n_layer=3 --n_head=2 --n_embd=64 --block_size=64 --batch_size=8 --eval_iters=5 --eval_interval=50 --max_iters=200 --gradient_accumulation_steps=1 --warmup_iters=10 --lr_decay_iters=200
python sample.py --out_dir=out-residual-shakespeare --device=cpu --dtype=float32 --num_samples=1 --max_new_tokens=100
```

Dataset preparation downloads Karpathy's Tiny Shakespeare text. Raw data, checkpoints, model downloads, and local caches are excluded from this repository.

## Configuration

| Setting | Meaning |
| --- | --- |
| `merge_ratio=2` | Completed window size; `1` disables compression |
| `merge_layer=2` | Number of full-resolution blocks; compression precedes block index 2 |
| `use_residual_cache=True` | Add the sum of the window's contextual vectors |
| `residual_tail=True` | Apply `x+x` to singleton representations |
| `unmerged_prob=0.1` | Train 10% of batches as singleton-only sequences in the merged architecture |

Merged training samples lengths from `max(1, block_size−R+1)` through `block_size`, covering every remainder when `block_size>=R`. Mixed `[C,…,C,x]` contexts are trained through tails; singleton-only batches additionally expose deep layers to fully uncompressed contexts. Set `unmerged_prob=0.0` to train only compressed-window contexts and their tails.

Resume restores the architecture from the checkpoint:

```bash
cd nanoGPT
python train.py config/train_residual_shakespeare.py --init_from=resume
```

The final update is saved even when it falls between evaluation intervals. `sample.py` reads the saved architecture. Older checkpoints without `residual_tail` retain the old tail convention on resume/sample.

## Reproducible ablation

```bash
python nanoGPT/data/shakespeare_threeway/prepare.py
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python nanoGPT/bench_ablation.py --device=cuda --seeds 1337 2027 3407 --prefix-iters=128 --output=results/ablation_cuda.json
```

This runs baseline nanoGPT, merged+residual, and merged without residual on disjoint 80/10/10 train/validation/test splits. Validation monitors common boundary targets; final test scores both common boundaries and next-token predictions with complete windows and singleton tails. Accuracy means top-1 next-token accuracy. Initial weights and batch plans are checked for equality within each seed.

Training timers exclude validation and warmup; actual input-token and supervised-prediction throughput are reported separately. Autoregressive generation has its own warmed-up timer. Allocated and reserved GPU memory peaks are recorded separately for training and generation. Optional `--checkpoint-dir=out-ablation` retains the final model and optimizer states for each run.

## Research status

Actual RTX 4060 measurements across three seeds and 1,000 updates per variant are recorded in [the GPU report](docs/experiments.md). On Tiny Shakespeare, residual aggregation improved plain merging: final-prefix test loss **5.533 vs 5.601**, accuracy **21.84% vs 21.14%**. Classical nanoGPT retained better quality: **5.407**, **22.61%**. These are scoped measurements on one dataset, not universal quality claims.

The residual configuration processed training inputs about **1.42x faster** than baseline, but produced fewer supervised predictions per input batch and had lower predictions/s. Generation acceleration was not demonstrated under the tested shapes. The mixed-training peak stayed around **1453 MiB** because singleton-only batches remain full sized. Isolated compressed-mode microbenchmarks used about **834.5 MiB vs 1454.5 MiB**; they measure compute/memory rather than trained long-context quality.

All **12 correctness/workflow tests pass with GPU access**, including float32 and BF16 causality/backward checks. No persistent KV cache is implemented: generation recomputes the cropped prefix. The aggregate is a rescaled constrained weighted average, not lossless individual-vector storage; [algebraic proofs and runtime witnesses](docs/proofs.md) explain the distinction.

`merged_transformer.py` and `test_generative_merged.py` are historical prototypes. The latter has future-token leakage through unmerging and is not the supported causal implementation. Use `nanoGPT/model.py` and `tests/` for the current implementation.

## Acknowledgments and upstream license

Respect and thanks to **Andrej Karpathy** for nanoGPT and for making Transformer implementations approachable, readable, and easy to experiment with. This project builds on his implementation; the original README and [MIT license](nanoGPT/LICENSE) are preserved. See [UPSTREAM.md](UPSTREAM.md) for the exact source revision and [AUTHORS.md](AUTHORS.md) for human and AI assistance credits.

## License and attribution

Project-specific contributions are licensed under [Apache License 2.0](LICENSE). Retain applicable copyright, license, and attribution notices when redistributing this code or derivatives, including the notices in [NOTICE](NOTICE). Modified files must identify changes as required by the license. The original nanoGPT material remains under its [MIT license](nanoGPT/LICENSE); its notices must also be preserved. See [licensing scope](UPSTREAM.md#licensing).

Please cite this repository when discussing or building on the experiment; [CITATION.cff](CITATION.cff) provides citation metadata. This citation request does not add conditions to Apache-2.0. The license does not require a public acknowledgment for every private use and does not give exclusive ownership of the underlying mathematical idea.
