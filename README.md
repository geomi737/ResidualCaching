# ResidualCaching

**Causal Token Merging with Residual Caching** is an experimental, one-way context compression architecture for autoregressive language models, built on [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT).

The idea: process tokens at full resolution in the first few Transformer blocks, merge completed windows, and let deeper blocks operate on fewer positions. Add the sum of the original contextual token vectors as a residual path. Predict directly from the compressed sequence: **no unmerging**.

Project idea and architectural direction: **[geomi737](https://github.com/geomi737)**. Development was assisted by **Gemini** and **ChatGPT (including Codex)**. See [credits and provenance](AUTHORS.md).

[Русское описание](README.ru.md) · [Architecture and limitations](docs/architecture.md) · [Validation](docs/validation.md) · [Original nanoGPT README](nanoGPT/README.md)

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
cd nanoGPT
python data/shakespeare/prepare.py
python bench_ablation.py --output=../results/ablation.json
```

This runs baseline nanoGPT, merged+residual, and merged without residual. Validation uses the same boundary targets and the same final-prefix targets. Training timings exclude validation and warmup; actual input-token throughput and supervised-prediction throughput are reported separately. Autoregressive generation has a separate warmed-up timer. GPU peak allocated memory is reported only on CUDA, separately for training and generation.

## Research status

Correctness and entry-point smoke tests pass on CPU. Quality gains, generation throughput, and GPU memory savings have **not** been established by those tests. No persistent KV cache is implemented: generation recomputes the cropped prefix. The residual sum is an implicit aggregated memory path, not lossless storage of individual tokens. At initialization, equal merge weights make it a scaled average. Read [the architecture discussion](docs/architecture.md) before interpreting ablation results.

`merged_transformer.py` and `test_generative_merged.py` are historical prototypes. The latter has future-token leakage through unmerging and is not the supported causal implementation. Use `nanoGPT/model.py` and `tests/` for the current implementation.

## Acknowledgments and upstream license

Respect and thanks to **Andrej Karpathy** for nanoGPT and for making Transformer implementations approachable, readable, and easy to experiment with. This project builds on his implementation; the original README and [MIT license](nanoGPT/LICENSE) are preserved. See [UPSTREAM.md](UPSTREAM.md) for the exact source revision and [AUTHORS.md](AUTHORS.md) for human and AI assistance credits.
