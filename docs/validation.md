# Validation and how to reproduce it

## CPU correctness suite

Run from the project root:

```bash
python -m unittest discover -s tests -v
```

The release preparation environment was Python 3.14 and PyTorch 2.14.0, with no CUDA device. Nine tests passed, including:

- Correct shifted-target indices for ratios 1, 2, and 3, including sequences shorter than a window and multi-token tails.
- The learned weighted merge plus residual-sum formula, singleton scaling, cache-off ablation, and legacy tail behavior.
- Explicit cross-entropy alignment, ignored targets, and finite merger gradients.
- Every full-sequence prediction matching the prediction from its corresponding exact causal prefix, with residuals on/off and singleton-only mode.
- Zero gradients into future input representations from a boundary prediction.
- The inference vocabulary projection receiving only one position and matching the last training-mode prediction.
- Generation through changing tails and context cropping.
- Invalid configuration and input rejection.
- Real singleton-only training, architecture restoration on merged resume, merge-weight updates, final-checkpoint saving, and sampling through the actual CLI scripts.

GitHub Actions runs the same suite on Python 3.11 with CPU PyTorch. GPU autocast, GPU memory use, compilation, and multi-GPU DDP are outside the local validation performed for this release.

## Benchmark smoke run

The committed [CPU smoke report](../results/smoke_cpu.json) comes from this deliberately tiny run:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B nanoGPT/bench_ablation.py --data-dir=nanoGPT/data/shakespeare --device=cpu --block-size=8 --batch-size=2 --max-iters=3 --warmup-iters=1 --eval-interval=3 --eval-iters=2 --n-layer=3 --n-head=2 --n-embd=16 --merge-layer=1 --generation-tokens=4 --generation-repeats=1 --output=results/smoke_cpu.json
```

Prepare the BPE dataset first with `cd nanoGPT && python data/shakespeare/prepare.py`, then return to the project root for the command above.

This checks that all three experiment paths, matched validation metrics, independent training/generation timers, and JSON reporting work. Three updates, two validation batches, and four generated tokens are **not sufficient** to assess quality, speed advantages, or convergence. CPU reports leave VRAM metrics null. Dataset hashes are included to identify the exact token streams.

## Controlled experiments still needed

Run longer training with several seeds, multiple context lengths, and the intended GPU. Report both input-token and supervised-prediction budgets, wall-clock time, final-prefix quality across window remainders, and generation speed. Add a scaled-average control to isolate residual amplification from learned unequal weighting. Persistent KV caching requires a separate implementation and correctness study.
