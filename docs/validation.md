# Validation and reproduction

## Correctness and workflow suite

```bash
python -m unittest discover -s tests -v
```

**Twelve tests passed with real CUDA access** on RTX 4060, Python 3.14.7, PyTorch 2.14.0/CUDA 13.3. In the restricted execution environment eleven pass and the GPU test is skipped, because those restrictions hide the GPU. The actual CUDA runs used execution outside those restrictions.

The suite covers shifted-target alignment for ratios 1/2/3 and short/multi-token tails; residual and singleton formulas; padding-aware loss; finite gradients; every boundary prediction matching its exact causal prefix; zero future-input gradients; last-position-only inference projection; cropping and generation; invalid inputs; real singleton training, merged resume, and sampling; independent collision/scaling/recomputation witnesses; and the correctness of held-out top-1 accuracy.

The GPU test checks exact-prefix prediction agreement and finite backward gradients for ratios 1/2/3 in float32 and BF16. CPU workflow tests generate their own tiny data locally and do not require downloads.

GitHub Actions is configured to run the suite on Python 3.11/CPU PyTorch. That configuration has not yet been executed remotely. Single-GPU eager BF16 training and GPU-memory/timing measurements were exercised by the actual experiments. Compilation and multi-GPU DDP remain unvalidated.

## Actual GPU experiments

See [the measured report](experiments.md), [algebraic proofs](proofs.md), and these artifacts:

- [Nine trained-model runs](../results/ablation_cuda.json), three variants and three seeds, 1,000 updates per run.
- [Sixty-nine warmed shape-study cases](../results/scaling_cuda.json), contexts 128/512/1024, multiple merge depths, and isolated singleton controls.
- [Numerical/runtime witnesses](../results/property_evidence.json).

Data preparation splits raw text before tokenization into separate train/validation/test segments. Final test targets are identical across compared models and are not used for gradient updates, monitoring, checkpoint selection, or hyperparameter tuning during these runs. The results measure sampled held-out predictions, not every token of the test corpus or general language understanding.

## CPU benchmark smoke run

The [CPU smoke report](../results/smoke_cpu.json) verifies that the current three-way runner, final test metrics, timers, plan-hash checks, and JSON output work without a GPU:

```bash
python nanoGPT/data/shakespeare_threeway/prepare.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python nanoGPT/bench_ablation.py --device=cpu --block-size=8 --batch-size=2 --max-iters=3 --warmup-iters=1 --lr-warmup-iters=1 --eval-interval=3 --eval-iters=2 --test-iters=2 --prefix-iters=2 --n-layer=3 --n-head=2 --n-embd=16 --merge-layer=1 --generation-tokens=4 --generation-repeats=1 --seeds=1337 --output=results/smoke_cpu.json
```

This deliberately tiny run is not evidence of convergence or comparative performance. Its VRAM metrics are null. Dataset hashes are included in the report.

## Remaining scope

Broader quality claims require additional datasets and tasks, widths, seeds, GPUs, and matched-supervision budgets. Explaining the residual mechanism requires controls that separate scale, constrained weighting, and singleton handling. The requested three-way experiment does not isolate those factors. Persistent KV caching requires a separate implementation and correctness study. The existing measurements already show that generation speedup and a lower full mixed-training memory maximum cannot be claimed for this setup.
