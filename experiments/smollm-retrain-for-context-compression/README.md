# Pretrained SmolLM compression retraining

This workspace tests whether a pretrained SmolLM2 model can adapt to permanent
mid-layer sliding token compression (plain or residual) without unmerging. It
is an adaptation study, not a continuation of the from-scratch sliding
experiments elsewhere in the repository.

## Workspace contract

| Path | Role |
|---|---|
| `train.py` | Adaptation runner and live status API |
| `dashboard.html` | Dashboard served only by this runner (default port 8080) |
| `plot_results.py` | Recreates this workspace's figures from local evaluation JSONL |
| `benchmark.py` | Writes local prefill, decode, cache, and memory evidence |
| `generate.py` | Qualitative baseline/student comparison |
| `docs/PROTOCOL.md` | Fixed procedure, boundary, and completion criteria |
| `results/` | Ignored checkpoints, JSONL, and benchmark JSON for this experiment only |
| `figures/` | Figures created only from this experiment's local results |

`results/` is deliberately ignored by Git. Review a run locally before copying
curated evidence to the repository-level `results/` publication archive.

## Run one isolated experiment

Run from the repository root. Use a unique directory per variant/seed; do not
reuse another experiment's output path.

```bash
RUN_DIR=experiments/smollm-retrain-for-context-compression/results/residual-seed42
python experiments/smollm-retrain-for-context-compression/train.py \
  --variant residual --seed 42 --output "$RUN_DIR" --port 8080
python experiments/smollm-retrain-for-context-compression/benchmark.py \
  --variant residual --output "$RUN_DIR"
python experiments/smollm-retrain-for-context-compression/plot_results.py \
  --residual-results "$RUN_DIR" \
  --figures experiments/smollm-retrain-for-context-compression/figures/residual-seed42
```

Open <http://127.0.0.1:8080> while the training process is running. For a
second concurrent run, choose both a new `RUN_DIR` and a new `--port`.

For a control-versus-residual figure, train each variant into its own run
directory, then pass both `--control-results` and `--residual-results` to the
plotter. The generated comparison is never written into either raw run folder.

## Metrics

The runner records student and teacher loss, perplexity, next-token accuracy,
teacher/student gap, KL divergence, throughput, VRAM usage, and checkpoint
state. The benchmark writes a schema-versioned JSON record with context length,
prefill latency, decode throughput, cache size, and peak allocator memory.

See [the protocol](docs/PROTOCOL.md) for comparability limits and acceptance
criteria.
