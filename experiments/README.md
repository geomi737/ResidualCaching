# Experiment index

Every active experiment is a self-contained workspace. Its README is the
source of truth for its question, run commands, dashboard, plotting command,
and local `results/` directory. Local run artifacts must never be written to
the repository-level `results/` tree: that tree is reserved for reviewed,
published evidence.

Run scripts from the repository root. Active training is always from scratch
and requires CUDA. New default comparisons use baseline, Sliding, and Sliding + R.
Conventional disjoint merging and Disjoint + R training are legacy; their
implementation remains available for reproducing the historical comparison.

| Group | Entry points | Status |
|---|---|---|
| Current scratch protocol | `smollm/train_sliding.py`, `smollm/sliding_model.py` | Active; [protocol](../docs/sliding-merging-experiment.md) |
| Shared causal disjoint adapter | `smollm/smollm_model.py` | Used by current inference; R=2, unpadded equal-length batches |
| Data and dashboard | `smollm/prepare_smollm.py`, `smollm/serve_smollm.py`, `smollm/sliding_dashboard.html` | Active |
| Earlier pretrained probes | `smollm/probe_smollm.py`, `smollm/check_smollm_cache.py` | Historical feasibility/cache checks |
| Legacy training | `smollm/legacy/` | Earlier tiny training, adaptation, and leaking unmerge experiment |
| nanoGPT evidence | `nanogpt/benchmark_scaling.py`, `nanogpt/verify_properties.py` | Earlier shape study and numerical witnesses |
| Reports | `reporting/export_sliding_analysis.py`, `reporting/plot_sliding_results.py` | Current English percentage report and figures |
| Other reports | `reporting/plot_results.py`, `reporting/write_report.py`, `reporting/plot_smollm_results.py` | Historical result formats |
| Pretrained compression retraining | `smollm-retrain-for-context-compression/` | Isolated active workspace; see its [README](smollm-retrain-for-context-compression/README.md) |

Raw results are in [results](../results/README.md). Do not mix historical
adaptation, random-input smoke tests, and scratch quality measurements.

Use [the experiment workspace template](experiment-template/README.md) when
starting a new experiment. Copy it first; do not add new runs to an existing
experiment directory.

Three-seed repetition: `smollm/train_sliding_multiseed.py` runs nine fresh CUDA trainings;
`reporting/plot_sliding_multiseed.py` plots aggregate and individual seed results.
See the [protocol](../docs/sliding-multiseed-experiment.md).
