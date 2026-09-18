# Experiment index

Run scripts from the repository root. Active training is always from scratch
and requires CUDA. New default comparisons use baseline and Sliding + R.

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

Raw results are in [results](../results/README.md). Do not mix historical
adaptation, random-input smoke tests, and scratch quality measurements.
