# Result artifact index

## Current scratch pilot

[`sliding_scratch/`](sliding_scratch/) contains the completed five-variant,
one-seed WikiText-2 pilot on RTX 4060:

- [`seed11.json`](sliding_scratch/seed11.json): exact original combined JSON,
  including all 7,500 update records, quality metrics, configs, hashes, and timings.
- [`dataset_manifest.json`](sliding_scratch/dataset_manifest.json): original
  split sizes and corpus/tokenizer revision/hash metadata.
- [`provenance.json`](sliding_scratch/provenance.json): artifact checksums and
  explanation of the manifest note inherited from the old adaptation workflow.
- [`percentage_analysis.md`](sliding_scratch/percentage_analysis.md) and
  [`percentage_analysis.json`](sliding_scratch/percentage_analysis.json): derived
  English absolute and relative metrics. Raw numbers are not rewritten.
- [`figures/`](sliding_scratch/figures/): overview, learning curves, and
  representation-mode diagnostics, each in PNG and SVG.

See the [protocol and interpretation](../docs/sliding-merging-experiment.md).

## Earlier artifacts

| Files | Meaning |
|---|---|
| `ablation_cuda.json`, `ablation_cuda.png` | Historical nanoGPT trained comparison, three seeds; retired protocol |
| `scaling_cuda.json`, `scaling_cuda.png` | Random-input nanoGPT shape/memory benchmarks, not language-quality evidence |
| `property_evidence.json` | Algebraic and runtime witnesses |
| `smoke_cpu.json` | CPU workflow smoke, not convergence evidence |
| `smollm135m_feasibility_*.json` | Pretrained 135M CUDA feasibility probes |
| `smollm_cuda_cache_check.json` | Earlier adapter cached/full-prefix diagnostic |

Corpora, model downloads, token binaries, and checkpoints remain in ignored
local directories and are not redistributed with these measurements.
