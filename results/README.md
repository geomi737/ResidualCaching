# Research results

Active variants: **Baseline, Sliding, Sliding + R**, trained from scratch.
Conventional disjoint merging and Disjoint + R training are legacy.

| Category | Report | Evidence and scope |
|---|---|---|
| Training and quality | [Three-seed comparison](training/three_seeds/comparison.md) | Nine fresh GPU trainings, seeds 17/29/43, 1500 updates each; mean and sample SD |
| Historical scratch pilot | [Seed-11 percentages](training/seed11_pilot/percentage_analysis.md) | Five variants, one seed; includes legacy merging controls and short one-shot inference timings |
| Speed, short context | [256/1024](speed/short_context_3seeds/speed-benchmark.md) | Three seeds, nine rotated repeats, shared model residency |
| Speed, context sweep | [2K–16K](speed/long_context_seed17/speed-benchmark.md) | Seed 17, six rotated repeats |
| Speed, 64K | [64K](speed/64k_seed17/speed-benchmark.md) | Seed 17, six rotated repeats |
| Speed, 128K | [128K shared residency](speed/128k_seed17/speed-benchmark.md) | Seed 17, six rotated repeats; separate from isolated memory test |
| Memory and speed together | [128K isolated](memory/128k_isolated/comparison.md) | One model per fresh process; allocator peaks, exact KV bytes, and speed measured together |
| Earlier research | [Legacy index](legacy/README.md) | nanoGPT shape/training and pretrained SmolLM feasibility artifacts |

[Training interpretation](../docs/reports/training.md) ·
[Speed interpretation](../docs/reports/speed.md) ·
[Memory interpretation](../docs/reports/memory.md) ·
[Reproduction protocol](../docs/sliding-multiseed-experiment.md)

Raw JSON accompanies every report. [checksums.json](checksums.json) preserves
SHA-256 checksums of published JSON evidence, excluding itself. Input data,
model downloads, checkpoints, local logs, and process IDs are not redistributed.

Timing repeats within one seed are not independent training repetitions.
Long-context tests measure performance, not language quality. Allocated and
reserved memory are distinct and must not be added. Shared-residency speed and
isolated memory measurements are kept separate.
