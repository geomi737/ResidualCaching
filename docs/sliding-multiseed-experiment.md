# Three-seed sliding experiment

## Question

Do the quality, memory, and inference speed differences between Baseline,
Sliding, and Sliding + R persist across random initializations? Conventional
disjoint merging and Disjoint + R training are legacy and are excluded.

## Fixed protocol

This experiment runs **nine fresh trainings from scratch**: three variants for
seeds **17, 29, and 43**. The earlier seed-11 pilot is not included in the three
repetitions. Each model receives 1500 optimizer updates, four microbatches of
four sequences per update, and alternating input lengths 255/256: 6,132,000
input tokens and supervised predictions per model. This is approximately 2.41
training-corpus token volumes, with randomly sampled windows rather than
ordered epochs. Across nine runs the budget is 13,500 updates and 55,188,000
input tokens.

Architecture, data, tokenizer, optimizer, learning-rate schedule, merge depth,
and training/inference representations match the
[single-seed protocol](sliding-merging-experiment.md). Training requires CUDA;
there is no CPU fallback and no pretrained-weight adaptation. Within each seed,
base weights and sampled training windows are identical across variants.
Across seeds, initial weights must differ. Evaluation windows use the same fixed
validation/test sampling seeds as the pilot so evaluation noise is not added
to the initialization comparison.

Run order rotates to reduce a fixed position effect:

| Seed | First | Second | Third |
|---|---|---|---|
| 17 | Baseline | Sliding | Sliding + R |
| 29 | Sliding | Sliding + R | Baseline |
| 43 | Sliding + R | Baseline | Sliding |

Runs are sequential on the same GPU. This rotation does not remove temperature,
clock, allocator, or background-load effects. Cached decoding remains a short
32-token benchmark; timing spread is not solely initialization spread.

## Measurements and analysis

Preserve all per-step loss, accuracy, gradient, learning-rate, allocated/reserved
memory, and throughput measurements. Record both dense sliding and compressed
disjoint validation/test quality, even/odd prefix quality, representation
mismatch, full-forward/prefill/decode consumption, KV bytes, and cache lengths.

Report each seed separately and aggregate **mean ± sample standard deviation**
across three seeds. Compute relative differences against the paired baseline
within each seed before aggregation; accuracy differences use percentage points.
Three seeds provide a limited stability check, not a precise confidence interval
or a general language-quality ranking. Compare Sliding directly with Sliding + R
as well as with baseline. The current test set was already inspected in the
pilot: this is an exploratory repeat, not fresh held-out confirmation.

## Reproduction and live progress

```bash
python experiments/smollm/train_sliding_multiseed.py \
  --seeds 17 29 43 --output out-sliding-three-seeds
python experiments/smollm/serve_smollm.py \
  --output out-sliding-three-seeds --port 8768
```

The controller uses the same Python environment for all child runs. The live
English dashboard lists all nine jobs and their seeds. Full child logs, JSON,
and checkpoints are in per-seed output directories. The controller validates
within-seed pairing and distinct initializations before writing combined
`results.json`, `summary.json`, and English `comparison.md`. Finalize aggregate comparisons and figures
from completed evidence with:

```bash
python experiments/reporting/finalize_sliding_multiseed.py \
  --output out-sliding-three-seeds
```

Local output directories and model checkpoints are ignored by Git. All nine jobs completed. Published results are organized in the
[results index](../results/README.md), with separate training, speed, and memory reports.

Add `--wait` to the finalizer to generate reports and figures automatically once
training completes; it exits with an error if the training suite fails.

## Repeated inference speed measurement

The training-adjacent decode timings are single 32-token measurements and do
not establish a stable speed difference. A separate, checkpoint-only benchmark
uses all three saved variants for each seed, batch 1, contexts 256 and 1024,
128 greedy decode steps, and nine timed repetitions. Each model/context receives
two warmup sequences before timing. Variant order rotates across repetitions;
all three models stay resident during each seed's measurements to avoid model
reloads between samples. Prefill and decode are timed separately with CUDA
synchronization around wall-clock intervals. FP32 weights, BF16 autocast, SDPA,
and four CPU threads match the training runner's inference settings.

```bash
python experiments/smollm/benchmark_sliding_speed.py \
  --checkpoints out-sliding-three-seeds --output out-sliding-speed-repeat \
  --seeds 17 29 43 --lengths 256 1024 --repeats 9 --decode-tokens 128
```

Individual timings, mean/median/sample SD, and total-token/total-time throughput
are preserved in `speed-benchmark.json`; `speed-benchmark.md` compares each seed
against its own baseline. Existing benchmark outputs cannot be overwritten.
The longer context is a performance workload, not a language-quality result.
Shared model residency means these runs do not measure isolated model VRAM.
Background load and GPU clock changes are not eliminated by repeated timings.

For a separate 64K performance stress test using seed 17:

```bash
python experiments/smollm/benchmark_sliding_speed.py \
  --output out-sliding-speed-64k --seeds 17 --lengths 65536 \
  --repeats 6 --decode-tokens 128
```

The prompt is passed directly to `prefill` without the generation helper's
context-cropping limit. After 128 decode steps, the logical history is 65,664
tokens. These models were trained with 255/256-token sequences; measuring a
64K workload does not demonstrate usable long-context language quality.

The matching 128K performance stress test uses the same checkpoint, warmup,
rotation, and timing protocol:

```bash
python experiments/smollm/benchmark_sliding_speed.py \
  --output out-sliding-speed-128k --seeds 17 --lengths 131072 \
  --repeats 6 --decode-tokens 128
```

The logical history after decode is 131,200 tokens. The performance-only and
single-seed limitations of the 64K stress test also apply here.

## Isolated speed and memory at 128K

A separate benchmark measures speed and GPU memory together, with **one model
per fresh subprocess**, seed 17, context 131,072, six repetitions, and 128 greedy
decode steps. Parameters are FP32, autocast is BF16, attention is SDPA, and batch
size is one. Two warmup sequences precede timing. The warmed allocator is retained;
peak counters are reset separately for prefill and decode.

```bash
python experiments/smollm/benchmark_sliding_resources.py \
  --output out-sliding-resources-128k --length 131072 --repeats 6
```

Outputs preserve weights-only allocated/reserved memory, parameter bytes, every
prefill/decode timing and allocator peak, final persistent KV bytes, and per-layer
cache lengths. Allocated includes live tensors; reserved includes allocator
caching and must not be added to allocated. Neither includes driver/context
memory or other applications. Decode peaks include prompt KV and temporary
allocations, so they need not decrease by the same percentage as persistent KV.
This isolates model residency but uses sequential variant order, so timing can
still depend on device conditions. Previous shared-residency timings are kept
separate rather than combined with the new memory measurements.

## Completed evidence

[Training and quality](reports/training.md) · [Speed](reports/speed.md) ·
[Memory](reports/memory.md). The isolated 128K results are kept separate from
the earlier shared-residency speed tests.
