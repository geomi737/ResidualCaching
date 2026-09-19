# Protocol: pretrained compression retraining

## Question

Can a pretrained SmolLM2-135M model adapt to permanent mid-layer sliding token
compression without unmerging, while retaining useful next-token behaviour?

## Boundary

This is an adaptation experiment. It must not be compared directly with the
from-scratch sliding studies in `experiments/smollm/`; the initialization,
objective, and evaluation population differ.

## Run identity

Use one directory below `results/` per run, for example
`results/residual-seed42/`. Pass it explicitly to every command. Do not reuse
another experiment's results directory. The dashboard listens on port 8080 by
default; use a different port for concurrent runs.

## Procedure

1. Record the model and dataset revisions in the run directory.
2. Train one variant and seed using `train.py --output <run-directory>`.
3. Generate plots from one or two explicit local run directories and place them
   under the workspace's local `figures/` directory; never place comparison
   figures in either raw run directory.
4. Run `benchmark.py` with the same output directory; it writes a versioned
   benchmark JSON beside that run's metrics.
5. Review the raw JSONL, benchmark JSON, and figures before publishing any
   derived evidence outside this workspace.

## Acceptance criteria

A run is complete only when it has training and evaluation JSONL, an explicit
variant and seed, generated figures, and benchmark evidence (or a documented
reason the benchmark was not run).
