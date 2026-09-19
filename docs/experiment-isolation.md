# Experiment isolation policy

Each new research question gets one directory under `experiments/` with a
stable, descriptive slug. The directory owns its runner, dashboard, plotting
script, protocol, and ignored `results/` directory. It may import a shared
model implementation, but it must not read or write another experiment's
configuration, run state, figures, checkpoints, or raw metrics.

## Required layout

```text
experiments/<experiment-slug>/
├── README.md              # question, scope, commands, and limitations
├── train.py               # one experiment's runner
├── dashboard.html         # one experiment's live UI
├── plot_results.py        # one experiment's figure generator
├── docs/PROTOCOL.md       # fixed procedure and acceptance criteria
├── figures/               # generated, reviewable figures for this experiment
└── results/               # ignored raw logs, checkpoints, and benchmark JSON
```

The repository-level `results/` directory is publication-only. Copy data there
only after review, document its provenance, and add a matching report. Never
point a live run at that directory.

## Isolation rules

1. Give every runner a local default output path derived from its own directory.
2. Use a unique dashboard port in the experiment protocol.
3. Keep figures in the experiment's `figures/` directory and regenerate them
   from that experiment's local metrics.
4. Include a schema version and experiment slug in machine-readable outputs.
5. Do not reuse a prior experiment's checkpoint unless the protocol explicitly
   describes it as an input and records its checksum.

The copyable workspace is at
[`experiments/experiment-template`](../experiments/experiment-template/README.md).
