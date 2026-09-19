# Experiment workspace template

Copy this directory to `experiments/<descriptive-slug>/` before adding a new
research question. Do not place a new run in an existing experiment workspace.

## Required contents

```text
<slug>/
├── README.md
├── train.py
├── dashboard.html
├── plot_results.py
├── docs/PROTOCOL.md
├── figures/
└── results/                 # ignored local run artifacts
```

Document the hypothesis, comparison boundary, inputs, unique dashboard port,
run command, output schema, and publication criteria in the local README and
protocol. Set every CLI's default `--output` to this workspace's `results/`
directory. The root `.gitignore` prevents those local artifacts from entering
Git; only reviewed evidence belongs in the top-level `results/` archive.

Follow the full [experiment isolation policy](../../docs/experiment-isolation.md).
