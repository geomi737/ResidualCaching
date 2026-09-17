# Upstream provenance

The `nanoGPT/` source is based on [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT), upstream revision [`3adf61e154c3fe3fca428ad6bc3818b27a3b8291`](https://github.com/karpathy/nanoGPT/tree/3adf61e154c3fe3fca428ad6bc3818b27a3b8291).

The original nanoGPT README, notebooks, assets, configuration examples, and MIT license are retained. The public repository vendors the source files directly; users do not need Git submodules.

Project changes primarily affect `model.py`, `train.py`, `sample.py`, and `bench_ablation.py`, with a new residual-merging training configuration, documentation, and regression tests. The local upstream repository's Git metadata is not part of the published source.

## Licensing

The upstream nanoGPT code is MIT licensed, copyright (c) 2022 Andrej Karpathy. Its complete notice is preserved at [nanoGPT/LICENSE](nanoGPT/LICENSE). This release does not introduce a separate license grant for the project's original contributions.
