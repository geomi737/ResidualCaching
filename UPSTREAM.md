# Upstream provenance

The `nanoGPT/` source is based on [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT), upstream revision [`3adf61e154c3fe3fca428ad6bc3818b27a3b8291`](https://github.com/karpathy/nanoGPT/tree/3adf61e154c3fe3fca428ad6bc3818b27a3b8291).

The original nanoGPT README, notebooks, assets, configuration examples, and MIT license are retained. The public repository vendors the source files directly; users do not need Git submodules.

Project changes primarily affect `model.py`, `train.py`, `sample.py`, and `bench_ablation.py`, with a new residual-merging training configuration, documentation, and regression tests. The local upstream repository's Git metadata is not part of the published source.

## Licensing

The upstream nanoGPT code is MIT licensed, copyright (c) 2022 Andrej Karpathy. Its complete notice is preserved at [nanoGPT/LICENSE](nanoGPT/LICENSE). Project-specific original contributions and modifications are licensed under the root [Apache License 2.0](LICENSE), copyright (c) 2026 geomi737, to the extent copyright applies. Upstream material is not relicensed: its MIT terms and notices remain applicable. Redistribution of the combined project must preserve both applicable sets of notices. [NOTICE](NOTICE) records attribution.

Modified upstream files are `nanoGPT/model.py`, `nanoGPT/train.py`, and `nanoGPT/sample.py`; their headers identify project modifications. New project files (including the benchmark, residual configuration, experiments, tests, and project documentation) use Apache-2.0 unless explicitly identified otherwise. Unmodified upstream files, notebooks, and assets retain their original licensing. AI assistance and authorship provenance are described in [AUTHORS.md](AUTHORS.md); no claim is made to exclusive rights in ideas or otherwise unprotectable material.
