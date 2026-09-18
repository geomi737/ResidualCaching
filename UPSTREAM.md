# Upstream provenance

The `nanoGPT/` source is based on [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT), upstream revision [`3adf61e154c3fe3fca428ad6bc3818b27a3b8291`](https://github.com/karpathy/nanoGPT/tree/3adf61e154c3fe3fca428ad6bc3818b27a3b8291).

The original nanoGPT README, notebooks, assets, configuration examples, and MIT license are retained. The public repository vendors the source files directly; users do not need Git submodules.

Project changes primarily affect `model.py`, `train.py`, `sample.py`, and `bench_ablation.py`, with a new residual-merging training configuration, documentation, and regression tests. The local upstream repository's Git metadata is not part of the published source.

## Licensing

The upstream nanoGPT code is MIT licensed, copyright (c) 2022 Andrej Karpathy. Its complete notice is preserved at [nanoGPT/LICENSE](nanoGPT/LICENSE). Project-specific original contributions and modifications are licensed under the root [Apache License 2.0](LICENSE), copyright (c) 2026 geomi737, to the extent copyright applies. Upstream material is not relicensed: its MIT terms and notices remain applicable. Redistribution of the combined project must preserve both applicable sets of notices. [NOTICE](NOTICE) records attribution.

Modified upstream files are `nanoGPT/model.py`, `nanoGPT/train.py`, and `nanoGPT/sample.py`; their headers identify project modifications. New project files (including the benchmark, residual configuration, experiments, tests, and project documentation) use Apache-2.0 unless explicitly identified otherwise. Unmodified upstream files, notebooks, and assets retain their original licensing. AI assistance and authorship provenance are described in [AUTHORS.md](AUTHORS.md); no claim is made to exclusive rights in ideas or otherwise unprotectable material.

## SmolLM2, Transformers, and WikiText

Thank you to the SmolLM2 team and Hugging Face for their model family, tokenizer,
configuration, and Transformers implementation. The current experiment trains a
22,123,296-parameter Llama model from random initialization using a reduced
SmolLM2 configuration and the SmolLM2 tokenizer. **No pretrained model weights
are loaded in the current experiment.** Historical adaptation runners remain in
`experiments/smollm/legacy/`.

- Model/configuration/tokenizer: [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/README.md), revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.
- Implementation dependency: Hugging Face Transformers **4.57.6**. The model card identifies Apache-2.0 licensing; Transformers is also Apache-2.0. Project-specific adapters do not claim authorship of the upstream architecture or tokenizer.
- Data: [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext/blob/b08601e04326c79dfdd32d625aee71d232d685c3/README.md), configuration `wikitext-2-raw-v1`, revision `b08601e04326c79dfdd32d625aee71d232d685c3`.

We thank Stephen Merity, Caiming Xiong, James Bradbury, Richard Socher,
Salesforce Research, and the Wikipedia contributors behind WikiText. Cite
[Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843) for WikiText.
The pinned dataset card's metadata lists CC BY-SA 3.0 and GFDL, while its prose
licensing section links CC BY-SA 4.0. Consult the original card and source
article terms when redistributing dataset material; project code licensing does
not replace those terms.

Downloaded weights, tokenizer assets, corpus text, token binaries, and model
checkpoints are not redistributed here. The repository preserves measurements,
source revisions, and split/hash metadata in `results/sliding_scratch/`.
