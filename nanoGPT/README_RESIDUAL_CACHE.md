# Causal Token Merging with Residual Caching

The maintained project documentation is at [../README.md](../README.md), with a [Russian overview](../README.ru.md) and [architecture details](../docs/architecture.md).

The current implementation uses one-way window compression, direct boundary predictions, consistent singleton residuals, and training on both merged and singleton-only contexts. It does not unmerge or implement a persistent KV cache. Performance and quality benefits remain research hypotheses pending controlled measurements.

Use `python train.py config/train_residual_shakespeare.py` after preparing the character-level Shakespeare dataset. Run regression tests from the project root with `python -m unittest discover -s tests -v`.

Built on nanoGPT by Andrej Karpathy. Project direction by geomi737; development assisted by Gemini and ChatGPT, including Codex.
