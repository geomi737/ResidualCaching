# Causal Token Merging with Residual Caching

The maintained project documentation is at [../README.md](../README.md), with [architecture details](../docs/architecture.md).

The current implementation uses one-way window compression, direct boundary predictions, unchanged incomplete tails, and mandatory compression of every complete window. It does not unmerge. nanoGPT has no persistent KV cache; the separate SmolLM2 adapter does. The older GPU report records the retired mixed-batch protocol. See the [actual GPU experiment report](../docs/experiments.md) for measured quality, throughput, and memory tradeoffs, and [proofs](../docs/proofs.md) for algebraic properties and runtime tracing.

Use `python train.py config/train_residual_shakespeare.py` after preparing the character-level Shakespeare dataset. Run regression tests from the project root with `python -m unittest discover -s tests -v`.

Built on nanoGPT by Andrej Karpathy. Project direction by geomi737; development assisted by Gemini and ChatGPT, including Codex.
