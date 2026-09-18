# Exploratory prototypes

`prototypes/merged_transformer.py` is a non-causal classification prototype.
`prototypes/test_generative_merged.py` is an exploratory LM with future-token
leakage through unmerging. Neither is part of the correctness suite or current
quality evidence. They are retained to record provenance rather than silently
removing previous investigations.

Use `experiments/smollm/train_sliding.py` and `tests/` for current experiments.
