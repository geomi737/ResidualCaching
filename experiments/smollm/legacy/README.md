# Legacy SmolLM experiments

These files preserve previous investigations. The active scratch runner is
[`../train_sliding.py`](../train_sliding.py).

- `pretrain_tiny.py`: eight-layer random-initialized model with disjoint merging.
- `finetune_smollm.py`: pretrained SmolLM2 adaptation, not scratch training.
- `pretrain_tiny_v2.py`, `smollm_model_v2.py`: full-length unmerge prototype.
  Its dense training loss leaks the second constituent into the first target.
  Do not use that loss as causal LM quality evidence.
- `memory_test.py`: dummy linear-layer/LM-head memory probe, not a full
  Transformer comparison. Live references span cases, so results are diagnostic.
- `chat_tiny.py`: generation utility for the older baseline/plain/residual
  checkpoints, not the current sliding checkpoint format.

Files were archived, not promoted into the current protocol. Earlier CLI paths
have changed. If reproducing a legacy script directly, run from the repository
root and use its new path under `experiments/smollm/legacy/`.
