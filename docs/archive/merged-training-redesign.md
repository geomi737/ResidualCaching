# ADR: Train usable compressed context without unmerging

Status: Archived proposal, not implemented or experimentally validated.
The active protocol is [sliding training from scratch](../sliding-merging-experiment.md), without a teacher.
Date: 2026-09-18.

## Context

The objective is to preserve information useful for future predictions and
queries in fewer deep-layer positions. Restoring the original sequence length
is not an objective. A compressed vector represents an already-read window,
not either individual constituent token.

`smollm_model.py` already implements causal boundary prediction without unmerge.
`experiments/smollm/legacy/smollm_model_v2.py` restores every position and trains on shifted targets at
every position. Its first prediction in a completed pair can see its own target
through the pair aggregate. This objective must not be used for causal quality
comparisons. Its full-length vocabulary projection also removes output-memory
savings.

Both tiny pretraining runners currently replace the loaded pretrained model
with a randomly initialized eight-layer model. These are from-scratch runs,
not adaptation of pretrained SmolLM2. Loading the merged wrapper also consumes
random numbers differently when it initializes extra modules. Paired runs must
explicitly reuse the same base initialization and verify base weight hashes.

## Decision

Keep compression one-way. Train the student to predict continuations and answer
queries using compressed history. Use a conventional model as a training-only
teacher, when practical, to guide adaptation. No teacher or reconstruction head
is required at inference.

### Causal language objective

For a completed window ending at original position e, predict token e+1.
Compare teacher and student at that same original endpoint:

    L_language = CE(student_logits[e], token[e+1])
    L_distill = tau^2 * KL(teacher_probs[e] || student_probs[e])

Teacher probabilities and student probabilities use the same temperature tau.
Stop gradients through the teacher. Also supervise untouched incomplete-tail
positions. Do not predict targets inside a completed window from its merged
representation. Boundary loss is a valid starting point, but supplies fewer
predictions than conventional training and is not itself proof of semantic
preservation.

### Delayed information-use objective

Mix natural continuations with controlled examples of this form:

    facts / relationships -> distractor text -> query -> answer

Examples should test names, numbers, ordered relations, negation, and facts that
require combining two statements. Put the tested facts in completed compressed
windows. Change query distance, window alignment, and distractor length. Split
entities, values, and templates across training and evaluation to measure
generalization rather than memorization.

    L_total = L_language + lambda_KD * L_distill + lambda_Q * L_answer

Coefficients are validation-selected experiment settings, not established values.
Answer loss supervises actual answer tokens. For the existing uniformly merged
model, obtain each answer prediction from its exact causal prefix, with both
complete-window and incomplete-tail prefixes represented in training. Batch or
sample those prefixes; never unmerge earlier positions for dense answer loss.
This costs additional training forwards. A later alternative is compressed
history plus an uncompressed recent query/answer suffix, but that changes the
architecture and requires new masking and cache tests.

The answer objective forces the model to use information after compression.
It does not guarantee arbitrary lossless reconstruction or preservation of
facts outside the training distribution. Ordinary continuations should remain
part of training to avoid overfitting to synthetic retrieval tasks.

### Adaptation schedule

1. Retain a shared pretrained base for teacher and student. Establish teacher
   quality and student quality before adaptation. Keep R=2 and start at a late
   merge boundary to limit the initial distribution shift.
2. Train merge parameters and deep-block adapters first, with early blocks
   frozen. If using the current two-weight aggregate, recognize its limited
   capacity; adapting deep blocks is essential to this first trial.
3. Unfreeze more blocks only if validation shows a benefit. Introduce earlier
   boundaries as separate trials after the late-boundary model works.
4. Select checkpoints on held-out causal continuation and information-use
   metrics, with both prefix parities. Test only after selection.

Do not use uncompressed bypass batches as the student adaptation path: the
student must actually encounter compressed context during adaptation.

## Options and trade-offs

Boundary CE alone is simple and cheap but gives sparse supervision of retained
information. Adding teacher distillation provides distributional guidance;
delayed queries test whether compressed history remains usable. Neither can
recover information that the merge operator has already irreversibly discarded.

The current residual formula is still a positive weighted linear aggregate.
At uniform initialization it equals three times the pair mean; the residual
sum does not store an independent copy of the constituents. If training changes
do not close the quality gap, test a separate content-dependent, order-aware
compressor with output width D. An endpoint-anchored learned correction is one
candidate because the final causal hidden state already observes the window.
Treat that as a separately named architecture experiment, not a silent change
to the existing residual formula.

Teacher training increases training compute and potentially peak VRAM. Use a
frozen teacher sequentially or precompute aligned targets if necessary; include
teacher overhead in training-cost reports. Teacher-free inference is the actual
deployment path. Distillation does not imply cheaper training.

## Validation and action items

- Implement a new adaptation runner; keep historical tiny results separate.
- Verify shared initial base hashes, identical document splits and data plans.
- Test target alignment, future-token independence, and cached/full-prefix
  agreement. Evaluate every answer token from its causal prefix.
- Compare baseline, boundary-only student, student plus distillation, and student
  plus distillation and delayed queries. Hold compression architecture fixed.
- Report matched endpoint and final-prefix loss, answer accuracy by distance and
  parity, original input tokens and supervised predictions, wall time, and
  inference allocated/reserved memory separately from training memory.
- Report results at equal input budgets and account for different supervised
  prediction counts. Include teacher and query-prefix computation in costs.
- Only then test a richer compressor if the remaining gap justifies it.

Success means retaining useful continuation and query behavior at a measured
inference cost reduction. Lower training loss alone is insufficient.
