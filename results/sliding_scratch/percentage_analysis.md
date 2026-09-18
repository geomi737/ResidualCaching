# Percentage analysis: training from scratch

All changes are relative to the baseline: `100 × (variant / baseline − 1)`. Accuracy includes percentage-point differences. A positive loss change is worse; a positive throughput change is faster. One seed, equal input budgets.

| Metric | Baseline | Merge | Sliding | Merge + R | Sliding + R |
|---|---:|---:|---:|---:|---:|
| Test boundary loss | 5.27429 | 5.46725 (+3.66%) | 5.37581 (+1.92%) | 5.46972 (+3.71%) | 5.35726 (+1.57%) |
| Test boundary accuracy (%) | 24.7086 | 22.9904 (-6.95%); -1.72 pp | 23.3292 (-5.58%); -1.38 pp | 23.0698 (-6.63%); -1.64 pp | 23.7137 (-4.03%); -0.99 pp |
| Test final-prefix loss | 5.20006 | 5.5434 (+6.60%) | 5.38945 (+3.64%) | 5.57912 (+7.29%) | 5.54073 (+6.55%) |
| Test final-prefix accuracy (%) | 24.4141 | 22.6562 (-7.20%); -1.76 pp | 22.6562 (-7.20%); -1.76 pp | 22.2656 (-8.80%); -2.15 pp | 22.4609 (-8.00%); -1.95 pp |
| Even-prefix loss | 5.3521 | 5.52904 (+3.31%) | 5.45725 (+1.96%) | 5.54568 (+3.62%) | 5.42384 (+1.34%) |
| Odd-prefix loss | 5.04802 | 5.55777 (+10.10%) | 5.32166 (+5.42%) | 5.61256 (+11.18%) | 5.65762 (+12.08%) |
| Even-prefix accuracy (%) | 24.2188 | 24.6094 (+1.61%); +0.39 pp | 22.6562 (-6.45%); -1.56 pp | 24.6094 (+1.61%); +0.39 pp | 24.6094 (+1.61%); +0.39 pp |
| Odd-prefix accuracy (%) | 24.6094 | 20.7031 (-15.87%); -3.91 pp | 22.6562 (-7.94%); -1.95 pp | 19.9219 (-19.05%); -4.69 pp | 20.3125 (-17.46%); -4.30 pp |
| Last training loss | 4.7948 | 5.12683 (+6.92%) | 4.83121 (+0.76%) | 5.13986 (+7.20%) | 4.85173 (+1.19%) |
| Last training accuracy (%) | 27.4902 | 25 (-9.06%); -2.49 pp | 26.8066 (-2.49%); -0.68 pp | 24.2676 (-11.72%); -3.22 pp | 26.7334 (-2.75%); -0.76 pp |
| Last gradient norm before clipping | 0.835909 | 0.886918 (+6.10%) | 0.755334 (-9.64%) | 0.896532 (+7.25%) | 0.78103 (-6.57%) |
| Final learning rate | 5e-05 | 5e-05 (+0.00%) | 5e-05 (+0.00%) | 5e-05 (+0.00%) | 5e-05 (+0.00%) |
| Training input tokens/s | 35984.3 | 41744.6 (+16.01%) | 36256.6 (+0.76%) | 41724.5 (+15.95%) | 35652.1 (-0.92%) |
| Training predictions/s | 35984.3 | 20913.1 (-41.88%) | 36256.6 (+0.76%) | 20903 (-41.91%) | 35652.1 (-0.92%) |
| Training update time (ms) | 113.605 | 97.9288 (-13.80%) | 112.752 (-0.75%) | 97.9762 (-13.76%) | 114.664 (+0.93%) |
| Training update total time (s) | 170.542 | 146.934 (-13.84%) | 169.13 (-0.83%) | 146.966 (-13.82%) | 171.995 (+0.85%) |
| Training allocated peak (MiB) | 1197.13 | 828.592 (-30.79%) | 1197.26 (+0.01%) | 828.592 (-30.79%) | 1197.26 (+0.01%) |
| Training reserved peak (MiB) | 1396 | 1092 (-21.78%) | 1272 (-8.88%) | 1092 (-21.78%) | 1272 (-8.88%) |
| Input tokens | 6.132e+06 | 6.132e+06 (+0.00%) | 6.132e+06 (+0.00%) | 6.132e+06 (+0.00%) | 6.132e+06 (+0.00%) |
| Supervised predictions | 6.132e+06 | 3.072e+06 (-49.90%) | 6.132e+06 (+0.00%) | 3.072e+06 (-49.90%) | 6.132e+06 (+0.00%) |
| Parameters | 2.21233e+07 | 2.21233e+07 (+0.00%) | 2.21233e+07 (+0.00%) | 2.21233e+07 (+0.00%) | 2.21233e+07 (+0.00%) |
| Sliding-mode boundary loss | 5.27429 | 5.47189 (+3.75%) | 5.27949 (+0.10%) | 5.4811 (+3.92%) | 5.30207 (+0.53%) |
| Sliding-mode boundary accuracy (%) | 24.7086 | 22.9691 (-7.04%); -1.74 pp | 24.4049 (-1.23%); -0.30 pp | 23.0164 (-6.85%); -1.69 pp | 24.3484 (-1.46%); -0.36 pp |
| sliding_full_forward: seconds | 0.00612971 | 0.00612142 (-0.14%) | 0.00582457 (-4.98%) | 0.00570756 (-6.89%) | 0.00602659 (-1.68%) |
| sliding_full_forward: peak_allocated_mib | 167.266 | 167.266 (-0.00%) | 167.267 (+0.00%) | 167.266 (-0.00%) | 167.267 (+0.00%) |
| sliding_full_forward: peak_reserved_mib | 222 | 192 (-13.51%) | 222 (+0.00%) | 192 (-13.51%) | 222 (+0.00%) |
| sliding_full_forward: input_tokens_per_second | 41763.8 | 41820.4 (+0.14%) | 43951.7 (+5.24%) | 44852.8 (+7.40%) | 42478.4 (+1.71%) |
| disjoint_full_forward: seconds | 0.00600028 | 0.00603755 (+0.62%) | 0.00589179 (-1.81%) | 0.00585492 (-2.42%) | 0.00587408 (-2.10%) |
| disjoint_full_forward: peak_allocated_mib | 167.266 | 155.054 (-7.30%) | 155.055 (-7.30%) | 155.054 (-7.30%) | 155.055 (-7.30%) |
| disjoint_full_forward: peak_reserved_mib | 222 | 180 (-18.92%) | 222 (+0.00%) | 182 (-18.02%) | 224 (+0.90%) |
| disjoint_full_forward: input_tokens_per_second | 42664.7 | 42401.3 (-0.62%) | 43450.3 (+1.84%) | 43723.9 (+2.48%) | 43581.3 (+2.15%) |
| cached_prefill: seconds | 0.0122427 | 0.00806305 (-34.14%) | 0.00818055 (-33.18%) | 0.00772594 (-36.89%) | 0.00789746 (-35.49%) |
| cached_prefill: peak_allocated_mib | 146.22 | 145.328 (-0.61%) | 145.329 (-0.61%) | 145.328 (-0.61%) | 145.329 (-0.61%) |
| cached_prefill: peak_reserved_mib | 224 | 168 (-25.00%) | 222 (-0.89%) | 168 (-25.00%) | 222 (-0.89%) |
| cached_decode: seconds | 0.232469 | 0.181859 (-21.77%) | 0.17762 (-23.59%) | 0.176414 (-24.11%) | 0.174495 (-24.94%) |
| cached_decode: peak_allocated_mib | 146.535 | 146.419 (-0.08%) | 146.42 (-0.08%) | 146.419 (-0.08%) | 146.42 (-0.08%) |
| cached_decode: peak_reserved_mib | 224 | 170 (-24.11%) | 224 (+0.00%) | 170 (-24.11%) | 224 (+0.00%) |
| cached_decode: tokens_per_second | 137.653 | 175.96 (+27.83%) | 180.159 (+30.88%) | 181.392 (+31.77%) | 183.386 (+33.22%) |
| cached_decode: kv_cache_mib | 3.375 | 2.53125 (-25.00%) | 2.53125 (-25.00%) | 2.53125 (-25.00%) | 2.53125 (-25.00%) |

## Representation-mode mismatch and merge weights

| Variant | Boundary loss change: sliding → disjoint | Final argmax agreement | Final KL(sliding ∥ disjoint) | Learned weights: previous / current | Normalized effective coefficients |
|---|---:|---:|---:|---|---|
| baseline | +0.00% | 100.00% | 0.0000 | n/a | n/a |
| disjoint | -0.08% | 82.03% | 0.1202 | 35.47% / 64.53% | 35.47% / 64.53% |
| sliding | +1.82% | 73.44% | 0.1989 | 35.41% / 64.59% | 35.41% / 64.59% |
| disjoint_residual | -0.21% | 78.52% | 0.1216 | 32.23% / 67.77% | 44.08% / 55.92% |
| sliding_residual | +1.04% | 72.27% | 0.2832 | 32.43% / 67.57% | 44.14% / 55.86% |

## Interpretation limits

- Boundary scores use 65,536 matched targets; final-prefix scores use 512 matched prefixes. Windows can overlap.
- Training loss target sets differ. Disjoint training supervises about half as many predictions per input budget.
- Last-step training scores come from one batch. They are not held-out generalization metrics.
- Update timings exclude validation. Inference timings are short sequential measurements, not independent timing repetitions.
- Reserved memory depends on allocator history. Allocated peaks and actual KV bytes are the primary memory comparisons.
- Percentage changes in cross-entropy are not percentages of lost understanding. Accuracy is top-1 next-token accuracy.
- Tiny negative baseline KL from floating-point rounding is displayed as zero. Raw evidence is unchanged.
- One seed is insufficient to establish a universal winner or rank small speed differences.
