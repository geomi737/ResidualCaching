# Three-seed sliding experiment

Nine fresh random-initialized GPU trainings; 1500 updates each. Initial base weights and training batches are paired within each seed. Values are mean ± sample standard deviation across seeds, not confidence intervals. Evaluation uses the same exploratory test split as the pilot; this is not fresh held-out confirmation.

| Metric | Baseline | Sliding | Sliding + R |
|---|---:|---:|---:|
| Test boundary loss | 5.2562 ± 0.0104 | 5.4022 ± 0.0077 | 5.3751 ± 0.0204 |
| Test boundary accuracy (%) | 24.7381 ± 0.1177 | 23.0693 ± 0.0527 | 23.4049 ± 0.2966 |
| Test final-prefix loss | 5.2241 ± 0.0309 | 5.4127 ± 0.0560 | 5.5419 ± 0.0527 |
| Test final-prefix accuracy (%) | 24.9349 ± 0.5638 | 22.7865 ± 0.7893 | 21.9401 ± 2.4808 |
| Even-prefix loss | 5.3432 ± 0.0309 | 5.4495 ± 0.0596 | 5.4440 ± 0.0171 |
| Odd-prefix loss | 5.1051 ± 0.0312 | 5.3758 ± 0.0829 | 5.6399 ± 0.0883 |
| Even-prefix accuracy (%) | 24.8698 ± 0.8132 | 23.4375 ± 1.0335 | 23.5677 ± 3.1574 |
| Odd-prefix accuracy (%) | 25.0000 ± 0.3906 | 22.1354 ± 2.1514 | 20.3125 ± 2.3438 |
| Last training loss | 4.7192 ± 0.1300 | 4.7651 ± 0.1242 | 4.7744 ± 0.1222 |
| Last training accuracy (%) | 28.0843 ± 0.9770 | 27.9460 ± 1.3822 | 27.7669 ± 0.7570 |
| Last gradient norm before clipping | 0.8004 ± 0.0076 | 0.7714 ± 0.0440 | 0.7917 ± 0.0494 |
| Final learning rate | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 |
| Training input tokens/s | 35746.2013 ± 904.2695 | 35456.2903 ± 196.9842 | 35686.7593 ± 534.8788 |
| Training predictions/s | 35746.2013 ± 904.2695 | 35456.2903 ± 196.9842 | 35686.7593 ± 534.8788 |
| Training update time (ms) | 114.4115 ± 2.9372 | 115.2994 ± 0.6419 | 114.5697 ± 1.7269 |
| Training update total time (s) | 171.6694 ± 4.4833 | 172.9982 ± 0.9438 | 171.8865 ± 2.5478 |
| Training allocated peak (MiB) | 1196.8363 ± 0.6126 | 1197.9637 ± 0.6162 | 1197.4637 ± 0.3570 |
| Training reserved peak (MiB) | 1313.3333 ± 71.5914 | 1313.3333 ± 71.5914 | 1313.3333 ± 71.5914 |
| Input tokens | 6132000.0000 ± 0.0000 | 6132000.0000 ± 0.0000 | 6132000.0000 ± 0.0000 |
| Supervised predictions | 6132000.0000 ± 0.0000 | 6132000.0000 ± 0.0000 | 6132000.0000 ± 0.0000 |
| Parameters | 22123296.0000 ± 0.0000 | 22123298.0000 ± 0.0000 | 22123298.0000 ± 0.0000 |
| Sliding-mode boundary loss | 5.2562 ± 0.0104 | 5.2765 ± 0.0075 | 5.2905 ± 0.0074 |
| Sliding-mode boundary accuracy (%) | 24.7381 ± 0.1177 | 24.4395 ± 0.1101 | 24.2442 ± 0.0661 |
| sliding_full_forward: seconds | 0.0060 ± 0.0002 | 0.0059 ± 0.0000 | 0.0061 ± 0.0003 |
| sliding_full_forward: peak_allocated_mib | 167.2661 ± 0.0000 | 167.2666 ± 0.0000 | 167.2666 ± 0.0000 |
| sliding_full_forward: peak_reserved_mib | 212.0000 ± 17.3205 | 212.0000 ± 17.3205 | 212.0000 ± 17.3205 |
| sliding_full_forward: input_tokens_per_second | 42881.7050 ± 1393.0257 | 43749.2197 ± 347.8509 | 42172.5934 ± 2251.4082 |
| disjoint_full_forward: seconds | 0.0059 ± 0.0003 | 0.0060 ± 0.0001 | 0.0062 ± 0.0003 |
| disjoint_full_forward: peak_allocated_mib | 167.2661 ± 0.0000 | 155.0547 ± 0.0000 | 155.0547 ± 0.0000 |
| disjoint_full_forward: peak_reserved_mib | 212.0000 ± 17.3205 | 208.0000 ± 24.2487 | 208.0000 ± 24.2487 |
| disjoint_full_forward: input_tokens_per_second | 43262.3605 ± 1901.0812 | 42851.1129 ± 509.5996 | 41350.2170 ± 2150.9169 |
| cached_prefill: seconds | 0.0092 ± 0.0022 | 0.0092 ± 0.0012 | 0.0083 ± 0.0002 |
| cached_prefill: peak_allocated_mib | 146.2202 ± 0.0000 | 145.3291 ± 0.0000 | 145.3291 ± 0.0000 |
| cached_prefill: peak_reserved_mib | 206.6667 ± 31.7700 | 204.0000 ± 31.1769 | 204.0000 ± 31.1769 |
| cached_decode: seconds | 0.2036 ± 0.0516 | 0.2033 ± 0.0281 | 0.1944 ± 0.0338 |
| cached_decode: peak_allocated_mib | 146.5352 ± 0.0000 | 146.4204 ± 0.0000 | 146.4204 ± 0.0000 |
| cached_decode: peak_reserved_mib | 206.6667 ± 30.0222 | 206.0000 ± 31.1769 | 206.0000 ± 31.1769 |
| cached_decode: tokens_per_second | 163.2611 ± 36.0782 | 159.3585 ± 21.2736 | 167.6797 ± 26.4808 |
| cached_decode: kv_cache_mib | 3.3750 ± 0.0000 | 2.5312 ± 0.0000 | 2.5312 ± 0.0000 |

## Paired differences from baseline

Percentages are computed within each seed before aggregation. Accuracy differences use percentage points.

| Metric | Sliding | Sliding + R |
|---|---:|---:|
| Test boundary loss | +2.778 ± 0.193 % | +2.263 ± 0.214 % |
| Test boundary accuracy (%) | -1.669 ± 0.157 pp | -1.333 ± 0.223 pp |
| Test final-prefix loss | +3.607 ± 0.460 % | +6.086 ± 1.203 % |
| Test final-prefix accuracy (%) | -2.148 ± 0.704 pp | -2.995 ± 1.917 pp |
| Even-prefix loss | +1.989 ± 0.782 % | +1.889 ± 0.699 % |
| Odd-prefix loss | +5.300 ± 1.080 % | +10.478 ± 1.818 % |
| Even-prefix accuracy (%) | -1.432 ± 0.597 pp | -1.302 ± 2.355 pp |
| Odd-prefix accuracy (%) | -2.865 ± 1.966 pp | -4.688 ± 1.953 pp |
| Last training loss | +0.975 ± 0.253 % | +1.173 ± 0.420 % |
| Last training accuracy (%) | -0.138 ± 0.731 pp | -0.317 ± 0.224 pp |
| Last gradient norm before clipping | -3.613 ± 5.563 % | -1.072 ± 6.409 % |
| Final learning rate | +0.000 ± 0.000 % | +0.000 ± 0.000 % |
| Training input tokens/s | -0.761 ± 2.980 % | -0.117 ± 3.245 % |
| Training predictions/s | -0.761 ± 2.980 % | -0.117 ± 3.245 % |
| Training update time (ms) | +0.827 ± 2.980 % | +0.188 ± 3.244 % |
| Training update total time (s) | +0.827 ± 3.034 % | +0.178 ± 3.282 % |
| Training allocated peak (MiB) | +0.094 ± 0.084 % | +0.052 ± 0.042 % |
| Training reserved peak (MiB) | +0.289 ± 9.319 % | +0.289 ± 9.319 % |
| Input tokens | +0.000 ± 0.000 % | +0.000 ± 0.000 % |
| Supervised predictions | +0.000 ± 0.000 % | +0.000 ± 0.000 % |
| Parameters | +0.000 ± 0.000 % | +0.000 ± 0.000 % |
| Sliding-mode boundary loss | +0.387 ± 0.176 % | +0.653 ± 0.133 % |
| Sliding-mode boundary accuracy (%) | -0.299 ± 0.061 pp | -0.494 ± 0.073 pp |
| sliding_full_forward: seconds | -1.965 ± 3.839 % | +1.880 ± 6.451 % |
| sliding_full_forward: peak_allocated_mib | +0.000 ± 0.000 % | +0.000 ± 0.000 % |
| sliding_full_forward: peak_reserved_mib | +0.704 ± 14.582 % | +0.704 ± 14.582 % |
| sliding_full_forward: input_tokens_per_second | +2.109 ± 3.983 % | -1.584 ± 6.192 % |
| disjoint_full_forward: seconds | +0.945 ± 3.734 % | +4.657 ± 1.163 % |
| disjoint_full_forward: peak_allocated_mib | -7.301 ± 0.000 % | -7.301 ± 0.000 % |
| disjoint_full_forward: peak_reserved_mib | -1.098 ± 17.298 % | -1.098 ± 17.298 % |
| disjoint_full_forward: input_tokens_per_second | -0.844 ± 3.744 % | -4.441 ± 1.065 % |
| cached_prefill: seconds | +4.985 ± 32.403 % | -7.106 ± 20.620 % |
| cached_prefill: peak_allocated_mib | -0.609 ± 0.000 % | -0.609 ± 0.000 % |
| cached_prefill: peak_reserved_mib | +1.273 ± 27.919 % | +1.344 ± 28.193 % |
| cached_decode: seconds | +4.119 ± 29.843 % | +0.455 ± 33.930 % |
| cached_decode: peak_allocated_mib | -0.078 ± 0.000 % | -0.078 ± 0.000 % |
| cached_decode: peak_reserved_mib | +2.042 ± 27.227 % | +2.042 ± 27.227 % |
| cached_decode: tokens_per_second | +1.664 ± 29.822 % | +8.119 ± 38.783 % |
| cached_decode: kv_cache_mib | -25.000 ± 0.000 % | -25.000 ± 0.000 % |

## Sliding + R compared directly with Sliding

| Metric | Paired difference, mean ± sample SD |
|---|---:|
| Test boundary loss | -0.501 ± 0.396 % |
| Test boundary accuracy (%) | +0.336 ± 0.349 pp |
| Test final-prefix loss | +2.396 ± 1.470 % |
| Test final-prefix accuracy (%) | -0.846 ± 2.195 pp |
| Even-prefix loss | -0.092 ± 1.334 % |
| Odd-prefix loss | +4.921 ± 1.679 % |
| Even-prefix accuracy (%) | +0.130 ± 2.289 pp |
| Odd-prefix accuracy (%) | -1.823 ± 2.151 pp |
| Last training loss | +0.195 ± 0.169 % |
| Last training accuracy (%) | -0.179 ± 0.855 pp |
| Last gradient norm before clipping | +2.611 ± 1.197 % |
| Final learning rate | +0.000 ± 0.000 % |
| Training input tokens/s | +0.655 ± 1.856 % |
| Training predictions/s | +0.655 ± 1.856 % |
| Training update time (ms) | -0.628 ± 1.831 % |
| Training update total time (s) | -0.638 ± 1.766 % |
| Training allocated peak (MiB) | -0.042 ± 0.042 % |
| Training reserved peak (MiB) | +0.289 ± 9.319 % |
| Input tokens | +0.000 ± 0.000 % |
| Supervised predictions | +0.000 ± 0.000 % |
| Parameters | +0.000 ± 0.000 % |
| Sliding-mode boundary loss | +0.264 ± 0.043 % |
| Sliding-mode boundary accuracy (%) | -0.195 ± 0.102 pp |
| sliding_full_forward: seconds | +3.961 ± 6.281 % |
| sliding_full_forward: peak_allocated_mib | +0.000 ± 0.000 % |
| sliding_full_forward: peak_reserved_mib | +0.704 ± 14.582 % |
| sliding_full_forward: input_tokens_per_second | -3.582 ± 5.656 % |
| disjoint_full_forward: seconds | +3.798 ± 4.944 % |
| disjoint_full_forward: peak_allocated_mib | +0.000 ± 0.000 % |
| disjoint_full_forward: peak_reserved_mib | +1.471 ± 21.165 % |
| disjoint_full_forward: input_tokens_per_second | -3.516 ± 4.500 % |
| cached_prefill: seconds | -9.545 ± 9.651 % |
| cached_prefill: peak_allocated_mib | +0.000 ± 0.000 % |
| cached_prefill: peak_reserved_mib | +2.606 ± 28.324 % |
| cached_decode: seconds | -1.995 ± 29.045 % |
| cached_decode: peak_allocated_mib | +0.000 ± 0.000 % |
| cached_decode: peak_reserved_mib | +2.553 ± 28.023 % |
| cached_decode: tokens_per_second | +7.699 ± 28.904 % |
| cached_decode: kv_cache_mib | +0.000 ± 0.000 % |
