| Config | Role | boosting rounds per fold | Train loss | Val loss | Val NLL (raw / T-scaled) | OOF F1 raw (fold mean±std) | OOF F1 nested-post (fold mean±std) | Pooled OOF F1 nested-post | OOF F1 cross-fit post (fold mean±std) / pooled | Per-class F1 nested-post [0,1,2,3] | Dev F1 raw / post |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G0_builtin_multiclass_sqrtbal | repo v18 recipe: built-in multiclass + SqrtBalanced | 107, 60, 128, 74, 74 | 0.2042 ± 0.0545 | 0.5069 ± 0.0144 | 0.3381 / 0.3201 | 0.7307 ± 0.0182 | 0.7328 ± 0.0160 | 0.7338 | 0.7422 ± 0.0193 / 0.7440 | 0.914, 0.627, 0.524, 0.869 | 0.6952 / 0.6943 |
| G0LL_builtin_sqrtbal_loglossES | robustness: G0 early-stopped on multi_logloss instead of macro-F1 | 188, 182, 183, 193, 184 | 0.0746 ± 0.0032 | 0.5255 ± 0.0271 | 0.3040 / 0.3046 | 0.7315 ± 0.0255 | 0.7418 ± 0.0195 | 0.7433 | 0.7479 ± 0.0183 / 0.7490 | 0.921, 0.634, 0.543, 0.876 | 0.6736 / 0.6973 |
| G3_logit_adjusted | candidate: logit-adjusted CE via training init_score | 480, 442, 412, 524, 440 | 0.0137 ± 0.0021 | 0.3444 ± 0.0205 | 0.3490 / 0.3487 | 0.7360 ± 0.0180 | 0.7441 ± 0.0236 | 0.7463 | 0.7405 ± 0.0201 / 0.7417 | 0.923, 0.638, 0.548, 0.877 | 0.6760 / 0.7046 |

Paired comparison — nested offsets (primary, strict):

| Config | Δ vs G0 per fold | folds > G0 | bootstrap Δ mean [95% CI], P(Δ>0) | Δ vs G0LL per fold | folds > G0LL | bootstrap Δ mean [95% CI], P(Δ>0) |
|---|---|---|---|---|---|---|
| G0_builtin_multiclass_sqrtbal | — | — | — | +0.0072, -0.0267, -0.0017, -0.0024, -0.0215 | 1/5 | -0.0093 [-0.0170, -0.0017], 0.01 |
| G0LL_builtin_sqrtbal_loglossES | -0.0072, +0.0267, +0.0017, +0.0024, +0.0215 | 4/5 | +0.0093 [+0.0017, +0.0170], 0.99 | — | — | — |
| G3_logit_adjusted | -0.0154, +0.0287, -0.0048, +0.0167, +0.0312 | 3/5 | +0.0123 [+0.0037, +0.0215], 1.00 | -0.0082, +0.0020, -0.0065, +0.0143, +0.0097 | 3/5 | +0.0030 [-0.0049, +0.0109], 0.76 |

Paired comparison — cross-fit offsets (scale-matched):

| Config | Δ vs G0 per fold | folds > G0 | bootstrap Δ mean [95% CI], P(Δ>0) | Δ vs G0LL per fold | folds > G0LL | bootstrap Δ mean [95% CI], P(Δ>0) |
|---|---|---|---|---|---|---|
| G0_builtin_multiclass_sqrtbal | — | — | — | -0.0066, -0.0073, -0.0019, -0.0077, -0.0049 | 0/5 | -0.0050 [-0.0113, +0.0012], 0.06 |
| G0LL_builtin_sqrtbal_loglossES | +0.0066, +0.0073, +0.0019, +0.0077, +0.0049 | 5/5 | +0.0050 [-0.0012, +0.0113], 0.94 | — | — | — |
| G3_logit_adjusted | -0.0016, -0.0091, -0.0006, +0.0103, -0.0075 | 1/5 | -0.0024 [-0.0119, +0.0068], 0.31 | -0.0082, -0.0164, -0.0024, +0.0026, -0.0124 | 1/5 | -0.0074 [-0.0164, +0.0012], 0.05 |
