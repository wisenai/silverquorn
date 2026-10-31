# MHSC elder-persona validation

Can an LLM elder persona be scored against real humans of *known exact age*? The Measuring Hate Speech Corpus is the one place this is even possible: `annotator_age` is an exact integer, 18-81. This file reports the attempt.

## 0. Setup

- Dataset: `ucberkeley-dlab/measuring-hate-speech` (135,556 rows, 39,565 comments, 7,912 annotators)
- Target variable: `respect`, reverse-coded `respectfulness = 5 - respect` (integer 1-5, 5 = most respectful)
- Models: openai/gpt-4o, anthropic/claude-sonnet-4.5, meta-llama/llama-3.1-70b-instruct, qwen/qwen-2.5-72b-instruct, google/gemini-2.5-flash, mistralai/mistral-large-2512 (same six as `config.json`)
- Temperature 0.7, 3 samples/cell, single-integer parse with one strict retry
- Rating instruction, verbatim:

  > Rate how respectful the following social media comment is toward the people or groups it refers to, from 1 (very disrespectful) to 5 (very respectful). Reply with only the number.

- Persona system message, verbatim (one example): `You are a 78-year-old adult.`

## 1. Feasibility

`annotator_age` is exact and integer-valued over [18, 81]. Annotator counts by threshold:

| threshold | distinct annotators |
|---|---|
| 60+ | 436 |
| 65+ | 203 |
| 70+ | 69 |
| 75+ | 12 |
| 80+ | 4 |

Single-year counts, 60-81 (this is the cliff):

| age | n | age | n |
|---|---|---|---|
| 60 | 56 | 71 | 11 |
| 61 | 42 | 72 | 9 |
| 62 | 55 | 73 | 8 |
| 63 | 43 | 74 | 5 |
| 64 | 37 | 75 | 2 |
| 65 | 25 | 76 | 4 |
| 66 | 31 | 77 | 0 |
| 67 | 27 | 78 | 1 |
| 68 | 26 | 79 | 1 |
| 69 | 25 | 80 | 3 |
| 70 | 24 | 81 | 1 |

### Per-comment elder coverage, whole corpus

| raters in band >= k | 65+ | 70+ | 75+ |
|---|---|---|---|
| 1 | 2834 | 1045 | 196 |
| 2 | 111 | 68 | 22 |
| 3 | 70 | 63 | 3 |
| 5 | 69 | 43 | - |
| 10 | 59 | - | - |

The distribution is bimodal, and that is the whole story. Number of comments by exact count of 65+ raters: 0->36717, 1->2723, 2->41, 3->1, 5->1, 6->1, ...

All elder replication lives in a high-replication core of **70 comments** with 243-815 raters each. Outside that core, only 41 comments have as many as 2 raters aged 65+. All 203 annotators aged 65+ rated ~6 of these 70 comments, so the core holds 100% of the available elder signal and n=70 comments is a hard ceiling.

### Band coverage inside the core

| band | annotators (corpus) | ratings in core | median raters/comment | comments >=1 | >=2 | >=3 |
|---|---|---|---|---|---|---|
| 18-29 | 2172 | 11739 | 188 | 70 | 70 | 70 |
| 45-54 | 1084 | 5963 | 94 | 70 | 70 | 70 |
| 55-64 | 619 | 3443 | 54 | 70 | 70 | 70 |
| 65-69 | 134 | 753 | 10 | 69 | 69 | 69 |
| 70-74 | 57 | 325 | 4 | 70 | 67 | 57 |
| 75-81 | 12 | 68 | 1 | 42 | 22 | 3 |

Direction of `respect` verified empirically: comment-level correlation with `hate_speech_score` is +0.969; the least hateful comment has mean `respect` 0.151 and the most hateful 3.976. Higher value = more disrespectful, hence the reverse coding.

## 2. Analysis

### (c) Power: how noisy are the human band means?

Reliability = (var of band means across comments - mean sampling error variance) / var of band means. Its square root is the ceiling on any correlation a persona could achieve against that target.

| band | comments n>=2 | median raters* | mean raters* | median 95% CI half-width | mean 95% CI half-width | reliability | max attainable r |
|---|---|---|---|---|---|---|---|
| 18-29 | 70 | 188 | 167.7 | 0.0709 | 0.0852 | 0.9984 | 0.9992 |
| 45-54 | 70 | 94 | 85.19 | 0.1131 | 0.1251 | 0.9964 | 0.9982 |
| 55-64 | 70 | 54 | 49.19 | 0.1361 | 0.1512 | 0.9944 | 0.9972 |
| 65-69 | 69 | 10 | 10.91 | 0.2486 | 0.3013 | 0.9699 | 0.9848 |
| 70-74 | 67 | 4 | 4.64 | 0.0 | 0.4056 | 0.9244 | 0.9614 |
| 75-81 | 22 | 2 | 1.62 | 0.0 | 0.3861 | 0.9354 | 0.9672 |

\* over comments with at least one rater in the band; CI columns and reliability use comments with n>=2.

Band *means* survive because comments differ enormously in severity (between-comment variance ~1.5 on a 1-5 scale), so even a handful of raters localises a comment. Note the median 95% CI half-width of 0 for 70-74 and 75-81: that is not precision, it is n=2 raters who happened to agree exactly. See `frac_comments_zero_width_ci` in the JSON.

Elder band *difference* scores -- the quantity the discrimination test must correlate against -- do not survive:

| band pair | comments | mean human diff | median abs human diff | median min. resolvable diff | error var / observed var | reliability of diff (raw) | max attainable r |
|---|---|---|---|---|---|---|---|
| 65-69 vs 70-74 | 66 | +0.0371 | 0.1144 | 0.4814 | 1.3633 | -0.3633 | 0.0 |
| 65-69 vs 75-81 | 22 | -0.0069 | 0.127 | 0.255 | 2.9727 | -1.9727 | 0.0 |
| 70-74 vs 75-81 | 22 | +0.0281 | 0.2667 | 0.6533 | 1.6857 | -0.6857 | 0.0 |

### Is there anything for a persona to recover? The human age gradient

Before asking whether personas track age, check that the humans do. Pooled band means of `respectfulness` over the core comments, and paired Wilcoxon on the per-comment band means:

| band | unweighted mean | rating-weighted mean | comments |
|---|---|---|---|
| 18-29 | 1.9929 | 2.0079 | 70 |
| 45-54 | 1.9811 | 2.0057 | 70 |
| 55-64 | 1.957 | 1.9866 | 70 |
| 65-69 | 1.9297 | 1.9708 | 69 |
| 70-74 | 1.9243 | 1.9231 | 70 |
| 75-81 | 2.0079 | 1.9706 | 42 |

| band pair | comments | mean human diff | Wilcoxon p | rank-biserial |
|---|---|---|---|---|
| 18-29 vs 45-54 | 70 | +0.0118 | 0.8537 | -0.025 |
| 18-29 vs 55-64 | 70 | +0.0359 | 0.201 | +0.176 |
| 18-29 vs 65-69 | 69 | +0.0600 | 0.04815 | +0.274 |
| 18-29 vs 70-74 | 57 | +0.1172 | 0.02919 | +0.332 |
| 18-29 vs 75-81 | 22 | +0.1131 | 0.07378 | +0.439 |
| 45-54 vs 55-64 | 70 | +0.0241 | 0.07286 | +0.247 |
| 45-54 vs 65-69 | 69 | +0.0522 | 0.0618 | +0.263 |
| 45-54 vs 70-74 | 57 | +0.0933 | 0.02822 | +0.337 |
| 45-54 vs 75-81 | 22 | +0.0966 | 0.08541 | +0.423 |
| 55-64 vs 65-69 | 69 | +0.0226 | 0.1515 | +0.205 |
| 55-64 vs 70-74 | 57 | +0.0719 | 0.05023 | +0.312 |
| 55-64 vs 75-81 | 22 | +0.0828 | 0.1305 | +0.377 |
| 65-69 vs 70-74 | 57 | +0.0339 | 0.5932 | +0.102 |
| 65-69 vs 75-81 | 22 | -0.0069 | 0.888 | +0.037 |
| 70-74 vs 75-81 | 20 | +0.0310 | 0.916 | +0.033 |

### (a) Fidelity: does the persona land closer to real raters of that age?

Per comment, |persona mean - human band mean| vs |no-persona mean - human band mean|. Wilcoxon signed-rank paired across comments, H1: neutral error > persona error. Positive rank-biserial favours the persona. Comments restricted to human band n>=3.

| model | band | comments | MAE persona | MAE neutral | improvement | Wilcoxon p | rank-biserial |
|---|---|---|---|---|---|---|---|
| openai/gpt-4o | 18-29 | 70 | 0.3198 | 0.3195 | -0.0004 | 0.726 | -0.139 |
| openai/gpt-4o | 45-54 | 70 | 0.3085 | 0.305 | -0.0035 | 0.6831 | -0.145 |
| openai/gpt-4o | 55-64 | 70 | 0.2948 | 0.2903 | -0.0044 | 0.5384 | -0.218 |
| openai/gpt-4o | 65-69 | 69 | 0.2513 | 0.2706 | +0.0193 | 0.2676 | +0.464 |
| openai/gpt-4o | 70-74 | 57 | 0.2904 | 0.2963 | +0.0058 | 0.9319 | +0.036 |
| openai/gpt-4o | 75-81 | 3 | - | - | - | - | insufficient coverage |
| openai/gpt-4o | 18-29 (n>=1) | 70 | 0.3198 | 0.3195 | -0.0004 | 0.726 | -0.139 |
| openai/gpt-4o | 45-54 (n>=1) | 70 | 0.3085 | 0.305 | -0.0035 | 0.6831 | -0.145 |
| openai/gpt-4o | 55-64 (n>=1) | 70 | 0.2948 | 0.2903 | -0.0044 | 0.5384 | -0.218 |
| openai/gpt-4o | 65-69 (n>=1) | 69 | 0.2513 | 0.2706 | +0.0193 | 0.2676 | +0.464 |
| openai/gpt-4o | 70-74 (n>=1) | 70 | 0.3294 | 0.3246 | -0.0048 | 0.509 | -0.244 |
| openai/gpt-4o | 75-81 (n>=1) | 42 | 0.254 | 0.2302 | -0.0238 | 0.1025 | -1.000 |
| anthropic/claude-sonnet-4.5 | 18-29 | 70 | 0.354 | 0.3445 | -0.0095 | 0.3173 | -1.000 |
| anthropic/claude-sonnet-4.5 | 45-54 | 70 | 0.3333 | 0.3324 | -0.0009 | 0.6547 | -0.333 |
| anthropic/claude-sonnet-4.5 | 55-64 | 70 | 0.3185 | 0.3137 | -0.0048 | 0.3173 | -1.000 |
| anthropic/claude-sonnet-4.5 | 65-69 | 69 | 0.3029 | 0.2932 | -0.0097 | 0.3173 | -1.000 |
| anthropic/claude-sonnet-4.5 | 70-74 | 57 | 0.3336 | 0.3356 | +0.0019 | 0.1797 | +1.000 |
| anthropic/claude-sonnet-4.5 | 75-81 | 3 | - | - | - | - | insufficient coverage |
| anthropic/claude-sonnet-4.5 | 18-29 (n>=1) | 70 | 0.354 | 0.3445 | -0.0095 | 0.3173 | -1.000 |
| anthropic/claude-sonnet-4.5 | 45-54 (n>=1) | 70 | 0.3333 | 0.3324 | -0.0009 | 0.6547 | -0.333 |
| anthropic/claude-sonnet-4.5 | 55-64 (n>=1) | 70 | 0.3185 | 0.3137 | -0.0048 | 0.3173 | -1.000 |
| anthropic/claude-sonnet-4.5 | 65-69 (n>=1) | 69 | 0.3029 | 0.2932 | -0.0097 | 0.3173 | -1.000 |
| anthropic/claude-sonnet-4.5 | 70-74 (n>=1) | 70 | 0.3359 | 0.3375 | +0.0016 | 0.1797 | +1.000 |
| anthropic/claude-sonnet-4.5 | 75-81 (n>=1) | 42 | 0.254 | 0.246 | -0.0079 | 0.3173 | -1.000 |
| meta-llama/llama-3.1-70b-instruct | 18-29 | 69 | 0.5873 | 0.4327 | -0.1546 | 0.05728 | -0.391 |
| meta-llama/llama-3.1-70b-instruct | 45-54 | 70 | 0.4399 | 0.4343 | -0.0057 | 0.8939 | -0.045 |
| meta-llama/llama-3.1-70b-instruct | 55-64 | 68 | 0.4347 | 0.4159 | -0.0187 | 0.5932 | -0.182 |
| meta-llama/llama-3.1-70b-instruct | 65-69 | 69 | 0.3558 | 0.4041 | +0.0483 | 0.1091 | +0.600 |
| meta-llama/llama-3.1-70b-instruct | 70-74 | 57 | 0.4096 | 0.3862 | -0.0234 | 0.4753 | -0.267 |
| meta-llama/llama-3.1-70b-instruct | 75-81 | 3 | - | - | - | - | insufficient coverage |
| meta-llama/llama-3.1-70b-instruct | 18-29 (n>=1) | 69 | 0.5873 | 0.4327 | -0.1546 | 0.05728 | -0.391 |
| meta-llama/llama-3.1-70b-instruct | 45-54 (n>=1) | 70 | 0.4399 | 0.4343 | -0.0057 | 0.8939 | -0.045 |
| meta-llama/llama-3.1-70b-instruct | 55-64 (n>=1) | 68 | 0.4347 | 0.4159 | -0.0187 | 0.5932 | -0.182 |
| meta-llama/llama-3.1-70b-instruct | 65-69 (n>=1) | 69 | 0.3558 | 0.4041 | +0.0483 | 0.1091 | +0.600 |
| meta-llama/llama-3.1-70b-instruct | 70-74 (n>=1) | 70 | 0.455 | 0.4407 | -0.0143 | 0.7232 | -0.115 |
| meta-llama/llama-3.1-70b-instruct | 75-81 (n>=1) | 41 | 0.2927 | 0.2764 | -0.0163 | 0.593 | -0.333 |
| qwen/qwen-2.5-72b-instruct | 18-29 | 70 | 0.3713 | 0.3709 | -0.0003 | 0.6549 | -0.151 |
| qwen/qwen-2.5-72b-instruct | 45-54 | 70 | 0.3369 | 0.3661 | +0.0292 | 0.1677 | +0.491 |
| qwen/qwen-2.5-72b-instruct | 55-64 | 70 | 0.3451 | 0.3392 | -0.0060 | 0.7784 | -0.111 |
| qwen/qwen-2.5-72b-instruct | 65-69 | 69 | 0.3096 | 0.3138 | +0.0042 | 0.5745 | +0.222 |
| qwen/qwen-2.5-72b-instruct | 70-74 | 57 | 0.3244 | 0.3483 | +0.0239 | 0.3452 | +0.467 |
| qwen/qwen-2.5-72b-instruct | 75-81 | 3 | - | - | - | - | insufficient coverage |
| qwen/qwen-2.5-72b-instruct | 18-29 (n>=1) | 70 | 0.3713 | 0.3709 | -0.0003 | 0.6549 | -0.151 |
| qwen/qwen-2.5-72b-instruct | 45-54 (n>=1) | 70 | 0.3369 | 0.3661 | +0.0292 | 0.1677 | +0.491 |
| qwen/qwen-2.5-72b-instruct | 55-64 (n>=1) | 70 | 0.3451 | 0.3392 | -0.0060 | 0.7784 | -0.111 |
| qwen/qwen-2.5-72b-instruct | 65-69 (n>=1) | 69 | 0.3096 | 0.3138 | +0.0042 | 0.5745 | +0.222 |
| qwen/qwen-2.5-72b-instruct | 70-74 (n>=1) | 70 | 0.3522 | 0.3764 | +0.0242 | 0.3105 | +0.429 |
| qwen/qwen-2.5-72b-instruct | 75-81 (n>=1) | 42 | 0.3413 | 0.3095 | -0.0317 | 0.1756 | -0.667 |
| google/gemini-2.5-flash | 18-29 | 70 | 0.4415 | 0.421 | -0.0205 | 0.4833 | -0.278 |
| google/gemini-2.5-flash | 45-54 | 70 | 0.4185 | 0.4177 | -0.0008 | 0.8135 | -0.077 |
| google/gemini-2.5-flash | 55-64 | 70 | 0.4391 | 0.3975 | -0.0416 | 0.1522 | -0.509 |
| google/gemini-2.5-flash | 65-69 | 69 | 0.4422 | 0.387 | -0.0552 | 0.05758 | -0.711 |
| google/gemini-2.5-flash | 70-74 | 57 | 0.4096 | 0.3453 | -0.0643 | 0.02771 | -1.000 |
| google/gemini-2.5-flash | 75-81 | 3 | - | - | - | - | insufficient coverage |
| google/gemini-2.5-flash | 18-29 (n>=1) | 70 | 0.4415 | 0.421 | -0.0205 | 0.4833 | -0.278 |
| google/gemini-2.5-flash | 45-54 (n>=1) | 70 | 0.4185 | 0.4177 | -0.0008 | 0.8135 | -0.077 |
| google/gemini-2.5-flash | 55-64 (n>=1) | 70 | 0.4391 | 0.3975 | -0.0416 | 0.1522 | -0.509 |
| google/gemini-2.5-flash | 65-69 (n>=1) | 69 | 0.4422 | 0.387 | -0.0552 | 0.05758 | -0.711 |
| google/gemini-2.5-flash | 70-74 (n>=1) | 70 | 0.4788 | 0.4121 | -0.0667 | 0.01223 | -0.891 |
| google/gemini-2.5-flash | 75-81 (n>=1) | 42 | 0.3254 | 0.3175 | -0.0079 | 0.7532 | -0.143 |
| mistralai/mistral-large-2512 | 18-29 | 70 | 0.3517 | 0.3652 | +0.0135 | 0.6456 | +0.164 |
| mistralai/mistral-large-2512 | 45-54 | 70 | 0.3262 | 0.3587 | +0.0324 | 0.255 | +0.372 |
| mistralai/mistral-large-2512 | 55-64 | 70 | 0.2877 | 0.3416 | +0.0539 | 0.09289 | +0.527 |
| mistralai/mistral-large-2512 | 65-69 | 69 | 0.2602 | 0.309 | +0.0488 | 0.168 | +0.449 |
| mistralai/mistral-large-2512 | 70-74 | 57 | 0.3431 | 0.3773 | +0.0343 | 0.3122 | +0.378 |
| mistralai/mistral-large-2512 | 75-81 | 3 | - | - | - | - | insufficient coverage |
| mistralai/mistral-large-2512 | 18-29 (n>=1) | 70 | 0.3517 | 0.3652 | +0.0135 | 0.6456 | +0.164 |
| mistralai/mistral-large-2512 | 45-54 (n>=1) | 70 | 0.3262 | 0.3587 | +0.0324 | 0.255 | +0.372 |
| mistralai/mistral-large-2512 | 55-64 (n>=1) | 70 | 0.2877 | 0.3416 | +0.0539 | 0.09289 | +0.527 |
| mistralai/mistral-large-2512 | 65-69 (n>=1) | 69 | 0.2602 | 0.309 | +0.0488 | 0.168 | +0.449 |
| mistralai/mistral-large-2512 | 70-74 (n>=1) | 70 | 0.3579 | 0.4001 | +0.0422 | 0.1349 | +0.487 |
| mistralai/mistral-large-2512 | 75-81 (n>=1) | 42 | 0.2857 | 0.2937 | +0.0079 | 0.8326 | +0.083 |

### (b) Discrimination: can personas tell the bands apart the way humans do?

Spearman correlation across comments between (persona_A - persona_B) and (human_A - human_B). This is the paper's actual question.

| model | band pair | comments | mean persona diff | sd persona diff | mean human diff | Spearman rho | p |
|---|---|---|---|---|---|---|---|
| openai/gpt-4o | 18-29 vs 45-54 | 70 | -0.0238 | 0.1637 | +0.0118 | +0.0531 | 0.6626 |
| openai/gpt-4o | 18-29 vs 55-64 | 70 | -0.0190 | 0.1377 | +0.0359 | -0.0671 | 0.5812 |
| openai/gpt-4o | 18-29 vs 65-69 | 69 | -0.0242 | 0.1041 | +0.0600 | -0.3484 | 0.003346 |
| openai/gpt-4o | 18-29 vs 70-74 | 57 | -0.0292 | 0.17 | +0.1172 | -0.0486 | 0.7198 |
| openai/gpt-4o | 18-29 vs 75-81 | 22 | +0.0000 | 0.1029 | +0.1131 | +0.1426 | 0.5268 |
| openai/gpt-4o | 45-54 vs 55-64 | 70 | +0.0048 | 0.0896 | +0.0241 | +0.0649 | 0.5933 |
| openai/gpt-4o | 45-54 vs 65-69 | 69 | -0.0000 | 0.1278 | +0.0522 | -0.0336 | 0.7837 |
| openai/gpt-4o | 45-54 vs 70-74 | 57 | +0.0058 | 0.0994 | +0.0933 | -0.0352 | 0.7949 |
| openai/gpt-4o | 45-54 vs 75-81 | 22 | -0.0152 | 0.0711 | +0.0966 | -0.2236 | 0.3172 |
| openai/gpt-4o | 55-64 vs 65-69 | 69 | -0.0048 | 0.1212 | +0.0226 | -0.1120 | 0.3594 |
| openai/gpt-4o | 55-64 vs 70-74 | 57 | -0.0058 | 0.0994 | +0.0719 | -0.0642 | 0.6351 |
| openai/gpt-4o | 55-64 vs 75-81 | 22 | +0.0000 | 0.1029 | +0.0828 | -0.1902 | 0.3967 |
| openai/gpt-4o | 65-69 vs 70-74 | 57 | +0.0000 | 0.1409 | +0.0339 | -0.0711 | 0.5992 |
| openai/gpt-4o | 65-69 vs 75-81 | 22 | +0.0152 | 0.1619 | -0.0069 | -0.0238 | 0.9163 |
| openai/gpt-4o | 70-74 vs 75-81 | 20 | 0.0 | 0.0 | 0.031 | - | no persona variation between bands |
| anthropic/claude-sonnet-4.5 | 18-29 vs 45-54 | 70 | +0.0095 | 0.0797 | +0.0118 | -0.1817 | 0.1322 |
| anthropic/claude-sonnet-4.5 | 18-29 vs 55-64 | 70 | -0.0048 | 0.0398 | +0.0359 | -0.1936 | 0.1082 |
| anthropic/claude-sonnet-4.5 | 18-29 vs 65-69 | 69 | 0.0 | 0.0 | 0.06 | - | no persona variation between bands |
| anthropic/claude-sonnet-4.5 | 18-29 vs 70-74 | 57 | -0.0058 | 0.0442 | +0.1172 | -0.1056 | 0.4344 |
| anthropic/claude-sonnet-4.5 | 18-29 vs 75-81 | 22 | +0.0152 | 0.0711 | +0.1131 | +0.2236 | 0.3172 |
| anthropic/claude-sonnet-4.5 | 45-54 vs 55-64 | 70 | -0.0143 | 0.0886 | +0.0241 | -0.1999 | 0.09703 |
| anthropic/claude-sonnet-4.5 | 45-54 vs 65-69 | 69 | -0.0097 | 0.0803 | +0.0522 | +0.1157 | 0.3438 |
| anthropic/claude-sonnet-4.5 | 45-54 vs 70-74 | 57 | -0.0175 | 0.098 | +0.0933 | +0.0563 | 0.6773 |
| anthropic/claude-sonnet-4.5 | 45-54 vs 75-81 | 22 | +0.0152 | 0.0711 | +0.0966 | +0.1204 | 0.5936 |
| anthropic/claude-sonnet-4.5 | 55-64 vs 65-69 | 69 | +0.0048 | 0.0401 | +0.0226 | +0.1644 | 0.177 |
| anthropic/claude-sonnet-4.5 | 55-64 vs 70-74 | 57 | +0.0000 | 0.063 | +0.0719 | +0.0399 | 0.7684 |
| anthropic/claude-sonnet-4.5 | 55-64 vs 75-81 | 22 | +0.0152 | 0.0711 | +0.0828 | +0.1204 | 0.5935 |
| anthropic/claude-sonnet-4.5 | 65-69 vs 70-74 | 57 | -0.0058 | 0.0442 | +0.0339 | +0.1167 | 0.3874 |
| anthropic/claude-sonnet-4.5 | 65-69 vs 75-81 | 22 | +0.0152 | 0.0711 | -0.0069 | +0.2929 | 0.1859 |
| anthropic/claude-sonnet-4.5 | 70-74 vs 75-81 | 20 | +0.0167 | 0.0745 | +0.0310 | +0.1019 | 0.6691 |
| meta-llama/llama-3.1-70b-instruct | 18-29 vs 45-54 | 69 | +0.2729 | 0.6157 | +0.0116 | -0.0688 | 0.5745 |
| meta-llama/llama-3.1-70b-instruct | 18-29 vs 55-64 | 67 | +0.3085 | 0.6108 | +0.0354 | -0.0209 | 0.8665 |
| meta-llama/llama-3.1-70b-instruct | 18-29 vs 65-69 | 68 | +0.1912 | 0.6907 | +0.0630 | -0.0683 | 0.5797 |
| meta-llama/llama-3.1-70b-instruct | 18-29 vs 70-74 | 56 | +0.1637 | 0.6894 | +0.1160 | +0.0451 | 0.7415 |
| meta-llama/llama-3.1-70b-instruct | 18-29 vs 75-81 | 21 | +0.1905 | 0.86 | +0.1158 | -0.1490 | 0.5193 |
| meta-llama/llama-3.1-70b-instruct | 45-54 vs 55-64 | 68 | +0.0221 | 0.1575 | +0.0208 | +0.2125 | 0.08189 |
| meta-llama/llama-3.1-70b-instruct | 45-54 vs 65-69 | 69 | -0.0797 | 0.2838 | +0.0522 | +0.1407 | 0.2489 |
| meta-llama/llama-3.1-70b-instruct | 45-54 vs 70-74 | 57 | -0.0731 | 0.2617 | +0.0933 | -0.2346 | 0.07901 |
| meta-llama/llama-3.1-70b-instruct | 45-54 vs 75-81 | 21 | -0.0714 | 0.6204 | +0.1043 | +0.0576 | 0.8042 |
| meta-llama/llama-3.1-70b-instruct | 55-64 vs 65-69 | 67 | -0.1045 | 0.3293 | +0.0326 | -0.2202 | 0.07339 |
| meta-llama/llama-3.1-70b-instruct | 55-64 vs 70-74 | 55 | -0.1030 | 0.3564 | +0.0824 | -0.3482 | 0.009191 |
| meta-llama/llama-3.1-70b-instruct | 55-64 vs 75-81 | 21 | -0.0635 | 0.5438 | +0.0868 | +0.0267 | 0.9086 |
| meta-llama/llama-3.1-70b-instruct | 65-69 vs 70-74 | 57 | +0.0117 | 0.1886 | +0.0339 | -0.2798 | 0.03502 |
| meta-llama/llama-3.1-70b-instruct | 65-69 vs 75-81 | 21 | +0.0794 | 0.6742 | -0.0039 | +0.2842 | 0.2118 |
| meta-llama/llama-3.1-70b-instruct | 70-74 vs 75-81 | 19 | +0.0175 | 0.8201 | +0.0747 | -0.1867 | 0.444 |
| qwen/qwen-2.5-72b-instruct | 18-29 vs 45-54 | 70 | -0.0286 | 0.1579 | +0.0118 | -0.0777 | 0.5226 |
| qwen/qwen-2.5-72b-instruct | 18-29 vs 55-64 | 70 | +0.0286 | 0.2252 | +0.0359 | +0.1812 | 0.1334 |
| qwen/qwen-2.5-72b-instruct | 18-29 vs 65-69 | 69 | +0.0193 | 0.2484 | +0.0600 | -0.0642 | 0.6 |
| qwen/qwen-2.5-72b-instruct | 18-29 vs 70-74 | 57 | +0.0175 | 0.1828 | +0.1172 | -0.0179 | 0.8946 |
| qwen/qwen-2.5-72b-instruct | 18-29 vs 75-81 | 22 | +0.0758 | 0.2708 | +0.1131 | -0.0252 | 0.9115 |
| qwen/qwen-2.5-72b-instruct | 45-54 vs 55-64 | 70 | +0.0571 | 0.1964 | +0.0241 | +0.0517 | 0.6707 |
| qwen/qwen-2.5-72b-instruct | 45-54 vs 65-69 | 69 | +0.0435 | 0.1973 | +0.0522 | +0.1523 | 0.2116 |
| qwen/qwen-2.5-72b-instruct | 45-54 vs 70-74 | 57 | +0.0468 | 0.1598 | +0.0933 | +0.1518 | 0.2597 |
| qwen/qwen-2.5-72b-instruct | 45-54 vs 75-81 | 22 | +0.0758 | 0.1761 | +0.0966 | +0.1008 | 0.6555 |
| qwen/qwen-2.5-72b-instruct | 55-64 vs 65-69 | 69 | -0.0145 | 0.1559 | +0.0226 | -0.1217 | 0.3192 |
| qwen/qwen-2.5-72b-instruct | 55-64 vs 70-74 | 57 | -0.0292 | 0.2116 | +0.0719 | -0.1060 | 0.4324 |
| qwen/qwen-2.5-72b-instruct | 55-64 vs 75-81 | 22 | +0.0000 | 0.1029 | +0.0828 | -0.0475 | 0.8336 |
| qwen/qwen-2.5-72b-instruct | 65-69 vs 70-74 | 57 | -0.0117 | 0.2354 | +0.0339 | +0.1316 | 0.3291 |
| qwen/qwen-2.5-72b-instruct | 65-69 vs 75-81 | 22 | +0.0303 | 0.2279 | -0.0069 | +0.0476 | 0.8334 |
| qwen/qwen-2.5-72b-instruct | 70-74 vs 75-81 | 20 | +0.0500 | 0.2236 | +0.0310 | +0.3260 | 0.1607 |
| google/gemini-2.5-flash | 18-29 vs 45-54 | 70 | -0.0000 | 0.2538 | +0.0118 | +0.3263 | 0.005842 |
| google/gemini-2.5-flash | 18-29 vs 55-64 | 70 | -0.0238 | 0.2148 | +0.0359 | -0.0161 | 0.8948 |
| google/gemini-2.5-flash | 18-29 vs 65-69 | 69 | -0.0338 | 0.2566 | +0.0600 | -0.0783 | 0.5226 |
| google/gemini-2.5-flash | 18-29 vs 70-74 | 57 | -0.0117 | 0.2179 | +0.1172 | +0.3258 | 0.0134 |
| google/gemini-2.5-flash | 18-29 vs 75-81 | 22 | +0.0152 | 0.3167 | +0.1131 | +0.2110 | 0.346 |
| google/gemini-2.5-flash | 45-54 vs 55-64 | 70 | -0.0238 | 0.2856 | +0.0241 | +0.1685 | 0.1632 |
| google/gemini-2.5-flash | 45-54 vs 65-69 | 69 | -0.0290 | 0.3221 | +0.0522 | +0.0251 | 0.8381 |
| google/gemini-2.5-flash | 45-54 vs 70-74 | 57 | -0.0292 | 0.202 | +0.0933 | -0.0170 | 0.9001 |
| google/gemini-2.5-flash | 45-54 vs 75-81 | 22 | +0.0152 | 0.2407 | +0.0966 | +0.3645 | 0.09537 |
| google/gemini-2.5-flash | 55-64 vs 65-69 | 69 | -0.0048 | 0.1761 | +0.0226 | +0.0197 | 0.8724 |
| google/gemini-2.5-flash | 55-64 vs 70-74 | 57 | +0.0234 | 0.1238 | +0.0719 | +0.3048 | 0.02113 |
| google/gemini-2.5-flash | 55-64 vs 75-81 | 22 | +0.0303 | 0.1755 | +0.0828 | -0.0785 | 0.7283 |
| google/gemini-2.5-flash | 65-69 vs 70-74 | 57 | +0.0234 | 0.2076 | +0.0339 | +0.0847 | 0.5312 |
| google/gemini-2.5-flash | 65-69 vs 75-81 | 22 | +0.0909 | 0.2942 | -0.0069 | +0.0999 | 0.6584 |
| google/gemini-2.5-flash | 70-74 vs 75-81 | 20 | +0.0167 | 0.0745 | +0.0310 | +0.1019 | 0.6691 |
| mistralai/mistral-large-2512 | 18-29 vs 45-54 | 70 | -0.0143 | 0.223 | +0.0118 | -0.0929 | 0.4443 |
| mistralai/mistral-large-2512 | 18-29 vs 55-64 | 70 | -0.0190 | 0.2332 | +0.0359 | +0.0452 | 0.7105 |
| mistralai/mistral-large-2512 | 18-29 vs 65-69 | 69 | -0.0338 | 0.1992 | +0.0600 | -0.0685 | 0.5761 |
| mistralai/mistral-large-2512 | 18-29 vs 70-74 | 57 | +0.0000 | 0.0891 | +0.1172 | -0.0479 | 0.7237 |
| mistralai/mistral-large-2512 | 18-29 vs 75-81 | 22 | -0.1364 | 0.5696 | +0.1131 | -0.3564 | 0.1035 |
| mistralai/mistral-large-2512 | 45-54 vs 55-64 | 70 | -0.0048 | 0.1749 | +0.0241 | -0.1496 | 0.2165 |
| mistralai/mistral-large-2512 | 45-54 vs 65-69 | 69 | -0.0193 | 0.1797 | +0.0522 | +0.1870 | 0.1238 |
| mistralai/mistral-large-2512 | 45-54 vs 70-74 | 57 | -0.0175 | 0.1828 | +0.0933 | +0.3394 | 0.009804 |
| mistralai/mistral-large-2512 | 45-54 vs 75-81 | 22 | -0.1212 | 0.4992 | +0.0966 | -0.3542 | 0.1058 |
| mistralai/mistral-large-2512 | 55-64 vs 65-69 | 69 | -0.0145 | 0.1333 | +0.0226 | +0.1541 | 0.2062 |
| mistralai/mistral-large-2512 | 55-64 vs 70-74 | 57 | -0.0058 | 0.1836 | +0.0719 | +0.2802 | 0.03477 |
| mistralai/mistral-large-2512 | 55-64 vs 75-81 | 22 | -0.1364 | 0.5696 | +0.0828 | -0.4041 | 0.06217 |
| mistralai/mistral-large-2512 | 65-69 vs 70-74 | 57 | +0.0058 | 0.0769 | +0.0339 | +0.2453 | 0.06589 |
| mistralai/mistral-large-2512 | 65-69 vs 75-81 | 22 | -0.1364 | 0.5696 | -0.0069 | +0.0623 | 0.7829 |
| mistralai/mistral-large-2512 | 70-74 vs 75-81 | 20 | -0.1500 | 0.597 | +0.0310 | -0.2062 | 0.3831 |

#### Multiple-comparison accounting

88 discrimination tests were run (6 models x band pairs). 8 reached nominal p<.05; 4.4 are expected by chance. **0** survive Benjamini-Hochberg at q=.05.

Nominally significant tests (note the signs -- several are in the *wrong* direction, i.e. the persona contrast anti-correlates with the human contrast):

| model | pair | rho | p |
|---|---|---|---|
| openai/gpt-4o | 18-29 vs 65-69 | -0.3484 | 0.003346 |
| google/gemini-2.5-flash | 18-29 vs 45-54 | +0.3263 | 0.005842 |
| meta-llama/llama-3.1-70b-instruct | 55-64 vs 70-74 | -0.3482 | 0.009191 |
| mistralai/mistral-large-2512 | 45-54 vs 70-74 | +0.3394 | 0.009804 |
| google/gemini-2.5-flash | 18-29 vs 70-74 | +0.3258 | 0.0134 |
| google/gemini-2.5-flash | 55-64 vs 70-74 | +0.3048 | 0.02113 |
| mistralai/mistral-large-2512 | 55-64 vs 70-74 | +0.2802 | 0.03477 |
| meta-llama/llama-3.1-70b-instruct | 65-69 vs 70-74 | -0.2798 | 0.03502 |

Headline pairs across models (sign test on rho>0, 6 models):

| band pair | rhos | median rho | n positive | sign-test p |
|---|---|---|---|---|
| 18-29 vs 65-69 | -0.35, -0.07, -0.06, -0.08, -0.07 | -0.0685 | 0/5 | 1.0 |
| 18-29 vs 70-74 | -0.05, -0.11, +0.05, -0.02, +0.33, -0.05 | -0.0329 | 2/6 | 0.8906 |
| 65-69 vs 70-74 | -0.07, +0.12, -0.28, +0.13, +0.08, +0.25 | +0.1007 | 4/6 | 0.3438 |
| 65-69 vs 75-81 | -0.02, +0.29, +0.28, +0.05, +0.10, +0.06 | +0.0811 | 5/6 | 0.1094 |
| 70-74 vs 75-81 | +0.10, -0.19, +0.33, +0.10, -0.21 | +0.1019 | 3/5 | 0.5 |

### Persona responsiveness (sanity: does the persona move the model at all?)

| model | neutral | 18-29 | 45-54 | 55-64 | 65-69 | 70-74 | 75-81 |
|---|---|---|---|---|---|---|---|
| openai/gpt-4o | 1.7952 | 1.7952 | 1.819 | 1.8143 | 1.819 | 1.819 | 1.7905 |
| anthropic/claude-sonnet-4.5 | 1.8238 | 1.8143 | 1.8048 | 1.819 | 1.8143 | 1.819 | 1.8095 |
| meta-llama/llama-3.1-70b-instruct | 1.8952 | 2.0531 | 1.7881 | 1.7892 | 1.8667 | 1.8524 | 1.7995 |
| qwen/qwen-2.5-72b-instruct | 1.6905 | 1.7238 | 1.7524 | 1.6952 | 1.7095 | 1.7095 | 1.7048 |
| google/gemini-2.5-flash | 1.719 | 1.7381 | 1.7381 | 1.7619 | 1.7667 | 1.7286 | 1.7381 |
| mistralai/mistral-large-2512 | 1.6714 | 1.7476 | 1.7619 | 1.7667 | 1.781 | 1.7571 | 1.7952 |

Across 36 model x band cells, the median absolute shift in mean rating caused by adding a persona is **0.0238** points on a 1-5 scale (max 0.1787). The personas barely move the models on this task, which is the proximate cause of both null results below -- and it applies to the young personas as much as the old.

### Run diagnostics

| model | cells | unparsed | refusal-flagged | worst-condition unparsed rate | cost USD |
|---|---|---|---|---|---|
| openai/gpt-4o | 1470 | 0 | 0 | 0.0% | 0.3683 |
| anthropic/claude-sonnet-4.5 | 1470 | 0 | 0 | 0.0% | 0.5777 |
| meta-llama/llama-3.1-70b-instruct | 1470 | 75 | 1 | 15.2% | 0.0802 |
| qwen/qwen-2.5-72b-instruct | 1470 | 0 | 0 | 0.0% | 0.0524 |
| google/gemini-2.5-flash | 1470 | 0 | 0 | 0.0% | 0.0425 |
| mistralai/mistral-large-2512 | 1470 | 0 | 0 | 0.0% | 0.0707 |

**Caveat.** Unparsed cells are dropped, and for the following models they are not distributed evenly across conditions, so the affected condition means are computed on a non-random subset:

- `meta-llama/llama-3.1-70b-instruct`: 24 32 (15.2%), 50 16 (7.6%), 60 10 (4.8%), 67 4 (1.9%), 72 4 (1.9%), 78 9 (4.3%)

Inspection of the raw strings shows these are format-following failures (prose of the form "I would rate this comment a 1, ..."), not refusals. Because the dropped responses skew toward the low end of the scale, the affected condition means are biased upward. This matters for that model's young-persona row and is not load-bearing for any elder-band conclusion.

Total API cost for the complete experiment (sum over all cached cells): **$1.1919** for 8,820 calls. This invocation was analysis-only and made no API calls.

## 3. Reading

Four findings, in the order that matters.

**1. The human age signal is real between distant ages and vanishes inside old age.** Across the core comments, raters aged 18-29 judge the same comments +0.117 points more respectful than raters aged 70-74 (paired Wilcoxon p=0.02919, n=57), and the band means fall monotonically from 18-29 through 70-74. But the elder-vs-elder contrasts are flat: 65-69 vs 70-74 = +0.034 (p=0.5932), 65-69 vs 75-81 = -0.007 (p=0.888), 70-74 vs 75-81 = +0.031 (p=0.916).

**2. That flatness is a density artefact, not evidence of homogeneity.** For every elder band pair the sampling-error variance of the per-comment difference *exceeds* its observed variance (ratios 1.36, 2.97, 1.69), giving a reliability of 0 and a ceiling of r=0 on any correlation a persona could achieve against these targets. The median smallest difference the available raters could resolve on a comment is 65-69 vs 70-74: 0.4814, 65-69 vs 75-81: 0.255, 70-74 vs 75-81: 0.6533 points, against true differences that are an order of magnitude smaller. The elder-vs-elder target is not a weak signal we failed to detect; at this density it is not a measurement.

**3. Fidelity: no model's elder persona gets reliably closer to real elders than no persona does.** Absolute errors are small in absolute terms (MAE ~0.25-0.45 on a 1-5 scale -- the models track comment severity well) but the persona contributes essentially nothing to that. No band x model cell shows a significant persona advantage; the only nominally significant fidelity result in the table is a persona making matters *worse*.

**4. Discrimination fails, but it fails everywhere, including the positive control.** Of 88 tests, 8 reach nominal p<.05 against 4.4 expected by chance, and 0 survive Benjamini-Hochberg. Crucially the models cannot separate a 24-year-old from a 67-year-old either. The proximate cause is visible in the responsiveness table: the median absolute shift a persona produces is 0.0238 points on a 1-5 scale. On this task these personas are close to inert, so the elder-vs-elder null is **not** specific to old age and must not be reported as though it were.

### What this does and does not license

- MHSC genuinely is an exception to the resolution claim *at the level of the sampling frame*: exact ages are recorded, so age-resolved elder targets can be constructed at all, which POPQUORN makes impossible. The claim "no dataset records age finely enough" is false and should not be made.
- MHSC is **not** an exception at the level of *statistical power*. Recording an age is not the same as sampling enough people at that age. 12 annotators aged 75-81 in the whole corpus, contributing 68 ratings spread over 42 comments, with a median of one 75+ rater per comment, cannot adjudicate anything. Fine-grained metadata over a thin tail produces bins that are nominally resolved and empirically empty.
- The honest statement is therefore: **MHSC can support a 65-69 elder target and, marginally, a 70-74 one; it cannot support a 75+ target, and it cannot support any contrast between elder bands.**
- Because the positive control also fails, this run cannot separate "elder personas are unvalidatable" from "these personas do not respond to age prompting on this task." Both are consistent with the data. A stronger persona elicitation might move the models; it would still have nothing reliable to be scored against inside old age, which is finding 2 and does not depend on the models at all.

### Limitations

- n=70 comments. Forced by the data, not chosen; see Section 1.
- The 70 core comments are MHSC's high-replication calibration set and are not a random sample of the corpus. Elder targets exist only there.
- Each comment's elder band mean is built from a different handful of people (every 65+ annotator rated ~6 of the 70), so band means confound age with individual rater severity. MHSC's own Rasch severity adjustment would address this but we deliberately do not use it -- see the header note on why `hate_speech_score` is unsuitable as a target here.
- One persona formulation, one rating instruction, one target column. The nulls are nulls for this operationalisation.
- `respect` is a judgement about a comment's stance toward a target group; it is the nearest available analogue to a politeness rating but it is not one.

