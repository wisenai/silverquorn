# Pilot analysis (frozen 2026-07-29)

Started: 8; completers (>= 30/40 real items): 5; excluded partials (real items rated): {'P05': 3, 'P07': 1, 'P08': 27}
Brackets: {'70–74': 3, '65–69': 2} | Residence: {'India': 3, 'United States': 2}

## Q1 pilot bracket means (bootstrap 95% CI over participants)
- 65–69: n=2 mean=3.312 CI=[3.15, 3.475]
- 70–74: n=3 mean=3.175 CI=[2.725, 3.425]
- all_65plus: n=5 mean=3.23 CI=[2.96, 3.435]

POPQUORN >65 band on these 40 items: mean=3.343 ± 0.429 (95% CI across items)

## Q2 pilot per-item SD vs persona within-cell SD (paired Wilcoxon)
- openai/gpt-4o|65: pilot med SD=0.707 vs persona 0.0 (diff 0.548, p=0.0, n=40)
- openai/gpt-4o|75: pilot med SD=0.707 vs persona 0.0 (diff 0.548, p=0.0, n=40)
- openai/gpt-4o|85: pilot med SD=0.707 vs persona 0.0 (diff 0.548, p=0.0, n=40)
- anthropic/claude-sonnet-4.5|65: pilot med SD=0.707 vs persona 0.0 (diff 0.548, p=0.0, n=40)
- anthropic/claude-sonnet-4.5|75: pilot med SD=0.707 vs persona 0.0 (diff 0.627, p=0.0, n=40)
- anthropic/claude-sonnet-4.5|85: pilot med SD=0.707 vs persona 0.0 (diff 0.598, p=0.0, n=40)
- meta-llama/llama-3.1-70b-instruct|65: pilot med SD=0.707 vs persona 0.0 (diff 0.533, p=0.0, n=40)
- meta-llama/llama-3.1-70b-instruct|75: pilot med SD=0.707 vs persona 0.0 (diff 0.548, p=0.0, n=40)
- meta-llama/llama-3.1-70b-instruct|85: pilot med SD=0.707 vs persona 0.0 (diff 0.548, p=0.0, n=40)
- qwen/qwen-2.5-72b-instruct|65: pilot med SD=0.707 vs persona 0.0 (diff 0.548, p=0.0, n=40)
- qwen/qwen-2.5-72b-instruct|75: pilot med SD=0.707 vs persona 0.0 (diff 0.548, p=0.0, n=40)
- qwen/qwen-2.5-72b-instruct|85: pilot med SD=0.707 vs persona 0.0 (diff 0.548, p=0.0, n=40)
- google/gemini-2.5-flash|65: pilot med SD=0.707 vs persona 0.0 (diff 0.548, p=0.0, n=40)
- google/gemini-2.5-flash|75: pilot med SD=0.707 vs persona 0.0 (diff 0.548, p=0.0, n=40)
- google/gemini-2.5-flash|85: pilot med SD=0.707 vs persona 0.0 (diff 0.447, p=0.0, n=40)
- mistralai/mistral-large-2512|65: pilot med SD=0.707 vs persona 0.0 (diff 0.548, p=0.0, n=40)
- mistralai/mistral-large-2512|75: pilot med SD=0.707 vs persona 0.0 (diff 0.548, p=0.0, n=40)
- mistralai/mistral-large-2512|85: pilot med SD=0.707 vs persona 0.0 (diff 0.548, p=0.0, n=40)

## Q3 |persona-75 - pilot| vs |neutral - pilot| per item
- openai/gpt-4o: med err persona=0.467 vs neutral=0.433; persona closer on 10/40 items (p=0.75791)
- anthropic/claude-sonnet-4.5: med err persona=0.533 vs neutral=0.4; persona closer on 1/40 items (p=0.01644)
- meta-llama/llama-3.1-70b-instruct: med err persona=0.733 vs neutral=0.8; persona closer on 7/40 items (p=0.25471)
- qwen/qwen-2.5-72b-instruct: med err persona=0.6 vs neutral=0.467; persona closer on 8/40 items (p=0.73181)
- google/gemini-2.5-flash: med err persona=0.6 vs neutral=0.533; persona closer on 9/40 items (p=0.11047)
- mistralai/mistral-large-2512: med err persona=0.433 vs neutral=0.4; persona closer on 5/40 items (p=0.93721)

## Residence covariate: {"India": {"n": 3, "mean": 3.317}, "United States": {"n": 2, "mean": 3.1}}
