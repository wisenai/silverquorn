# Harness QC report

Runs dir: `runs` | items: 200 | samples/cell: 3

Refusals are unparseable responses matching refusal phrasing; format failures are other unparseable responses; errors are cells with no response content (network/API). All three plus missing cells count as needs-rerun.

| model | condition | expected | attempted | parsed | parse rate | refusals | refusal rate | format fail | errors | needs rerun |
|---|---|---|---|---|---|---|---|---|---|---|
| openai/gpt-4o | neutral | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| openai/gpt-4o | 25 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| openai/gpt-4o | 35 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| openai/gpt-4o | 45 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| openai/gpt-4o | 55 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| openai/gpt-4o | 65 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| openai/gpt-4o | 75 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| openai/gpt-4o | 85 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| anthropic/claude-sonnet-4.5 | neutral | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| anthropic/claude-sonnet-4.5 | 25 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| anthropic/claude-sonnet-4.5 | 35 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| anthropic/claude-sonnet-4.5 | 45 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| anthropic/claude-sonnet-4.5 | 55 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| anthropic/claude-sonnet-4.5 | 65 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| anthropic/claude-sonnet-4.5 | 75 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| anthropic/claude-sonnet-4.5 | 85 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| meta-llama/llama-3.1-70b-instruct | neutral | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| meta-llama/llama-3.1-70b-instruct | 25 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| meta-llama/llama-3.1-70b-instruct | 35 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| meta-llama/llama-3.1-70b-instruct | 45 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| meta-llama/llama-3.1-70b-instruct | 55 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| meta-llama/llama-3.1-70b-instruct | 65 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| meta-llama/llama-3.1-70b-instruct | 75 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| meta-llama/llama-3.1-70b-instruct | 85 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| qwen/qwen-2.5-72b-instruct | neutral | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| qwen/qwen-2.5-72b-instruct | 25 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| qwen/qwen-2.5-72b-instruct | 35 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| qwen/qwen-2.5-72b-instruct | 45 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| qwen/qwen-2.5-72b-instruct | 55 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| qwen/qwen-2.5-72b-instruct | 65 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| qwen/qwen-2.5-72b-instruct | 75 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| qwen/qwen-2.5-72b-instruct | 85 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| google/gemini-2.5-flash | neutral | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| google/gemini-2.5-flash | 25 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| google/gemini-2.5-flash | 35 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| google/gemini-2.5-flash | 45 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| google/gemini-2.5-flash | 55 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| google/gemini-2.5-flash | 65 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| google/gemini-2.5-flash | 75 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| google/gemini-2.5-flash | 85 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| mistralai/mistral-large-2512 | neutral | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| mistralai/mistral-large-2512 | 25 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| mistralai/mistral-large-2512 | 35 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| mistralai/mistral-large-2512 | 45 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| mistralai/mistral-large-2512 | 55 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| mistralai/mistral-large-2512 | 65 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| mistralai/mistral-large-2512 | 75 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |
| mistralai/mistral-large-2512 | 85 | 600 | 600 | 600 | 100.0% | 0 | 0.0% | 0 | 0 | 0 |

## Cells needing re-run

None. All cells have a valid parse.
