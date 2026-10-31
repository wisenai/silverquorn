# Table 2, recomputed from released raw data

Counts are over distinct annotators, deduplicated by each release's
own annotator/participant ID. `?` means the release cannot answer the
question, which is not the same as zero.

| Dataset | Age recording | Last bin | n>=65 | n>=75 | annotators |
|---|---|---|---:|---:|---:|
| POPQUORN politeness | 5-yr brackets | >65 | 69 | ? | 506 |
| POPQUORN offensiveness | 5-yr brackets | >65 | 29 | ? | 262 |
| D3CODE | 3 bins to 50+ | 50+ | ? | ? | 4309 |
| DICES-350 | generation labels | gen x+ | ? | ? | 123 |
| PRISM | 10-yr brackets | 65+ | 106 | ? | 1500 |
| Measuring Hate Speech | exact age | none | 203 | 12 | 7912 |
| SBIC | exact age | none | 6 | 0 | 301 |
| NLPositionality | exact age | none | -- | -- | -- |

## Notes

- **POPQUORN politeness** — The terminal bin is open-ended, so nobody can say how many of these annotators are over 75.
- **POPQUORN offensiveness** — The terminal bin is open-ended, so nobody can say how many of these annotators are over 75.
- **D3CODE** — Released as balanced across age, but the last boundary is 50: 65 is not a question this release can answer.
- **DICES-350** — A 65+ bracket was collected at recruitment; the release carries only generation labels, so the terminal cell's 31 raters cannot be resolved against 65.
- **PRISM** — The alignment corpus built for pluralistic feedback stops at 65+, so none of its participants is identifiably over 75.
- **Measuring Hate Speech** — Exact ages, and still only 12 annotators at 75 or over, none past 81. Resolution without density.
- **SBIC** — Excludes 6 entries recorded as exactly 100; the data contain no ages between 72 and 99, so those are not credible. Counted in, SBIC would outrank every other pool.
- **NLPositionality** — The authors' endpoints (delphi-litw / toxicity-litw.apps.allenai.org) were already offline when the paper was written, and the mirror used instead (HuggingFace asaakyan/culture-safety, documented as an exact download and cross-checked 759/759 against the authors' processed release) has since gone private. The project page is also down. The published row -- 15 annotators at 65+, one at 75+ (an 89-year-old; next oldest 73), of 759 -- cannot currently be reverified by anyone, including us.

## Agreement with the paper

Every recomputable cell and caption fact matches what the paper prints.
