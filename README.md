# SilverQuorn — You Can't Simulate Grandma

Replication material for *You Can't Simulate Grandma: Elder User Simulation
Against a Record That Stops at 65*, Sathvik Subramanian (WisenAI; Newark
Academy), UserSim @ NeurIPS 2026.

The paper reports that simulated elder users cannot be validated, because the
human records they would be checked against fail past 65 in one of two ways:
either the age brackets stop resolving, or the pool runs out of people. This
repository holds the harness that measured what six models do in that region,
the dataset audit, the MHSC power analysis, and the aggregate results of a
small pilot with real elder raters.

## The name

POPQUORN is the politeness corpus this work runs on. The silver is hair.
SilverQuorn is POPQUORN asked of older adults — which is the one question
POPQUORN cannot answer, because everyone past 65 sits in a single open-ended
bracket.

## Setup

```bash
pip install -r requirements.txt
python3 harness/data/get_data.py      # fetches POPQUORN + MHSC
python3 harness/rebuild_items.py      # restores item text from your download
export OPENROUTER_API_KEY=sk-...      # only needed to re-run model queries
```

Nothing needs an API key if you only want to reproduce the analyses: every
model response is already in `harness/runs*/`, and every script reads those
caches before making a call.

## What is here

| Path | Contents |
|---|---|
| `harness/*.py` | Item selection, the four experiment runners, QC, analysis, figures |
| `harness/runs/` | Main sweep: 6 models x 8 persona conditions x 200 items x 3 samples |
| `harness/runs_inbin/` | In-bin control: 8 ages inside the >65 bin x 40 items |
| `harness/runs_floor8/` | Matched noise floor: 8 draws at a single age |
| `harness/runs_mhsc/` | MHSC validation: 8,820 ratings |
| `harness/audit/` | The seven-dataset age audit: fetches each corpus, recomputes Table 2 |
| `harness/results/` | Every number the paper reports, as JSON and Markdown |
| `pilot/` | Aggregate elder-rater results (see the limits below) |

Each `runs*/` record is one API call: model, persona age, item ID, raw
response, parsed rating, token counts, and cost. The runners are resumable, so
re-running one fills gaps rather than starting over.

### Reproducing the paper's numbers

```bash
python3 harness/audit/fetch_corpora.py  # the seven audited corpora
python3 harness/audit/audit_age.py      # Table 2, recomputed from source
python3 harness/qc.py                 # parse rates, refusals, cost
python3 harness/make_fig1.py --wide   # Figure 1 (+ model_curves.csv)
python3 harness/analyze_pilot.py      # Figure 2, pilot statistics
python3 harness/run_inbin.py          # Table 3, right-hand columns
python3 harness/run_floor8.py         # Table 3, left-hand columns
python3 harness/run_mhsc_val.py       # Appendix B
python3 harness/power_target.py       # the recruitment target in Section 5
```

`audit_age.py` recomputes every cell of Table 2 from each release's own raw
data, deduplicating by annotator ID, and diffs the result against the numbers
the paper prints. It exits non-zero on any mismatch. One row cannot be
reverified: NLPositionality's endpoints were already offline when the paper
was written, and the mirror used instead has since gone private. The script
reports that row as unavailable rather than silently carrying the number.

`make_fig1.py` needs `--wide`; without it you get a different panel geometry
that does not match the paper.

## What is deliberately not here

**Participant-level pilot data.** Participants consented to their ratings
appearing "always as group results" and to "anonymous group data" being shared
with other researchers. The participant x item matrix is neither, so it stays
out. `pilot/` carries per-item aggregates, the five participant means with
their age brackets — the same five numbers Figure 2 plots — and demographic
marginals. Those are deliberately marginals and not a joint table: with five
completers recruited through a community network, a bracket x country x
education cross-tab could single someone out, while the marginals cannot.
Everything the paper claims about the pilot is recomputable from these files.
`harness/analyze_pilot.py` ships as it ran and expects the withheld export, so
it will not execute here; the aggregates it produced are in `pilot/`.

**The source corpora.** POPQUORN and MHSC belong to their authors and carry
their own licences. `harness/data/get_data.py` fetches both and verifies
POPQUORN against the SHA-256 of the exact file the paper used.

**Item text.** POPQUORN's politeness items come from the Enron email corpus and
contain real people's names, email addresses and phone numbers. Those strings
are already public in POPQUORN, so withholding them protects nobody, but
republishing them keyed and tidied in a second place spreads them further for
no scientific gain. The item files carry IDs and statistics;
`harness/rebuild_items.py` restores the exact text from your own download.

## Cost

The full set of model queries cost about $12 at 2026 OpenRouter prices. The
caches in `harness/runs*/` mean you should not need to spend it. `config.json`
carries a budget ceiling; the runners stop rather than exceed it.

## Licence

Code is MIT (`LICENSE`). Model outputs in `harness/runs*/` and the aggregates
in `pilot/` are CC BY 4.0. POPQUORN and MHSC are not covered by either — they
remain under their own terms, which is why they are downloaded rather than
redistributed.
