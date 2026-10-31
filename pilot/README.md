# Pilot data (aggregate only)

Five adults aged 65+ rated the same 40 POPQUORN politeness items the models
rated, recruited through community networks. This directory holds the group
results. The participant x item matrix is not released.

## Why

The consent text participants agreed to says their ratings "will be combined
with everyone else's and may appear in a research paper, always as group
results", and that "anonymous group data may also be shared with other
researchers". Per-participant rows are not group results, so they stay out.
This is not a formality: with five completers drawn from one community
network, a row carrying age bracket, gender, country of residence, education
and English-use frequency could identify someone to a reader inside that
network, even with no name attached.

## Files

| File | Contents |
|---|---|
| `pilot_item_aggregates.csv` | Per item: rater count, mean, SD across the five completers |
| `pilot_participant_means.csv` | Five participant means with age bracket only, sorted |
| `pilot_marginals.json` | Completer count, bracket counts, country counts and means |
| `../harness/results/pilot_stats.json` | The full analysis output the paper draws on |

`pilot_participant_means.csv` carries the same five numbers Figure 2 plots, so
it reveals nothing the paper does not already show, and it lets you regenerate
that panel.
The demographic breakdowns are marginals and not a joint table, for the reason
above.

## Inclusion rule

Prespecified: a participant counts as a completer having rated at least 30 of
the 40 real items. Eight people started; five met the rule. One of the three
excluded rated 27 items and was in the 80–84 bracket, which is why the paper
reports no completer over 74 — the pilot reproduced the density failure it set
out to fix.

## Reproducing the paper's pilot numbers

- Median per-item SD **0.707** — median of the `sd` column in
  `pilot_item_aggregates.csv`.
- Pilot mean on these items **3.23**, brackets 65–69 **3.31** and 70–74 **3.18**
  — from `pilot_participant_means.csv` or `pilot_marginals.json`.
- Country means, India **3.32** and United States **3.10** — `pilot_marginals.json`.
- POPQUORN's `>65` mean recomputed on these 40 items, **3.34**, CI [2.91, 3.77]
  — from POPQUORN itself, via `harness/analyze_pilot.py`.

`harness/analyze_pilot.py` ships as it ran and reads the withheld export, so it
will not execute here. `harness/make_pilot_release.py` in the working repo is
the script that produced these files.
