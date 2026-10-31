#!/usr/bin/env python3
"""Recompute Table 2 from the released raw data, and check it against print.

The paper's central empirical claim is that annotation and preference corpora
fail to record old age, in one of two ways: the brackets stop resolving, or the
pool runs out of people. That claim is a table of counts taken from other
people's datasets, which is exactly the kind of thing a reader should not have
to take on trust. This script derives every cell from source and diffs the
result against what the paper prints, so drift shows up as a failure rather
than as a reviewer's email.

Every count is over DISTINCT annotators, deduplicated by the release's own
annotator/participant ID. Rating rows are not annotators: POPQUORN politeness
has 25,000+ rating rows and 506 people, and counting rows would inflate every
cell by a different factor.

Where a release cannot answer a question, the answer is "?" and not zero.
DICES publishes generation labels, so how many of its raters are 65 or over is
unknowable from the release, which is a different fact from there being none.
Keeping those apart is the point of the table.

Usage: python3 audit_age.py [--json PATH] [--md PATH]
Requires: fetch_corpora.py to have run first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE.parent / "results"

UNKNOWN = "?"


# --------------------------------------------------------------------------
# What the paper prints. A mismatch here is a bug in the paper or in this file,
# and either way it needs a human before submission.
# --------------------------------------------------------------------------
PUBLISHED = {
    "POPQUORN politeness":  dict(recording="5-yr brackets", last_bin=">65",
                                 n_ge65=69, n_ge75=UNKNOWN),
    "POPQUORN offensiveness": dict(recording="5-yr brackets", last_bin=">65",
                                   n_ge65=29, n_ge75=UNKNOWN),
    "D3CODE":               dict(recording="3 bins to 50+", last_bin="50+",
                                 n_ge65=UNKNOWN, n_ge75=UNKNOWN),
    "DICES-350":            dict(recording="generation labels", last_bin="gen x+",
                                 n_ge65=UNKNOWN, n_ge75=UNKNOWN),
    "PRISM":                dict(recording="10-yr brackets", last_bin="65+",
                                 n_ge65=106, n_ge75=UNKNOWN),
    "Measuring Hate Speech": dict(recording="exact age", last_bin="none",
                                  n_ge65=203, n_ge75=12),
    "SBIC":                 dict(recording="exact age", last_bin="none",
                                 n_ge65=6, n_ge75=0),
    "NLPositionality":      dict(recording="exact age", last_bin="none",
                                 n_ge65=15, n_ge75=1),
}

# Side-facts asserted in the Table 2 caption and in Section 3.
PUBLISHED_EXTRAS = {
    "popquorn_politeness_total": 506,
    "mhsc_total_annotators": 7912,
    "mhsc_oldest": 81,
    "mhsc_n_ge70": 69,
    "mhsc_n_ge80": 4,
    "sbic_n_exactly_100": 6,
    "prism_total": 1500,
    "dices_terminal_cell_n": 31,
    "nlpositionality_total": 759,
    "nlpositionality_oldest": 89,
    "nlpositionality_next_oldest": 73,
}


def distinct(df: pd.DataFrame, id_col: str, age_col: str) -> pd.Series:
    """One age per annotator, indexed by annotator id."""
    return df.dropna(subset=[id_col]).groupby(id_col)[age_col].first()


def bracket_audit(path: Path, id_col: str, age_col: str, name: str,
                  ge65_labels: set[str]) -> dict:
    """A corpus that records age as brackets: counts are knowable only at the
    bracket boundaries the release chose."""
    df = pd.read_csv(path, usecols=[id_col, age_col])
    ages = distinct(df, id_col, age_col).astype(str)
    counts = ages.value_counts().to_dict()
    n65 = int(sum(v for k, v in counts.items() if k in ge65_labels))
    return {
        "dataset": name, "source": path.name, "mode": "brackets",
        "n_annotators": int(len(ages)),
        "bracket_counts": dict(sorted(counts.items())),
        "n_ge65": n65,
        # A bracket that starts at or below 65 and has no upper edge cannot
        # distinguish 75 from 95. That is the resolution failure itself.
        "n_ge75": UNKNOWN,
    }


def exact_audit(ages: pd.Series, name: str, source: str) -> dict:
    a = pd.to_numeric(ages, errors="coerce").dropna()
    return {
        "dataset": name, "source": source, "mode": "exact",
        "n_annotators": int(len(ages)),
        "n_with_age": int(len(a)),
        "age_min": int(a.min()), "age_max": int(a.max()),
        "n_ge65": int((a >= 65).sum()),
        "n_ge70": int((a >= 70).sum()),
        "n_ge75": int((a >= 75).sum()),
        "n_ge80": int((a >= 80).sum()),
        "ages_65_plus": sorted(int(x) for x in a[a >= 65]),
    }


# --------------------------------------------------------------------------
# Per-dataset auditors
# --------------------------------------------------------------------------

def audit_popquorn(kind: str) -> dict:
    path = DATA / f"popquorn_{kind}.csv"
    r = bracket_audit(path, "user_id", "age", f"POPQUORN {kind}",
                      ge65_labels={">65"})
    r["last_bin"] = ">65"
    r["recording"] = "5-yr brackets"
    r["note"] = ("The terminal bin is open-ended, so nobody can say how many "
                 "of these annotators are over 75.")
    return r


def audit_d3code() -> dict:
    r = bracket_audit(DATA / "d3code_raters.csv", "rater_id", "Age", "D3CODE",
                      ge65_labels=set())
    r["last_bin"] = "50+"
    r["recording"] = f"{len(r['bracket_counts'])} bins to 50+"
    # The last boundary is 50, so 65 is not a question the release can answer.
    r["n_ge65"] = UNKNOWN
    r["note"] = ("Released as balanced across age, but the last boundary is 50: "
                 "65 is not a question this release can answer.")
    return r


def audit_dices() -> dict:
    r = bracket_audit(DATA / "dices350.csv", "rater_id", "rater_age",
                      "DICES-350", ge65_labels=set())
    r["last_bin"] = "gen x+"
    r["recording"] = "generation labels"
    r["n_ge65"] = UNKNOWN
    r["terminal_cell_n"] = int(r["bracket_counts"].get("gen x+", 0))
    r["note"] = ("A 65+ bracket was collected at recruitment; the release "
                 "carries only generation labels, so the terminal cell's "
                 f"{r['terminal_cell_n']} raters cannot be resolved against 65.")
    return r


def audit_prism() -> dict:
    rows = [json.loads(l) for l in (DATA / "prism_survey.jsonl").open()]
    ages = pd.Series({r["user_id"]: str(r["age"]) for r in rows})
    counts = ages.value_counts().to_dict()
    n65 = int(sum(v for k, v in counts.items() if k.startswith("65+")))
    return {
        "dataset": "PRISM", "source": "prism_survey.jsonl", "mode": "brackets",
        "recording": "10-yr brackets", "last_bin": "65+",
        "n_annotators": int(len(ages)),
        "bracket_counts": dict(sorted(counts.items())),
        "n_ge65": n65, "n_ge75": UNKNOWN,
        "note": ("The alignment corpus built for pluralistic feedback stops at "
                 "65+, so none of its participants is identifiably over 75."),
    }


def audit_mhsc() -> dict:
    df = pd.read_parquet(DATA / "mhsc.parquet",
                         columns=["annotator_id", "annotator_age"])
    r = exact_audit(distinct(df, "annotator_id", "annotator_age"),
                    "Measuring Hate Speech", "mhsc.parquet")
    r["recording"] = "exact age"
    r["last_bin"] = "none"
    r["note"] = ("Exact ages, and still only "
                 f"{r['n_ge75']} annotators at 75 or over, none past "
                 f"{r['age_max']}. Resolution without density.")
    return r


def audit_sbic() -> dict:
    parts = [pd.read_csv(DATA / f, usecols=["WorkerId", "annotatorAge"])
             for f in ("SBIC.v2.trn.csv", "SBIC.v2.dev.csv", "SBIC.v2.tst.csv")]
    df = pd.concat(parts, ignore_index=True)
    ages = distinct(df, "WorkerId", "annotatorAge")
    a = pd.to_numeric(ages, errors="coerce").dropna()
    # Six workers report exactly 100 and there is nothing between 72 and 99.
    # Treating those as real would put SBIC's elder density above MHSC's, so
    # the paper excludes them and says so.
    implausible = int((a == 100).sum())
    r = exact_audit(ages[pd.to_numeric(ages, errors="coerce") != 100],
                    "SBIC", "SBIC.v2.*.csv")
    r["recording"] = "exact age"
    r["last_bin"] = "none"
    r["n_exactly_100"] = implausible
    r["gap_ages_72_to_99"] = sorted(int(x) for x in a if 72 < x < 99)
    r["note"] = (f"Excludes {implausible} entries recorded as exactly 100; the "
                 "data contain no ages between 72 and 99, so those are not "
                 "credible. Counted in, SBIC would outrank every other pool.")
    return r


def audit_nlpositionality() -> dict:
    """Unreachable. Report the provenance rather than a number."""
    return {
        "dataset": "NLPositionality", "source": None, "mode": "unavailable",
        "recording": "exact age", "last_bin": "none",
        "n_ge65": None, "n_ge75": None,
        "status": "SOURCE UNAVAILABLE",
        "note": ("The authors' endpoints (delphi-litw / toxicity-litw"
                 ".apps.allenai.org) were already offline when the paper was "
                 "written, and the mirror used instead (HuggingFace "
                 "asaakyan/culture-safety, documented as an exact download and "
                 "cross-checked 759/759 against the authors' processed "
                 "release) has since gone private. The project page is also "
                 "down. The published row -- 15 annotators at 65+, one at 75+ "
                 "(an 89-year-old; next oldest 73), of 759 -- cannot currently "
                 "be reverified by anyone, including us."),
    }


AUDITORS = [
    lambda: audit_popquorn("politeness"),
    lambda: audit_popquorn("offensiveness"),
    audit_d3code,
    audit_dices,
    audit_prism,
    audit_mhsc,
    audit_sbic,
    audit_nlpositionality,
]


# --------------------------------------------------------------------------

def compare(rows: list[dict]) -> list[str]:
    """Diff recomputed cells against what the paper prints."""
    problems = []
    for r in rows:
        pub = PUBLISHED.get(r["dataset"])
        if pub is None:
            problems.append(f"{r['dataset']}: no published row to compare")
            continue
        if r.get("status") == "SOURCE UNAVAILABLE":
            continue
        for field, key in (("n_ge65", "n_ge65"), ("n_ge75", "n_ge75")):
            got, want = r.get(key), pub[field]
            if got != want:
                problems.append(
                    f"{r['dataset']}: {field} recomputed {got!r}, paper prints {want!r}")
        if r.get("last_bin") != pub["last_bin"]:
            problems.append(
                f"{r['dataset']}: last_bin {r.get('last_bin')!r} vs {pub['last_bin']!r}")
    return problems


def compare_extras(rows: list[dict]) -> list[str]:
    by = {r["dataset"]: r for r in rows}
    got = {
        "popquorn_politeness_total": by["POPQUORN politeness"]["n_annotators"],
        "mhsc_total_annotators": by["Measuring Hate Speech"]["n_annotators"],
        "mhsc_oldest": by["Measuring Hate Speech"]["age_max"],
        "mhsc_n_ge70": by["Measuring Hate Speech"]["n_ge70"],
        "mhsc_n_ge80": by["Measuring Hate Speech"]["n_ge80"],
        "sbic_n_exactly_100": by["SBIC"]["n_exactly_100"],
        "prism_total": by["PRISM"]["n_annotators"],
        "dices_terminal_cell_n": by["DICES-350"]["terminal_cell_n"],
    }
    out = []
    for k, v in got.items():
        want = PUBLISHED_EXTRAS[k]
        if v != want:
            out.append(f"caption fact {k}: recomputed {v}, paper prints {want}")
    return out


def render_md(rows: list[dict], problems: list[str]) -> str:
    L = ["# Table 2, recomputed from released raw data", "",
         "Counts are over distinct annotators, deduplicated by each release's",
         "own annotator/participant ID. `?` means the release cannot answer the",
         "question, which is not the same as zero.", "",
         "| Dataset | Age recording | Last bin | n>=65 | n>=75 | annotators |",
         "|---|---|---|---:|---:|---:|"]
    for r in rows:
        g = lambda k: "--" if r.get(k) is None else r.get(k)
        L.append(f"| {r['dataset']} | {r.get('recording','')} | "
                 f"{r.get('last_bin','')} | {g('n_ge65')} | {g('n_ge75')} | "
                 f"{r.get('n_annotators','--')} |")
    L += ["", "## Notes", ""]
    for r in rows:
        if r.get("note"):
            L.append(f"- **{r['dataset']}** — {r['note']}")
    L += ["", "## Agreement with the paper", ""]
    if problems:
        L.append(f"**{len(problems)} MISMATCH(ES):**")
        L += [f"- {p}" for p in problems]
    else:
        L.append("Every recomputable cell and caption fact matches what the "
                 "paper prints.")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(RESULTS / "audit_table2.json"))
    ap.add_argument("--md", default=str(RESULTS / "audit_table2.md"))
    args = ap.parse_args()

    missing = [n for n in ("popquorn_politeness.csv", "mhsc.parquet")
               if not (DATA / n).exists()]
    if missing:
        print(f"ERROR: missing {missing}. Run fetch_corpora.py first.")
        return 1

    rows = [f() for f in AUDITORS]
    problems = compare(rows) + compare_extras(rows)

    RESULTS.mkdir(exist_ok=True)
    Path(args.json).write_text(json.dumps(
        {"rows": rows, "published": PUBLISHED, "published_extras": PUBLISHED_EXTRAS,
         "mismatches": problems}, indent=1) + "\n")
    md = render_md(rows, problems)
    Path(args.md).write_text(md)
    print(md)

    unavailable = [r["dataset"] for r in rows if r.get("status")]
    if unavailable:
        print(f"NOTE: source unavailable for {unavailable} -- row not reverified.")
    if problems:
        print(f"\nFAIL: {len(problems)} mismatch(es) against the paper.")
        return 1
    print("\nOK: every recomputable cell matches the paper.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
