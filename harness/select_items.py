#!/usr/bin/env python3
"""Item selection for SilverQuorn (HARNESS_SPEC.md §2, revised Jul 26).

Selection method (v2 — generational-contrast curation):
  Stage 1 (curate_items.py): score all 3,718 items on (a) empirical age
  signal — old(60+)-vs-young(18-34) mean difference and bracket-Spearman —
  and (b) generational register markers (slang, initialisms, ALL CAPS,
  emoticons, texting brevity, missing salutations, dated/effusive
  formality). Top 30 per politeness quintile -> 150 candidates.
  Stage 2 (this file): all 150 candidates were read manually; CURATED_40
  below are hand-picked for plausible generational politeness divergence,
  8 per quintile, content-filtered (no profanity/slurs/sexual/distressing).

Produces:
  harness/items_200.json — the 40 curated items + 160 stratified by
                           politeness-mean quintile (40 total per quintile)
  app/src/items_40.json  — the curated 40 + 2 anchors
  harness/results/human_age_profiles.csv — per-item per-bracket mean/SD/n
  harness/results/selection_report.md    — audit trail

Age brackets verified in the raw CSV: 18-24 ... 54-59, 60-64, >65 (open-
ended top bracket = the data-support ceiling; '54-59' is the label as it
appears in the file, sic).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260726
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "harness" / "data" / "popquorn_politeness_raw.csv"

AGE_MID = {
    "18-24": 21, "25-29": 27, "30-34": 32, "35-39": 37, "40-44": 42,
    "45-49": 47, "50-54": 52, "54-59": 56.5, "60-64": 62, ">65": 70,
}
BRACKET_ORDER = list(AGE_MID)

# Hand-curated after reading all 150 candidates in
# results/candidates_for_review.md. Tag = the generational register feature
# that motivated the pick. Pairs: (386, 2912) and (1698, 3179) are the same
# message in formal vs. texting register — direct generational probes.
CURATED_40 = {
    # Quintile 1 (least polite)
    "1729": "parental-authority teasing; old-young diff +3.0",
    "2205": "joking insult softened with 'j/k' — banter vs. rudeness",
    "2278": "passive-aggressive trailing ellipsis",
    "2303": "pure initialisms (FYI. ILY. SYT. TGIF.)",
    "547":  "direct prying personal question",
    "712":  "lowercase self-deprecating slang ('i'm a brat!')",
    "864":  "'LOL' + self-deprecation",
    "1698": "casual lowercase apology w/ typo (formal twin: 3179)",
    # Quintile 2
    "386":  "formal register version of 2912 (lion's heart pair)",
    "679":  "cold formal rebuke — formality as distance",
    "1273": "'the dude' + all-lowercase casual plan",
    "1277": "curt one-line request, no softening",
    "1340": "playful protest slang ('Don't shush me man')",
    "1643": "'FYI' + 'addy' abbreviation",
    "2232": "dropped-copula texting style ('We drinking beer tonight?')",
    "3424": "rambling chatty 'Hey guys… Anyways' register",
    # Quintile 3
    "127":  "formal wrapper (Dear/Sincerely) on casual content",
    "1424": "terse lowercase vacation note, initials sign-off",
    "1579": "'pls' abbreviation + initialism sign-off",
    "1672": "doubled question marks ('??')",
    "2168": "winking emoticon + guilt-tease",
    "2912": "texting register version of 386 (lion's heart pair)",
    "3004": "ALL-CAPS-as-default friendly letter",
    "3703": "'need ASAP' demand wrapped in thanks",
    # Quintile 4
    "30":   "no-pleasantries direct work question; diff +2.5",
    "507":  "all-lowercase lunch invite",
    "513":  "ALL-CAPS heartfelt affection + '!!!!!'",
    "1557": "effusive casual praise ('super job!… awesome')",
    "2644": "old-style spaced emoticon ': )'",
    "3015": "'you guys' + unpunctuated sign-off",
    "3179": "formal twin of 1698 (superstitious pair)",
    "3325": "mock-humble formality for a bar invite",
    # Quintile 5 (most polite)
    "205":  "'Dear Sir' full formality",
    "1141": "exclamation-dense warmth",
    "1751": "formal apology letter; elders rated it lower (-2.0)",
    "1961": "dated honorific 'madam'",
    "2748": "over-hedged apology ('I hope this does not bother you')",
    "3023": "presumptive courtesy ('Thank you for prioritizing this')",
    "3417": "stilted hyper-formality ('I would kindly like to share')",
    "3612": "'Thank you in advance' + 'at your earliest convenience'",
}

ANCHORS = {"anchor_polite": "3643", "anchor_rude": "2612"}


def main() -> None:
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(RAW, dtype={"instance_id": str})
    df = df.rename(columns={"politeness": "rating"})
    df["rating"] = df["rating"].astype(float)
    observed = sorted(set(df["age"]))
    assert set(AGE_MID) <= set(observed), f"age brackets changed: {observed}"

    g = df.groupby("instance_id")["rating"]
    stats = pd.DataFrame({"mean": g.mean(), "sd": g.std(), "n": g.size()})
    stats["text"] = df.drop_duplicates("instance_id").set_index("instance_id")["text"]
    stats["quintile"] = pd.qcut(stats["mean"], 5, labels=[1, 2, 3, 4, 5])

    missing = [i for i in list(CURATED_40) + list(ANCHORS.values())
               if i not in stats.index]
    assert not missing, f"curated ids not in data: {missing}"
    assert len(CURATED_40) == 40

    curated_q = stats.loc[list(CURATED_40)]["quintile"].astype(int)
    per_q = curated_q.value_counts().sort_index()

    # items_200: curated 40 + stratified fill to 40 per quintile
    ids200 = list(CURATED_40)
    for q in [1, 2, 3, 4, 5]:
        pool = stats[(stats["quintile"] == q) & ~stats.index.isin(ids200)].index.to_numpy()
        ids200.extend(str(i) for i in rng.choice(pool, size=40 - per_q.get(q, 0), replace=False))

    def item_row(iid, role="item", tag=None):
        r = stats.loc[iid]
        row = {"item_id": iid, "text": r["text"], "role": role,
               "popquorn_mean": round(float(r["mean"]), 3),
               "popquorn_sd": round(float(r["sd"]), 3),
               "popquorn_n": int(r["n"]), "quintile": int(r["quintile"])}
        if tag:
            row["curation_tag"] = tag
        return row

    (ROOT / "harness" / "items_200.json").write_text(
        json.dumps([item_row(i) for i in ids200], indent=1, ensure_ascii=False))

    app_items = [item_row(i, tag=t) for i, t in CURATED_40.items()]
    for role, iid in ANCHORS.items():
        app_items.append(item_row(iid, role=role))
    (ROOT / "app" / "src" / "items_40.json").write_text(
        json.dumps(app_items, indent=1, ensure_ascii=False))

    aged = df[df["age"].isin(AGE_MID)]
    prof = (aged.groupby(["instance_id", "age"])["rating"]
            .agg(["mean", "std", "size"]).reset_index()
            .rename(columns={"size": "n", "std": "sd"}))
    prof["age"] = pd.Categorical(prof["age"], categories=BRACKET_ORDER, ordered=True)
    prof.sort_values(["instance_id", "age"]).to_csv(
        ROOT / "harness" / "results" / "human_age_profiles.csv", index=False)

    report = [
        "# Item selection report (v2 — generational-contrast curation, Jul 26)", "",
        f"Source: {RAW.name} ({len(df)} annotations, {stats.shape[0]} items, "
        f"{df['user_id'].nunique()} annotators). Seed {SEED} (stratified fill only).",
        f"Age brackets observed (verbatim): {observed}.",
        "Data-support ceiling: open-ended '>65' bracket "
        f"({(df['age'] == '>65').sum()} annotations).",
        "",
        "Method: curate_items.py scored all items on empirical age signal "
        "(old-vs-young mean diff, bracket Spearman) x generational register "
        "markers; top-150 (30/quintile) read manually; 40 hand-picked for "
        "plausible generational politeness divergence, 8 per quintile, "
        "content-filtered. Register pairs included: (386, 2912), (1698, 3179).",
        "",
        f"items_40 per-quintile counts: {per_q.to_dict()}",
        f"anchors: polite={ANCHORS['anchor_polite']} "
        f"(mean {stats.loc[ANCHORS['anchor_polite']]['mean']:.2f}), "
        f"rude={ANCHORS['anchor_rude']} "
        f"(mean {stats.loc[ANCHORS['anchor_rude']]['mean']:.2f})",
        "",
        "## Curated 40 with rationale tags",
        *(f"- {i} (q{int(curated_q[i])}, mean "
          f"{stats.loc[i]['mean']:.1f}): {t}" for i, t in CURATED_40.items()),
        "",
        "## Interpretive caveat (for the paper)",
        "Items were SELECTED for hypothesized generational contrast; pilot "
        "results therefore speak to whether elder judgments diverge on "
        "divergence-prone items, not on a random POPQUORN sample. Say this "
        "explicitly in Methods.",
    ]
    (ROOT / "harness" / "results" / "selection_report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report[:14]))
    print(f"... report written; items_200={len(ids200)}, items_40={len(CURATED_40)}+2 anchors")


if __name__ == "__main__":
    main()
