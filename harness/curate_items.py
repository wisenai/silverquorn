#!/usr/bin/env python3
"""Stage 1 of item curation: score every POPQUORN politeness item for
*generational contestedness* and dump a candidate list for manual reading.

Two families of evidence, combined:
  1. Empirical age signal (noisy: ~7 annotations/item):
     - old_young_diff: mean rating from annotators 60+ minus mean from 18-34
     - abs_rho: |Spearman| of bracket means vs bracket midpoints
  2. Register markers where politeness norms plausibly shifted across
     generations (slang, texting brevity, missing salutation/sign-off,
     ALL CAPS, emoticons, curt imperatives, effusive formality).

Output: harness/results/candidates_for_review.md (top ~150, quintile-covered)
for a human/Claude read; final 40 are hand-picked into select_items.py.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "harness" / "data" / "popquorn_politeness_raw.csv"

AGE_MID = {
    "18-24": 21, "25-29": 27, "30-34": 32, "35-39": 37, "40-44": 42,
    "45-49": 47, "50-54": 52, "54-59": 56.5, "60-64": 62, ">65": 70,
}
YOUNG = {"18-24", "25-29", "30-34"}
OLD = {"60-64", ">65"}

SLANG = re.compile(
    r"\b(dude|yo|bro|buddy|haha|hahaha|lol|jk|gonna|wanna|gotta|kinda|sorta|"
    r"sucks|suck|awesome|cool|hooky|whoopin|freakin|dang|heck|crap|screwed|"
    r"hot minute|killer|sweet offer|knuckleheads|hang out|hit me up|catch up|"
    r"what's up|whats up|wassup)\b", re.I)
TEXTING_ABBREV = re.compile(r"\b(b/c|bc|u r|thx|pls|plz|btw|fyi|asap|lmk|w/e|w/o)\b", re.I)
MULTI_PUNCT = re.compile(r"([!?])\1{1,}|!{2,}|\?{2,}")
ALL_CAPS_RUN = re.compile(r"\b[A-Z]{2,}(?:\s+[A-Z]{2,}){2,}\b")
EMOTICON = re.compile(r"[:;]-?[)(DPp]|<3")
GREETING = re.compile(
    r"^\s*(hi|hello|hey|dear|good (morning|afternoon|evening)|greetings)\b", re.I)
SIGNOFF = re.compile(
    r"(regards|sincerely|best|thanks|thank you|cheers|yours)[,.!\s]*\n?\s*$", re.I)
FORMAL = re.compile(
    r"\b(kindly|sincerely|i would appreciate|would you be so kind|"
    r"at your earliest convenience|please do not hesitate|i am pleased to|"
    r"i sincerely apologize|highly valued|your patience and understanding)\b", re.I)
PROFANITY = re.compile(
    r"\b(fuck|shit|bitch|asshole|dick|cunt|bastard|damn|hell|ass)\b", re.I)


def features(text):
    t = " ".join(text.split())
    lower_start = bool(re.match(r"^[a-z]", t))
    return {
        "slang": len(SLANG.findall(t)),
        "abbrev": len(TEXTING_ABBREV.findall(t)),
        "multi_punct": len(MULTI_PUNCT.findall(t)),
        "caps_run": len(ALL_CAPS_RUN.findall(text)),
        "emoticon": len(EMOTICON.findall(t)),
        "no_greeting": 0 if GREETING.search(text) else 1,
        "no_signoff": 0 if SIGNOFF.search(text.strip()) else 1,
        "lower_start": int(lower_start),
        "short": int(len(t) < 90),
        "formal": len(FORMAL.findall(t)),
        "profanity": len(PROFANITY.findall(t)),
        "len": len(t),
    }


def main():
    df = pd.read_csv(RAW, dtype={"instance_id": str})
    df = df.rename(columns={"politeness": "rating"})
    df["rating"] = df["rating"].astype(float)

    g = df.groupby("instance_id")["rating"]
    stats = pd.DataFrame({"mean": g.mean(), "sd": g.std(), "n": g.size()})
    stats["text"] = df.drop_duplicates("instance_id").set_index("instance_id")["text"]
    stats["quintile"] = pd.qcut(stats["mean"], 5, labels=[1, 2, 3, 4, 5])

    aged = df[df["age"].isin(AGE_MID)].copy()
    aged["mid"] = aged["age"].map(AGE_MID)

    young = aged[aged["age"].isin(YOUNG)].groupby("instance_id")["rating"].agg(["mean", "size"])
    old = aged[aged["age"].isin(OLD)].groupby("instance_id")["rating"].agg(["mean", "size"])
    both = young.join(old, lsuffix="_young", rsuffix="_old", how="inner")
    both = both[(both["size_young"] >= 2) & (both["size_old"] >= 2)]
    stats["oy_diff"] = both["mean_old"] - both["mean_young"]
    stats["n_young"] = both["size_young"]
    stats["n_old"] = both["size_old"]

    rho = {}
    for iid, sub in aged.groupby("instance_id"):
        bm = sub.groupby("mid")["rating"].mean()
        if len(bm) >= 4:
            r, _ = spearmanr(bm.index.to_numpy(), bm.to_numpy())
            rho[iid] = 0.0 if np.isnan(r) else r
    stats["rho"] = pd.Series(rho)

    feats = stats["text"].apply(features).apply(pd.Series)
    stats = stats.join(feats)

    # Generational-register score: informality markers + missing conventions,
    # small credit for effusive formality (contested from the other side).
    stats["register"] = (
        1.2 * stats["slang"].clip(0, 3)
        + stats["abbrev"].clip(0, 2)
        + 0.8 * stats["multi_punct"].clip(0, 2)
        + stats["caps_run"].clip(0, 2)
        + stats["emoticon"].clip(0, 2)
        + 0.6 * (stats["no_greeting"] & stats["lower_start"])
        + 0.4 * stats["no_signoff"]
        + 0.5 * stats["short"]
        + 0.8 * stats["formal"].clip(0, 2)
    )
    stats["age_signal"] = stats["oy_diff"].abs().fillna(0) + 0.5 * stats["rho"].abs().fillna(0)
    stats["score"] = stats["register"] + 2.0 * stats["age_signal"]

    eligible = stats[(stats["profanity"] == 0) & (stats["len"] <= 550) & (stats["n"] >= 5)]

    # top 30 per quintile by score -> 150 candidates for the manual read
    out = [
        "# Candidates for generational-contrast curation (manual read)",
        "",
        "Columns: quintile | overall mean | old-young diff (n_old/n_young) | rho | register score",
        "",
    ]
    for q in [1, 2, 3, 4, 5]:
        pool = eligible[eligible["quintile"] == q].sort_values("score", ascending=False).head(30)
        out.append(f"\n## Quintile {q}\n")
        for iid, r in pool.iterrows():
            t = " ".join(r["text"].split())
            oy = f"{r['oy_diff']:+.2f} ({int(r['n_old'])}/{int(r['n_young'])})" if pd.notna(r["oy_diff"]) else "n/a"
            out.append(
                f"### {iid} | q{q} | mean {r['mean']:.1f} | oy {oy} | "
                f"rho {r['rho'] if pd.notna(r['rho']) else 0:.2f} | reg {r['register']:.1f}")
            out.append(t)
            out.append("")
    path = ROOT / "harness" / "results" / "candidates_for_review.md"
    path.write_text("\n".join(out))
    print(f"wrote {path} ({sum(1 for l in out if l.startswith('### '))} candidates)")


if __name__ == "__main__":
    main()
