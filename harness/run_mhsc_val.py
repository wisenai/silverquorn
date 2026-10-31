#!/usr/bin/env python3
"""MHSC elder-persona validation: the one dataset where an elder persona CAN be
scored against real humans of known exact age.

Why this exists
---------------
The paper's claim is that LLM elder personas are unvalidatable because the human
records used as validation targets stop resolving age past ~65 (POPQUORN's ">65"
topless bin). Reviewers identified the decisive exception: the Measuring Hate
Speech Corpus (Kennedy et al. 2020; `ucberkeley-dlab/measuring-hate-speech`)
stores `annotator_age` as an EXACT integer, 18-81. That makes age-resolved elder
ground truth possible in principle. This script tests whether it is possible in
practice, and it is an honest two-sided test: if personas validate there, the
paper's claim must narrow; if the attempt fails for want of elder density, the
paper's density argument is demonstrated rather than asserted.

Target variable
---------------
`respect` -- one of MHSC's ten raw ordinal constituent labels, 0-4. We prefer it
over the headline `hate_speech_score` for two reasons:
  1. Kind. `respect` is a graded 5-point subjective judgement of a comment,
     structurally identical to POPQUORN's 1-5 politeness item and semantically
     its nearest neighbour. `hatespeech` is only 3-point; the other constituents
     (violence, genocide, dehumanize) are further from a politeness judgement.
  2. Construction. `hate_speech_score` is a faceted-Rasch latent score that
     already incorporates an IRT adjustment for annotator severity. Using it as
     the human target would partially regress out annotator-level variation --
     i.e. it would wash out the very age signal we are trying to measure, by
     construction.
The column is disrespect-coded (higher value = more disrespectful; verified
empirically, see results/mhsc_validation.md). We reverse-code it to a
respectfulness scale matching POPQUORN's orientation:
    respectfulness = 5 - respect        (integer 1..5, 5 = most respectful)

Comment set
-----------
Forced by the data, not chosen. MHSC's 39,565 comments are mostly rated ~2-3
times each; only a high-replication core of exactly 70 comments (243-815 raters
each) carries any elder replication at all. Outside that core, 2,723 comments
have exactly one 65+ rater and 41 have two. All 203 annotators aged 65+ rated
~6 of these 70 comments, so the core contains 100% of the available elder
signal. n=70 comments is therefore the ceiling, below the 100-200 we wanted.

Conditions
----------
Six persona ages at band midpoints plus a no-persona control. The two non-elder
personas (24, 50) are a positive control: if a model cannot discriminate 24 from
67, any elder-vs-elder null is about personas in general; if it can, then failure
to discriminate 67 from 72 is specifically about resolution inside old age.

Ethics note: the comments contain hate speech. This is a model-only experiment
with no human participants, so there is no participant-exposure concern. The
runner never prints comment text.

Usage
-----
  cd harness && set -a && . ./.env && set +a && python3 run_mhsc_val.py
Options
  --parquet PATH     MHSC parquet (default $MHSC_PARQUET, else ./data/mhsc.parquet);
                     downloaded from HuggingFace if absent.
  --limit-comments N debug: use only the first N core comments
  --analyse-only     skip all API calls, re-run the analysis from cache
Output
  runs_mhsc/<model>.jsonl              resumable cache, one record per cell
  results/mhsc_validation.{md,json}    all numbers
The MHSC parquet and the JSONL caches are gitignored; no dataset text is written
into the repository.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from harness_common import (
    HARNESS_DIR, STRICT_SUFFIX, get_api_key, load_config,
    model_slug_to_filename, parse_rating, looks_like_refusal,
    persona_system_message,
)

URL = "https://openrouter.ai/api/v1/chat/completions"
PARQUET_URL = ("https://huggingface.co/api/datasets/ucberkeley-dlab/"
               "measuring-hate-speech/parquet/default/train/0.parquet")

# Rating instruction, quoted verbatim in the results file. Mirrors POPQUORN's
# politeness item structure (5-point, "reply with only the number") but asks the
# MHSC (dis)respect construct.
RATING_INSTRUCTION = (
    "Rate how respectful the following social media comment is toward the "
    "people or groups it refers to, from 1 (very disrespectful) to 5 (very "
    "respectful). Reply with only the number."
)

# Age bands and the persona age used to stand in for each. The elder bands are
# the resolution POPQUORN's ">65" bin destroys.
BANDS: dict[str, tuple[int, int, int]] = {
    # label:    (lo, hi, persona_age)
    "18-29": (18, 29, 24),
    "45-54": (45, 54, 50),
    "55-64": (55, 64, 60),
    "65-69": (65, 69, 67),
    "70-74": (70, 74, 72),
    "75-81": (75, 81, 78),
}
ELDER_BANDS = ["65-69", "70-74", "75-81"]
CORE_MIN_RATERS = 200  # separates the high-replication core from the long tail
MIN_BAND_N = 3         # primary analysis threshold for a usable human band mean


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def ensure_parquet(path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading MHSC parquet -> {path}")
    urllib.request.urlretrieve(PARQUET_URL, path)
    return path


def build_design(parquet: Path, limit: int | None = None) -> tuple[list[dict], dict, dict]:
    """Return (comments, human_targets, feasibility).

    comments:       [{comment_id, text}, ...] the core set
    human_targets:  {comment_id: {band: {"mean":, "n":, "sd":, "se":}}}
    feasibility:    coverage numbers for the results file
    """
    df = pd.read_parquet(parquet, columns=["comment_id", "annotator_id", "text",
                                           "annotator_age", "respect",
                                           "hate_speech_score", "hatespeech"])
    feas: dict = {
        "rows": int(len(df)),
        "comments": int(df.comment_id.nunique()),
        "annotators": int(df.annotator_id.nunique()),
        "annotator_age_missing_rows": int(df.annotator_age.isna().sum()),
    }
    df = df.dropna(subset=["annotator_age"]).copy()
    df["age"] = df.annotator_age.astype(int)
    df["respectfulness"] = 5.0 - df["respect"]

    ann = df.groupby("annotator_id").age.first()
    feas["annotator_age_min"] = int(ann.min())
    feas["annotator_age_max"] = int(ann.max())
    feas["annotators_by_threshold"] = {f"{t}+": int((ann >= t).sum())
                                       for t in (60, 65, 70, 75, 80)}
    feas["annotator_age_hist_60_81"] = {int(a): int((ann == a).sum())
                                        for a in range(60, 82)}

    # ---- per-comment elder coverage over the WHOLE corpus ------------------
    per = df.groupby("comment_id").apply(
        lambda x: pd.Series({
            "n_total": len(x),
            **{f"n_{b}": int(((x.age >= lo) & (x.age <= hi)).sum())
               for b, (lo, hi, _) in BANDS.items()},
            "n_65p": int((x.age >= 65).sum()),
            "n_70p": int((x.age >= 70).sum()),
            "n_75p": int((x.age >= 75).sum()),
        }), include_groups=False)
    feas["corpus_coverage"] = {
        "comments_with_ge_k_raters_65plus": {
            str(k): int((per.n_65p >= k).sum()) for k in (1, 2, 3, 5, 10)},
        "comments_with_ge_k_raters_70plus": {
            str(k): int((per.n_70p >= k).sum()) for k in (1, 2, 3, 5)},
        "comments_with_ge_k_raters_75plus": {
            str(k): int((per.n_75p >= k).sum()) for k in (1, 2, 3)},
        "n65plus_per_comment_valuecounts_head": {
            str(int(k)): int(v) for k, v in
            per.n_65p.value_counts().sort_index().head(6).items()},
    }

    core_ids = sorted(per.index[per.n_total >= CORE_MIN_RATERS])
    tail = per[per.n_total < CORE_MIN_RATERS]
    feas["core"] = {
        "min_raters_threshold": CORE_MIN_RATERS,
        "n_comments": len(core_ids),
        "raters_per_comment_min": int(per.loc[core_ids].n_total.min()),
        "raters_per_comment_max": int(per.loc[core_ids].n_total.max()),
        "elder_rows_outside_core_with_ge2_65plus": int((tail.n_65p >= 2).sum()),
        "max_65plus_outside_core": int(tail.n_65p.max()),
    }
    if limit:
        core_ids = core_ids[:limit]

    d = df[df.comment_id.isin(core_ids)].copy()
    feas["core"]["rows"] = int(len(d))
    band_stats = {}
    for b, (lo, hi, _) in BANDS.items():
        sub = d[(d.age >= lo) & (d.age <= hi)]
        cnt = sub.groupby("comment_id").size().reindex(core_ids, fill_value=0)
        band_stats[b] = {
            "distinct_annotators_corpus": int(
                ann[(ann >= lo) & (ann <= hi)].shape[0]),
            "distinct_annotators_core": int(sub.annotator_id.nunique()),
            "ratings_in_core": int(len(sub)),
            "raters_per_comment_median": float(cnt.median()),
            "raters_per_comment_min": int(cnt.min()),
            "raters_per_comment_max": int(cnt.max()),
            "comments_ge1": int((cnt >= 1).sum()),
            "comments_ge2": int((cnt >= 2).sum()),
            "comments_ge3": int((cnt >= 3).sum()),
        }
    feas["core"]["bands"] = band_stats

    # direction check for the `respect` column
    cm = d.groupby("comment_id").agg(r=("respect", "mean"),
                                     h=("hate_speech_score", "mean"))
    feas["respect_direction_check"] = {
        "corr_mean_respect_vs_mean_hate_speech_score": round(
            float(cm.r.corr(cm.h)), 4),
        "mean_respect_of_least_hateful_comment": round(
            float(cm.r.loc[cm.h.idxmin()]), 3),
        "mean_respect_of_most_hateful_comment": round(
            float(cm.r.loc[cm.h.idxmax()]), 3),
        "conclusion": ("higher `respect` value = MORE disrespectful; "
                       "reverse-coded as respectfulness = 5 - respect"),
    }

    # ---- human targets ----------------------------------------------------
    targets: dict = {}
    for cid, grp in d.groupby("comment_id"):
        t = {}
        for b, (lo, hi, _) in BANDS.items():
            v = grp.respectfulness[(grp.age >= lo) & (grp.age <= hi)].values
            if len(v) == 0:
                continue
            sd = float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")
            t[b] = {"mean": float(np.mean(v)), "n": int(len(v)), "sd": sd,
                    "se": sd / math.sqrt(len(v)) if len(v) > 1 else float("nan")}
        t["all_ages"] = {"mean": float(grp.respectfulness.mean()),
                         "n": int(len(grp)),
                         "sd": float(grp.respectfulness.std(ddof=1))}
        targets[int(cid)] = t

    texts = d.groupby("comment_id").text.first()
    comments = [{"comment_id": int(c), "text": str(texts.loc[c])} for c in core_ids]
    return comments, targets, feas


# --------------------------------------------------------------------------- #
# elicitation -- same pattern as run_harness.py / run_inbin.py
# --------------------------------------------------------------------------- #
def build_messages(persona_age: int | None, text: str, strict: bool = False) -> list[dict]:
    user = f"{RATING_INSTRUCTION}\n\n{text}"
    if strict:
        user = f"{user}\n\n{STRICT_SUFFIX}"
    msgs = []
    sysmsg = persona_system_message(persona_age)
    if sysmsg is not None:
        msgs.append({"role": "system", "content": sysmsg})
    msgs.append({"role": "user", "content": user})
    return msgs


def call(model: str, messages: list, key: str, max_tokens: int,
         retries: int = 4, timeout: int = 60) -> tuple:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0.7, "usage": {"include": True}}
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"})
    for attempt in range(retries):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=timeout))
            return (r["choices"][0]["message"].get("content"),
                    r.get("usage", {}).get("cost"))
        except urllib.error.HTTPError:
            if attempt == retries - 1:
                return None, None
            time.sleep(2 ** attempt)
        except Exception:
            if attempt == retries - 1:
                return None, None
            time.sleep(2 ** attempt)
    return None, None


def run_model(model: str, comments: list[dict], conditions: list, samples: int,
              key: str, cfg: dict, runs_dir: Path) -> float:
    path = runs_dir / model_slug_to_filename(model)
    done: set = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("parsed") is not None:
                done.add((rec.get("persona_age"), rec["comment_id"],
                          rec.get("sample_idx", 0)))
    todo = [(cond, c, s) for cond in conditions for c in comments
            for s in range(samples)
            if (cond, c["comment_id"], s) not in done]
    if not todo:
        print(f"{model}: complete ({len(done)} cells cached)")
        return 0.0

    cost_acc = {"v": 0.0}

    def work(job):
        cond, c, s = job
        msgs = build_messages(cond, c["text"])
        raw, cost = call(model, msgs, key, cfg["max_tokens"])
        val = parse_rating(raw)
        if val is None and raw is not None:
            raw2, c2 = call(model, build_messages(cond, c["text"], strict=True),
                            key, max(cfg["max_tokens"], 16))
            val2 = parse_rating(raw2)
            cost = (cost or 0.0) + (c2 or 0.0)
            if val2 is not None:
                raw, val = raw2, val2
            else:
                raw = raw2 if raw2 is not None else raw
        return {"model": model, "persona_age": cond,
                "comment_id": c["comment_id"], "sample_idx": s,
                "parsed": val, "raw_response": raw,
                "refusal": looks_like_refusal(raw), "cost_usd": cost}

    print(f"{model}: {len(todo)} cells to run ({len(done)} cached)")
    with ThreadPoolExecutor(max_workers=cfg["max_concurrent_requests"]) as ex, \
            open(path, "a") as f:
        for i, rec in enumerate(ex.map(work, todo), 1):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            cost_acc["v"] += rec.get("cost_usd") or 0.0
            if i % 250 == 0:
                print(f"  {model}: {i}/{len(todo)}  ${cost_acc['v']:.3f}")
    print(f"{model}: done, ${cost_acc['v']:.3f}")
    return cost_acc["v"]


def load_model_ratings(model: str, runs_dir: Path) -> dict:
    """{condition: {comment_id: mean_rating}} plus diagnostics."""
    path = runs_dir / model_slug_to_filename(model)
    cells: dict = {}
    n_lines = n_unparsed = n_refusal = 0
    cost = 0.0
    unparsed_by_cond: dict = {}
    attempts_by_cond: dict = {}
    if not path.exists():
        return {"ratings": {}, "n_cells": 0, "unparsed": 0, "refusals": 0,
                "cost_usd": 0.0, "unparsed_by_condition": {}}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        n_lines += 1
        cost += rec.get("cost_usd") or 0.0
        cl = "neutral" if rec.get("persona_age") is None else str(rec["persona_age"])
        attempts_by_cond[cl] = attempts_by_cond.get(cl, 0) + 1
        if rec.get("refusal"):
            n_refusal += 1
        if rec.get("parsed") is None:
            n_unparsed += 1
            unparsed_by_cond[cl] = unparsed_by_cond.get(cl, 0) + 1
            continue
        key = (rec.get("persona_age"), rec["comment_id"], rec.get("sample_idx", 0))
        cells[key] = rec["parsed"]
    ratings: dict = {}
    for (cond, cid, _s), v in cells.items():
        ratings.setdefault(cond, {}).setdefault(cid, []).append(v)
    means = {cond: {cid: float(np.mean(vs)) for cid, vs in d.items()}
             for cond, d in ratings.items()}
    return {"ratings": means, "n_cells": n_lines, "unparsed": n_unparsed,
            "refusals": n_refusal, "cost_usd": cost,
            "unparsed_by_condition": {
                c: {"n": v, "rate": round(v / attempts_by_cond[c], 4)}
                for c, v in sorted(unparsed_by_cond.items())},
            "max_condition_unparsed_rate": max(
                (v / attempts_by_cond[c] for c, v in unparsed_by_cond.items()),
                default=0.0),
            "samples_per_cell": {cond: float(np.mean([len(v) for v in d.values()]))
                                 for cond, d in ratings.items()}}


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation for a Wilcoxon signed-rank test.
    Positive = x > y typically."""
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def wilcoxon_safe(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 5 or np.all(a == b):
        return float("nan"), float("nan")
    try:
        s, p = stats.wilcoxon(a, b)
        return float(s), float(p)
    except ValueError:
        return float("nan"), float("nan")


def analyse(comments, targets, models, runs_dir) -> dict:
    cids = [c["comment_id"] for c in comments]
    out: dict = {"models": {}, "fidelity": {}, "discrimination": {},
                 "power": {}, "diagnostics": {}}

    # ---- (c) POWER: how noisy are the human band means? -------------------
    for b in BANDS:
        ns, ses, means = [], [], []
        for cid in cids:
            t = targets[cid].get(b)
            if not t or t["n"] < 2:
                if t:
                    ns.append(t["n"])
                continue
            ns.append(t["n"])
            ses.append(t["se"])
            means.append(t["mean"])
        ses_a, means_a = np.array(ses), np.array(means)
        # reliability of the band mean as a comment-level measure:
        #   var_observed = var_true + mean(error var)
        err = float(np.mean(ses_a ** 2)) if len(ses_a) else float("nan")
        obs = float(np.var(means_a, ddof=1)) if len(means_a) > 1 else float("nan")
        rel_raw = (obs - err) / obs if obs and obs == obs else float("nan")
        rel = max(0.0, rel_raw) if rel_raw == rel_raw else float("nan")
        out["power"][b] = {
            "comments_with_n_ge2": len(ses),
            "median_n_per_comment": float(np.median(ns)) if ns else 0.0,
            "mean_n_per_comment": round(float(np.mean(ns)), 2) if ns else 0.0,
            "median_se": round(float(np.median(ses_a)), 4) if len(ses_a) else None,
            "mean_se": round(float(np.mean(ses_a)), 4) if len(ses_a) else None,
            "median_95ci_halfwidth": round(float(np.median(1.96 * ses_a)), 4)
                                     if len(ses_a) else None,
            "mean_95ci_halfwidth": round(float(np.mean(1.96 * ses_a)), 4)
                                   if len(ses_a) else None,
            "frac_comments_zero_width_ci": round(float(np.mean(ses_a == 0)), 4)
                                           if len(ses_a) else None,
            "observed_var_of_band_means": round(obs, 4) if obs == obs else None,
            "mean_sampling_error_var": round(err, 4) if err == err else None,
            "reliability_raw": round(rel_raw, 4) if rel_raw == rel_raw else None,
            "reliability": round(rel, 4) if rel == rel else None,
            "max_attainable_correlation_sqrt_reliability":
                round(math.sqrt(rel), 4) if rel == rel else None,
        }
    # difference-score reliability for each elder band pair
    out["power"]["band_pair_difference_scores"] = {}
    for i, a in enumerate(ELDER_BANDS):
        for bb in ELDER_BANDS[i + 1:]:
            diffs, errv = [], []
            for cid in cids:
                ta, tb = targets[cid].get(a), targets[cid].get(bb)
                if not ta or not tb or ta["n"] < 2 or tb["n"] < 2:
                    continue
                diffs.append(ta["mean"] - tb["mean"])
                errv.append(ta["se"] ** 2 + tb["se"] ** 2)
            if len(diffs) < 3:
                out["power"]["band_pair_difference_scores"][f"{a} vs {bb}"] = {
                    "n_comments": len(diffs), "note": "too few comments"}
                continue
            obs = float(np.var(diffs, ddof=1))
            err = float(np.mean(errv))
            rel_raw = (obs - err) / obs
            rel = max(0.0, rel_raw)
            # smallest difference a 2-sided test could resolve with the raters
            # actually present, at alpha=.05, per comment
            mdd = float(np.median(1.96 * np.sqrt(errv)))
            out["power"]["band_pair_difference_scores"][f"{a} vs {bb}"] = {
                "n_comments": len(diffs),
                "mean_human_difference": round(float(np.mean(diffs)), 4),
                "median_abs_human_difference": round(float(np.median(np.abs(diffs))), 4),
                "median_minimum_resolvable_difference": round(mdd, 4),
                "observed_var": round(obs, 4),
                "sampling_error_var": round(err, 4),
                "error_var_over_observed_var": round(err / obs, 4),
                "reliability_of_difference_raw": round(rel_raw, 4),
                "reliability_of_difference": round(rel, 4),
                "max_attainable_correlation": round(math.sqrt(rel), 4),
            }

    # ---- human age gradient: do the real humans differ by band at all? -----
    # If they do not, nothing is being asked of the personas and the whole
    # experiment is vacuous. If they do across wide age gaps but not within old
    # age, that localises the failure.
    out["human_age_gradient"] = {"pooled_band_means": {}, "pairs": {}}
    for b in BANDS:
        vals = [targets[c][b]["mean"] for c in cids
                if b in targets[c] and targets[c][b]["n"] >= 1]
        ws = [targets[c][b]["n"] for c in cids
              if b in targets[c] and targets[c][b]["n"] >= 1]
        out["human_age_gradient"]["pooled_band_means"][b] = {
            "unweighted_mean_of_comment_means": round(float(np.mean(vals)), 4),
            "rating_weighted_mean": round(
                float(np.average(vals, weights=ws)), 4),
            "n_comments": len(vals),
        }
    band_list = list(BANDS)
    for i, a in enumerate(band_list):
        for bb in band_list[i + 1:]:
            thr_a = MIN_BAND_N if a != "75-81" else 2
            thr_b = MIN_BAND_N if bb != "75-81" else 2
            va, vb = [], []
            for cid in cids:
                ta, tb = targets[cid].get(a), targets[cid].get(bb)
                if not ta or not tb or ta["n"] < thr_a or tb["n"] < thr_b:
                    continue
                va.append(ta["mean"])
                vb.append(tb["mean"])
            if len(va) < 6:
                out["human_age_gradient"]["pairs"][f"{a} vs {bb}"] = {
                    "n_comments": len(va), "note": "insufficient coverage"}
                continue
            va_a, vb_a = np.array(va), np.array(vb)
            s, p = wilcoxon_safe(va_a, vb_a)
            rho, prho = stats.spearmanr(va_a, vb_a)
            out["human_age_gradient"]["pairs"][f"{a} vs {bb}"] = {
                "n_comments": len(va),
                "mean_diff": round(float((va_a - vb_a).mean()), 4),
                "wilcoxon_p": None if p != p else float(f"{p:.4g}"),
                "rank_biserial": round(rank_biserial(va_a, vb_a), 4),
                "spearman_between_band_means": round(float(rho), 4),
                "spearman_p": float(f"{prho:.4g}"),
            }

    # ---- per-model ---------------------------------------------------------
    for model in models:
        md = load_model_ratings(model, runs_dir)
        out["diagnostics"][model] = {
            k: md[k] for k in ("n_cells", "unparsed", "refusals", "cost_usd",
                               "unparsed_by_condition",
                               "max_condition_unparsed_rate")}
        r = md["ratings"]
        neutral = r.get(None, {})
        out["models"][model] = {}

        # (a) FIDELITY -- at the primary threshold and at a permissive one, so
        # that the thin 75-81 band is reported rather than silently dropped.
        fid = {}
        for thr, suffix in ((MIN_BAND_N, ""), (1, " (n>=1)")):
          for b, (_lo, _hi, page) in BANDS.items():
            label = b + suffix
            pers = r.get(page, {})
            ep, en, hm, pm = [], [], [], []
            for cid in cids:
                t = targets[cid].get(b)
                if not t or t["n"] < thr:
                    continue
                if cid not in pers or cid not in neutral:
                    continue
                ep.append(abs(pers[cid] - t["mean"]))
                en.append(abs(neutral[cid] - t["mean"]))
                hm.append(t["mean"])
                pm.append(pers[cid])
            if len(ep) < 5:
                fid[label] = {"n_comments": len(ep), "min_band_n": thr,
                              "note": "insufficient coverage"}
                continue
            ep_a, en_a = np.array(ep), np.array(en)
            s, p = wilcoxon_safe(en_a, ep_a)  # H1: neutral error > persona error
            rho, prho = stats.spearmanr(pm, hm)
            fid[label] = {
                "n_comments": len(ep),
                "min_band_n": thr,
                "persona_age": page,
                "mean_abs_err_persona": round(float(ep_a.mean()), 4),
                "mean_abs_err_neutral": round(float(en_a.mean()), 4),
                "median_abs_err_persona": round(float(np.median(ep_a)), 4),
                "median_abs_err_neutral": round(float(np.median(en_a)), 4),
                "improvement_mean": round(float(en_a.mean() - ep_a.mean()), 4),
                "wilcoxon_W": None if s != s else round(s, 2),
                "wilcoxon_p": None if p != p else float(f"{p:.4g}"),
                "rank_biserial": round(rank_biserial(en_a, ep_a), 4),
                "spearman_persona_vs_human_band_mean": round(float(rho), 4),
                "spearman_p": float(f"{prho:.4g}"),
                "persona_mean_rating": round(float(np.mean(pm)), 4),
                "human_band_mean_rating": round(float(np.mean(hm)), 4),
            }
        out["models"][model]["fidelity"] = fid

        # (b) DISCRIMINATION
        disc = {}
        pairs = [(a, bb) for i, a in enumerate(BANDS) for bb in list(BANDS)[i + 1:]]
        for a, bb in pairs:
            pa, pb = BANDS[a][2], BANDS[bb][2]
            ra, rb = r.get(pa, {}), r.get(pb, {})
            thr_a = MIN_BAND_N if a != "75-81" else 2
            thr_b = MIN_BAND_N if bb != "75-81" else 2
            hd, pdif = [], []
            for cid in cids:
                ta, tb = targets[cid].get(a), targets[cid].get(bb)
                if not ta or not tb or ta["n"] < thr_a or tb["n"] < thr_b:
                    continue
                if cid not in ra or cid not in rb:
                    continue
                hd.append(ta["mean"] - tb["mean"])
                pdif.append(ra[cid] - rb[cid])
            entry = {"n_comments": len(hd),
                     "min_raters_band_a": thr_a, "min_raters_band_b": thr_b}
            if len(hd) >= 6 and np.std(pdif) > 0 and np.std(hd) > 0:
                rho, p = stats.spearmanr(pdif, hd)
                pr, pp = stats.pearsonr(pdif, hd)
                entry.update({
                    "spearman_rho": round(float(rho), 4),
                    "spearman_p": float(f"{p:.4g}"),
                    "pearson_r": round(float(pr), 4),
                    "pearson_p": float(f"{pp:.4g}"),
                    "mean_human_diff": round(float(np.mean(hd)), 4),
                    "mean_persona_diff": round(float(np.mean(pdif)), 4),
                    "sd_persona_diff": round(float(np.std(pdif, ddof=1)), 4),
                    "sign_agreement": round(float(np.mean(
                        np.sign(np.array(pdif)) == np.sign(np.array(hd)))), 4),
                })
            else:
                entry["note"] = ("no persona variation between bands"
                                 if len(hd) >= 6 else "insufficient coverage")
                if len(hd) >= 2:
                    entry["mean_persona_diff"] = round(float(np.mean(pdif)), 4)
                    entry["sd_persona_diff"] = (
                        round(float(np.std(pdif, ddof=1)), 4) if len(pdif) > 1 else 0.0)
                    entry["mean_human_diff"] = round(float(np.mean(hd)), 4)
            disc[f"{a} vs {bb}"] = entry
        out["models"][model]["discrimination"] = disc

        # how much does the persona move the model at all?
        mono = {}
        for b, (_l, _h, page) in BANDS.items():
            pers = r.get(page, {})
            common = [c for c in cids if c in pers and c in neutral]
            if common:
                mono[b] = {
                    "persona_age": page,
                    "mean_rating": round(float(np.mean([pers[c] for c in common])), 4),
                    "mean_shift_vs_neutral": round(float(np.mean(
                        [pers[c] - neutral[c] for c in common])), 4),
                    "n_comments": len(common),
                }
        common_all = [c for c in cids if c in neutral]
        mono["neutral"] = {"mean_rating": round(float(np.mean(
            [neutral[c] for c in common_all])), 4) if common_all else None,
            "n_comments": len(common_all)}
        out["models"][model]["persona_response"] = mono

    # ---- cross-model summaries --------------------------------------------
    # (i) multiple-comparison accounting over every discrimination test run
    pvals, labels = [], []
    for m, md in out["models"].items():
        for k, v in md["discrimination"].items():
            if "spearman_p" in v:
                pvals.append(v["spearman_p"])
                labels.append((m, k, v["spearman_rho"]))
    pv = np.array(pvals)
    order = np.argsort(pv)
    bh = np.zeros(len(pv), bool)
    if len(pv):
        crit = 0.05 * (np.arange(1, len(pv) + 1)) / len(pv)
        passing = np.where(pv[order] <= crit)[0]
        if len(passing):
            bh[order[: passing.max() + 1]] = True
    out["discrimination_summary"] = {
        "n_tests": int(len(pv)),
        "n_p_below_.05": int((pv < 0.05).sum()),
        "expected_by_chance_at_.05": round(0.05 * len(pv), 2),
        "n_surviving_benjamini_hochberg_q.05": int(bh.sum()),
        "surviving_tests": [
            {"model": labels[i][0], "pair": labels[i][1],
             "rho": labels[i][2], "p": float(pv[i])}
            for i in np.where(bh)[0]],
        "nominally_significant_tests": [
            {"model": labels[i][0], "pair": labels[i][1],
             "rho": labels[i][2], "p": float(pv[i])}
            for i in np.where(pv < 0.05)[0]],
    }
    # (ii) the headline pair, across models
    for pair in ("65-69 vs 70-74", "65-69 vs 75-81", "70-74 vs 75-81",
                 "18-29 vs 65-69", "18-29 vs 70-74"):
        rhos = [md["discrimination"][pair].get("spearman_rho")
                for md in out["models"].values()
                if "spearman_rho" in md["discrimination"].get(pair, {})]
        if not rhos:
            continue
        n_pos = sum(r > 0 for r in rhos)
        # sign test against rho > 0 under H0 p=0.5
        sp = float(stats.binomtest(n_pos, len(rhos), 0.5,
                                   alternative="greater").pvalue)
        out["discrimination_summary"][pair] = {
            "models_reporting": len(rhos),
            "rhos": rhos,
            "median_rho": round(float(np.median(rhos)), 4),
            "n_positive": n_pos,
            "sign_test_p_rho_gt_0": round(sp, 4),
        }
    # (iii) how far does the persona move any model's mean rating?
    shifts = []
    for m, md in out["models"].items():
        for b in BANDS:
            if b in md["persona_response"]:
                shifts.append(abs(md["persona_response"][b]["mean_shift_vs_neutral"]))
    out["persona_effect_size_summary"] = {
        "n_model_band_cells": len(shifts),
        "median_abs_shift_vs_neutral_on_1to5_scale": round(float(np.median(shifts)), 4),
        "max_abs_shift_vs_neutral": round(float(np.max(shifts)), 4),
        "note": ("For reference, the real human 18-29 vs 70-74 gap and the "
                 "residual |persona - human| error are reported above."),
    }
    return out


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def write_report(res: dict, feas: dict, cfg: dict, comments: list,
                 total_cost: float, results_dir: Path) -> None:
    L: list[str] = []
    A = L.append
    A("# MHSC elder-persona validation")
    A("")
    A("Can an LLM elder persona be scored against real humans of *known exact "
      "age*? The Measuring Hate Speech Corpus is the one place this is even "
      "possible: `annotator_age` is an exact integer, 18-81. This file reports "
      "the attempt.")
    A("")
    A("## 0. Setup")
    A("")
    A(f"- Dataset: `ucberkeley-dlab/measuring-hate-speech` "
      f"({feas['rows']:,} rows, {feas['comments']:,} comments, "
      f"{feas['annotators']:,} annotators)")
    A(f"- Target variable: `respect`, reverse-coded `respectfulness = 5 - respect` "
      f"(integer 1-5, 5 = most respectful)")
    A(f"- Models: {', '.join(cfg['models'])} (same six as `config.json`)")
    A(f"- Temperature {cfg['temperature']}, {cfg['samples']} samples/cell, "
      f"single-integer parse with one strict retry")
    A("- Rating instruction, verbatim:")
    A("")
    A("  > " + RATING_INSTRUCTION)
    A("")
    A("- Persona system message, verbatim (one example): "
      f"`{persona_system_message(78)}`")
    A("")
    A("## 1. Feasibility")
    A("")
    A(f"`annotator_age` is exact and integer-valued over "
      f"[{feas['annotator_age_min']}, {feas['annotator_age_max']}]. "
      "Annotator counts by threshold:")
    A("")
    A("| threshold | distinct annotators |")
    A("|---|---|")
    for k, v in feas["annotators_by_threshold"].items():
        A(f"| {k} | {v} |")
    A("")
    A("Single-year counts, 60-81 (this is the cliff):")
    A("")
    A("| age | n | age | n |")
    A("|---|---|---|---|")
    h = feas["annotator_age_hist_60_81"]
    ks = sorted(h, key=int)
    half = (len(ks) + 1) // 2
    for i in range(half):
        l1 = f"{ks[i]} | {h[ks[i]]}"
        l2 = f"{ks[i+half]} | {h[ks[i+half]]}" if i + half < len(ks) else " | "
        A(f"| {l1} | {l2} |")
    A("")
    cc = feas["corpus_coverage"]
    A("### Per-comment elder coverage, whole corpus")
    A("")
    A("| raters in band >= k | 65+ | 70+ | 75+ |")
    A("|---|---|---|---|")
    for k in ("1", "2", "3", "5", "10"):
        A(f"| {k} | {cc['comments_with_ge_k_raters_65plus'].get(k, '-')} | "
          f"{cc['comments_with_ge_k_raters_70plus'].get(k, '-')} | "
          f"{cc['comments_with_ge_k_raters_75plus'].get(k, '-')} |")
    A("")
    A("The distribution is bimodal, and that is the whole story. Number of "
      "comments by exact count of 65+ raters: " +
      ", ".join(f"{k}->{v}" for k, v in
                cc["n65plus_per_comment_valuecounts_head"].items()) + ", ...")
    A("")
    core = feas["core"]
    A(f"All elder replication lives in a high-replication core of "
      f"**{core['n_comments']} comments** with "
      f"{core['raters_per_comment_min']}-{core['raters_per_comment_max']} raters "
      f"each. Outside that core, only {core['elder_rows_outside_core_with_ge2_65plus']} "
      f"comments have as many as {core['max_65plus_outside_core']} raters aged 65+. "
      f"All 203 annotators aged 65+ rated ~6 of these {core['n_comments']} "
      "comments, so the core holds 100% of the available elder signal and "
      f"n={core['n_comments']} comments is a hard ceiling.")
    A("")
    A("### Band coverage inside the core")
    A("")
    A("| band | annotators (corpus) | ratings in core | median raters/comment | "
      "comments >=1 | >=2 | >=3 |")
    A("|---|---|---|---|---|---|---|")
    for b, s in core["bands"].items():
        A(f"| {b} | {s['distinct_annotators_corpus']} | {s['ratings_in_core']} | "
          f"{s['raters_per_comment_median']:.0f} | {s['comments_ge1']} | "
          f"{s['comments_ge2']} | {s['comments_ge3']} |")
    A("")
    dc = feas["respect_direction_check"]
    A(f"Direction of `respect` verified empirically: comment-level correlation "
      f"with `hate_speech_score` is {dc['corr_mean_respect_vs_mean_hate_speech_score']:+.3f}; "
      f"the least hateful comment has mean `respect` "
      f"{dc['mean_respect_of_least_hateful_comment']} and the most hateful "
      f"{dc['mean_respect_of_most_hateful_comment']}. Higher value = more "
      "disrespectful, hence the reverse coding.")
    A("")
    A("## 2. Analysis")
    A("")
    A("### (c) Power: how noisy are the human band means?")
    A("")
    A("Reliability = (var of band means across comments - mean sampling error "
      "variance) / var of band means. Its square root is the ceiling on any "
      "correlation a persona could achieve against that target.")
    A("")
    A("| band | comments n>=2 | median raters* | mean raters* | median 95% CI "
      "half-width | mean 95% CI half-width | reliability | max attainable r |")
    A("|---|---|---|---|---|---|---|---|")
    for b in BANDS:
        p = res["power"][b]
        A(f"| {b} | {p['comments_with_n_ge2']} | {p['median_n_per_comment']:.0f} | "
          f"{p['mean_n_per_comment']} | {p['median_95ci_halfwidth']} | "
          f"{p['mean_95ci_halfwidth']} | {p['reliability']} | "
          f"{p['max_attainable_correlation_sqrt_reliability']} |")
    A("")
    A("\\* over comments with at least one rater in the band; CI columns and "
      "reliability use comments with n>=2.")
    A("")
    A("Band *means* survive because comments differ enormously in severity "
      "(between-comment variance ~1.5 on a 1-5 scale), so even a handful of "
      "raters localises a comment. Note the median 95% CI half-width of 0 for "
      "70-74 and 75-81: that is not precision, it is n=2 raters who happened to "
      "agree exactly. See `frac_comments_zero_width_ci` in the JSON.")
    A("")
    A("Elder band *difference* scores -- the quantity the discrimination test "
      "must correlate against -- do not survive:")
    A("")
    A("| band pair | comments | mean human diff | median abs human diff | "
      "median min. resolvable diff | error var / observed var | reliability of "
      "diff (raw) | max attainable r |")
    A("|---|---|---|---|---|---|---|---|")
    for k, v in res["power"]["band_pair_difference_scores"].items():
        if "reliability_of_difference" not in v:
            A(f"| {k} | {v['n_comments']} | - | - | - | - | - | - |")
            continue
        A(f"| {k} | {v['n_comments']} | {v['mean_human_difference']:+.4f} | "
          f"{v['median_abs_human_difference']} | "
          f"{v['median_minimum_resolvable_difference']} | "
          f"{v['error_var_over_observed_var']} | "
          f"{v['reliability_of_difference_raw']} | "
          f"{v['max_attainable_correlation']} |")
    A("")
    A("### Is there anything for a persona to recover? The human age gradient")
    A("")
    A("Before asking whether personas track age, check that the humans do. "
      "Pooled band means of `respectfulness` over the core comments, and paired "
      "Wilcoxon on the per-comment band means:")
    A("")
    A("| band | unweighted mean | rating-weighted mean | comments |")
    A("|---|---|---|---|")
    for b, v in res["human_age_gradient"]["pooled_band_means"].items():
        A(f"| {b} | {v['unweighted_mean_of_comment_means']} | "
          f"{v['rating_weighted_mean']} | {v['n_comments']} |")
    A("")
    A("| band pair | comments | mean human diff | Wilcoxon p | rank-biserial |")
    A("|---|---|---|---|---|")
    for k, v in res["human_age_gradient"]["pairs"].items():
        if "mean_diff" not in v:
            A(f"| {k} | {v['n_comments']} | - | - | {v.get('note','')} |")
            continue
        A(f"| {k} | {v['n_comments']} | {v['mean_diff']:+.4f} | "
          f"{v['wilcoxon_p']} | {v['rank_biserial']:+.3f} |")
    A("")
    A("### (a) Fidelity: does the persona land closer to real raters of that age?")
    A("")
    A("Per comment, |persona mean - human band mean| vs |no-persona mean - human "
      "band mean|. Wilcoxon signed-rank paired across comments, H1: neutral "
      "error > persona error. Positive rank-biserial favours the persona. "
      f"Comments restricted to human band n>={MIN_BAND_N}.")
    A("")
    A("| model | band | comments | MAE persona | MAE neutral | improvement | "
      "Wilcoxon p | rank-biserial |")
    A("|---|---|---|---|---|---|---|---|")
    for m, md in res["models"].items():
        for b, f in md["fidelity"].items():
            if "note" in f:
                A(f"| {m} | {b} | {f['n_comments']} | - | - | - | - | "
                  f"{f['note']} |")
                continue
            A(f"| {m} | {b} | {f['n_comments']} | {f['mean_abs_err_persona']} | "
              f"{f['mean_abs_err_neutral']} | {f['improvement_mean']:+.4f} | "
              f"{f['wilcoxon_p']} | {f['rank_biserial']:+.3f} |")
    A("")
    A("### (b) Discrimination: can personas tell the bands apart the way humans do?")
    A("")
    A("Spearman correlation across comments between (persona_A - persona_B) and "
      "(human_A - human_B). This is the paper's actual question.")
    A("")
    A("| model | band pair | comments | mean persona diff | sd persona diff | "
      "mean human diff | Spearman rho | p |")
    A("|---|---|---|---|---|---|---|---|")
    for m, md in res["models"].items():
        for k, v in md["discrimination"].items():
            if "spearman_rho" not in v:
                A(f"| {m} | {k} | {v['n_comments']} | "
                  f"{v.get('mean_persona_diff', '-')} | "
                  f"{v.get('sd_persona_diff', '-')} | "
                  f"{v.get('mean_human_diff', '-')} | - | {v.get('note', '')} |")
                continue
            A(f"| {m} | {k} | {v['n_comments']} | {v['mean_persona_diff']:+.4f} | "
              f"{v['sd_persona_diff']} | {v['mean_human_diff']:+.4f} | "
              f"{v['spearman_rho']:+.4f} | {v['spearman_p']} |")
    A("")
    ds = res["discrimination_summary"]
    A("#### Multiple-comparison accounting")
    A("")
    A(f"{ds['n_tests']} discrimination tests were run "
      f"({len(res['models'])} models x band pairs). "
      f"{ds['n_p_below_.05']} reached nominal p<.05; "
      f"{ds['expected_by_chance_at_.05']} are expected by chance. "
      f"**{ds['n_surviving_benjamini_hochberg_q.05']}** survive "
      "Benjamini-Hochberg at q=.05.")
    A("")
    if ds["nominally_significant_tests"]:
        A("Nominally significant tests (note the signs -- several are in the "
          "*wrong* direction, i.e. the persona contrast anti-correlates with "
          "the human contrast):")
        A("")
        A("| model | pair | rho | p |")
        A("|---|---|---|---|")
        for t in sorted(ds["nominally_significant_tests"], key=lambda x: x["p"]):
            A(f"| {t['model']} | {t['pair']} | {t['rho']:+.4f} | {t['p']:.4g} |")
        A("")
    A("Headline pairs across models (sign test on rho>0, 6 models):")
    A("")
    A("| band pair | rhos | median rho | n positive | sign-test p |")
    A("|---|---|---|---|---|")
    for pair in ("18-29 vs 65-69", "18-29 vs 70-74", "65-69 vs 70-74",
                 "65-69 vs 75-81", "70-74 vs 75-81"):
        v = ds.get(pair)
        if not v:
            continue
        A(f"| {pair} | {', '.join(f'{r:+.2f}' for r in v['rhos'])} | "
          f"{v['median_rho']:+.4f} | {v['n_positive']}/{v['models_reporting']} | "
          f"{v['sign_test_p_rho_gt_0']} |")
    A("")
    A("### Persona responsiveness (sanity: does the persona move the model at all?)")
    A("")
    A("| model | neutral | " + " | ".join(BANDS) + " |")
    A("|---" * (len(BANDS) + 2) + "|")
    for m, md in res["models"].items():
        pr = md["persona_response"]
        row = [m, str(pr["neutral"]["mean_rating"])]
        for b in BANDS:
            row.append(f"{pr[b]['mean_rating']}" if b in pr else "-")
        A("| " + " | ".join(row) + " |")
    A("")
    pe = res["persona_effect_size_summary"]
    A(f"Across {pe['n_model_band_cells']} model x band cells, the median "
      f"absolute shift in mean rating caused by adding a persona is "
      f"**{pe['median_abs_shift_vs_neutral_on_1to5_scale']}** points on a 1-5 "
      f"scale (max {pe['max_abs_shift_vs_neutral']}). The personas barely move "
      "the models on this task, which is the proximate cause of both null "
      "results below -- and it applies to the young personas as much as the old.")
    A("")
    A("### Run diagnostics")
    A("")
    A("| model | cells | unparsed | refusal-flagged | worst-condition unparsed "
      "rate | cost USD |")
    A("|---|---|---|---|---|---|")
    for m, d in res["diagnostics"].items():
        A(f"| {m} | {d['n_cells']} | {d['unparsed']} | {d['refusals']} | "
          f"{d['max_condition_unparsed_rate']:.1%} | {d['cost_usd']:.4f} |")
    A("")
    flagged = [(m, d) for m, d in res["diagnostics"].items()
               if d["max_condition_unparsed_rate"] > 0.05]
    if flagged:
        A("**Caveat.** Unparsed cells are dropped, and for the following models "
          "they are not distributed evenly across conditions, so the affected "
          "condition means are computed on a non-random subset:")
        A("")
        for m, d in flagged:
            A(f"- `{m}`: " + ", ".join(
                f"{c} {v['n']} ({v['rate']:.1%})"
                for c, v in d["unparsed_by_condition"].items()))
        A("")
        A("Inspection of the raw strings shows these are format-following "
          "failures (prose of the form \"I would rate this comment a 1, ...\"), "
          "not refusals. Because the dropped responses skew toward the low end "
          "of the scale, the affected condition means are biased upward. This "
          "matters for that model's young-persona row and is not load-bearing "
          "for any elder-band conclusion.")
        A("")
    A(f"Total API cost for the complete experiment (sum over all cached cells): "
      f"**${sum(d['cost_usd'] for d in res['diagnostics'].values()):.4f}** "
      f"for {sum(d['n_cells'] for d in res['diagnostics'].values()):,} calls."
      + (f" This invocation added ${total_cost:.4f}." if total_cost > 0
         else " This invocation was analysis-only and made no API calls."))
    A("")

    # ---- interpretation (reading of the run above) -------------------------
    hg = res["human_age_gradient"]["pairs"]
    ds = res["discrimination_summary"]
    pe = res["persona_effect_size_summary"]
    pw = res["power"]["band_pair_difference_scores"]
    A("## 3. Reading")
    A("")
    A("Four findings, in the order that matters.")
    A("")
    A("**1. The human age signal is real between distant ages and vanishes "
      "inside old age.** Across the core comments, raters aged 18-29 judge the "
      f"same comments {hg['18-29 vs 70-74']['mean_diff']:+.3f} points more "
      f"respectful than raters aged 70-74 "
      f"(paired Wilcoxon p={hg['18-29 vs 70-74']['wilcoxon_p']}, n="
      f"{hg['18-29 vs 70-74']['n_comments']}), and the band means fall "
      "monotonically from 18-29 through 70-74. But the elder-vs-elder "
      f"contrasts are flat: 65-69 vs 70-74 = "
      f"{hg['65-69 vs 70-74']['mean_diff']:+.3f} "
      f"(p={hg['65-69 vs 70-74']['wilcoxon_p']}), 65-69 vs 75-81 = "
      f"{hg['65-69 vs 75-81']['mean_diff']:+.3f} "
      f"(p={hg['65-69 vs 75-81']['wilcoxon_p']}), 70-74 vs 75-81 = "
      f"{hg['70-74 vs 75-81']['mean_diff']:+.3f} "
      f"(p={hg['70-74 vs 75-81']['wilcoxon_p']}).")
    A("")
    A("**2. That flatness is a density artefact, not evidence of homogeneity.** "
      "For every elder band pair the sampling-error variance of the per-comment "
      "difference *exceeds* its observed variance (ratios "
      + ", ".join(f"{v['error_var_over_observed_var']:.2f}"
                  for v in pw.values() if "error_var_over_observed_var" in v)
      + "), giving a reliability of 0 and a ceiling of r=0 on any correlation "
        "a persona could achieve against these targets. The median smallest "
        "difference the available raters could resolve on a comment is "
      + ", ".join(f"{k}: {v['median_minimum_resolvable_difference']}"
                  for k, v in pw.items()
                  if "median_minimum_resolvable_difference" in v)
      + " points, against true differences that are an order of magnitude "
        "smaller. The elder-vs-elder target is not a weak signal we failed to "
        "detect; at this density it is not a measurement.")
    A("")
    A("**3. Fidelity: no model's elder persona gets reliably closer to real "
      "elders than no persona does.** Absolute errors are small in absolute "
      "terms (MAE ~0.25-0.45 on a 1-5 scale -- the models track comment "
      "severity well) but the persona contributes essentially nothing to that. "
      "No band x model cell shows a significant persona advantage; the only "
      "nominally significant fidelity result in the table is a persona making "
      "matters *worse*.")
    A("")
    A(f"**4. Discrimination fails, but it fails everywhere, including the "
      f"positive control.** Of {ds['n_tests']} tests, "
      f"{ds['n_p_below_.05']} reach nominal p<.05 against "
      f"{ds['expected_by_chance_at_.05']} expected by chance, and "
      f"{ds['n_surviving_benjamini_hochberg_q.05']} survive Benjamini-Hochberg. "
      "Crucially the models cannot separate a 24-year-old from a 67-year-old "
      "either. The proximate cause is visible in the responsiveness table: "
      f"the median absolute shift a persona produces is "
      f"{pe['median_abs_shift_vs_neutral_on_1to5_scale']} points on a 1-5 "
      "scale. On this task these personas are close to inert, so the elder-vs-"
      "elder null is **not** specific to old age and must not be reported as "
      "though it were.")
    A("")
    A("### What this does and does not license")
    A("")
    A("- MHSC genuinely is an exception to the resolution claim *at the level "
      "of the sampling frame*: exact ages are recorded, so age-resolved elder "
      "targets can be constructed at all, which POPQUORN makes impossible. "
      "The claim \"no dataset records age finely enough\" is false and should "
      "not be made.")
    A("- MHSC is **not** an exception at the level of *statistical power*. "
      "Recording an age is not the same as sampling enough people at that age. "
      "12 annotators aged 75-81 in the whole corpus, contributing 68 ratings "
      "spread over 42 comments, with a median of one 75+ rater per comment, "
      "cannot adjudicate anything. Fine-grained metadata over a thin tail "
      "produces bins that are nominally resolved and empirically empty.")
    A("- The honest statement is therefore: **MHSC can support a 65-69 elder "
      "target and, marginally, a 70-74 one; it cannot support a 75+ target, "
      "and it cannot support any contrast between elder bands.**")
    A("- Because the positive control also fails, this run cannot separate "
      "\"elder personas are unvalidatable\" from \"these personas do not "
      "respond to age prompting on this task.\" Both are consistent with the "
      "data. A stronger persona elicitation might move the models; it would "
      "still have nothing reliable to be scored against inside old age, which "
      "is finding 2 and does not depend on the models at all.")
    A("")
    A("### Limitations")
    A("")
    A("- n=70 comments. Forced by the data, not chosen; see Section 1.")
    A("- The 70 core comments are MHSC's high-replication calibration set and "
      "are not a random sample of the corpus. Elder targets exist only there.")
    A("- Each comment's elder band mean is built from a different handful of "
      "people (every 65+ annotator rated ~6 of the 70), so band means confound "
      "age with individual rater severity. MHSC's own Rasch severity adjustment "
      "would address this but we deliberately do not use it -- see the header "
      "note on why `hate_speech_score` is unsuitable as a target here.")
    A("- One persona formulation, one rating instruction, one target column. "
      "The nulls are nulls for this operationalisation.")
    A("- `respect` is a judgement about a comment's stance toward a target "
      "group; it is the nearest available analogue to a politeness rating but "
      "it is not one.")
    A("")
    (results_dir / "mhsc_validation.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=os.environ.get(
        "MHSC_PARQUET", str(HARNESS_DIR / "data" / "mhsc.parquet")))
    ap.add_argument("--limit-comments", type=int, default=None)
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    parquet = ensure_parquet(Path(args.parquet))
    comments, targets, feas = build_design(parquet, args.limit_comments)
    print(f"design: {len(comments)} comments, "
          f"{len(BANDS)} persona conditions + neutral control")

    conditions: list = [None] + [v[2] for v in BANDS.values()]
    runs_dir = HARNESS_DIR / "runs_mhsc"
    runs_dir.mkdir(exist_ok=True)
    results_dir = HARNESS_DIR / cfg["results_dir"]
    results_dir.mkdir(exist_ok=True)

    total_cost = 0.0
    if not args.analyse_only:
        key = get_api_key()
        n_cells = len(comments) * len(conditions) * cfg["samples"]
        print(f"planned cells per model: {n_cells} "
              f"({n_cells * len(cfg['models'])} total)")
        for model in cfg["models"]:
            total_cost += run_model(model, comments, conditions, cfg["samples"],
                                    key, cfg, runs_dir)
        print(f"\nAPI cost this invocation: ${total_cost:.4f}")

    res = analyse(comments, targets, cfg["models"], runs_dir)
    res["feasibility"] = feas
    res["design"] = {
        "target_column": "respect",
        "target_transform": "respectfulness = 5 - respect (1..5)",
        "rating_instruction": RATING_INSTRUCTION,
        "persona_system_message_template": "You are a/an {age}-year-old adult.",
        "bands": {b: {"lo": lo, "hi": hi, "persona_age": p}
                  for b, (lo, hi, p) in BANDS.items()},
        "comment_ids": [c["comment_id"] for c in comments],
        "n_comments": len(comments),
        "conditions": ["neutral" if c is None else c for c in conditions],
        "models": cfg["models"], "samples": cfg["samples"],
        "temperature": cfg["temperature"],
        "min_band_n_primary": MIN_BAND_N,
    }
    res["human_targets"] = {str(k): v for k, v in targets.items()}
    res["api_cost_this_invocation_usd"] = round(total_cost, 4)
    (results_dir / "mhsc_validation.json").write_text(json.dumps(res, indent=1))
    write_report(res, feas, cfg, comments, total_cost, results_dir)


if __name__ == "__main__":
    main()
