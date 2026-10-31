#!/usr/bin/env python3
"""Config B pilot analysis (HARNESS_SPEC §5) + Figure 2.

Inputs:
  --pilot-dir  frozen export dir containing ratings_long.csv (default:
               ../data-exports/2026-07-29-FROZEN)
  runs/        real harness cache (per-item model ratings)
  results/human_age_profiles.csv  POPQUORN per-bracket per-item stats

Analyses on the 40 real app items (anchors excluded), completers only
(>=30/40 real items):
  Q1 gradient resolution: pilot bracket means + bootstrap CIs vs POPQUORN's
     18->64 bracket means and the >65 band, all restricted to the 40 items.
  Q2 variance: pilot per-item SD (across participants) vs each model's
     within-cell SD at persona 65/75/85, paired across items (Wilcoxon).
  Q3 fidelity: per model, per item |persona-75 mean - pilot mean| vs
     |neutral mean - pilot mean| (Wilcoxon; median paired diff; sign).
  Covariate: pilot means split by residence country.

Outputs: results/pilot_analysis.md, results/fig2.pdf, results/pilot_stats.json
"""

from __future__ import annotations

import argparse
import json
import os
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from harness_common import HARNESS_DIR, cell_key, load_config, load_run_lines

BRACKET_MIDPOINTS = {
    "18-24": 21.0, "25-29": 27.0, "30-34": 32.0, "35-39": 37.0,
    "40-44": 42.0, "45-49": 47.0, "50-54": 52.0, "54-59": 56.5,
    "60-64": 62.0, ">65": 82.5,
}
BAND_LO, BAND_HI = 65.0, 100.0
PILOT_BRACKET_X = {"65–69": 67.0, "70–74": 72.0, "75–79": 77.0,
                   "80–84": 82.0, "85+": 90.0}
MIN_REAL_ITEMS = 30
RNG = np.random.default_rng(20260729)


def bootstrap_ci(values: np.ndarray, n_boot: int = 10000) -> tuple[float, float]:
    """95% CI of the mean by participant-level bootstrap."""
    if len(values) < 2:
        return (math.nan, math.nan)
    idx = RNG.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def load_pilot(pilot_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    long_df = pd.read_csv(pilot_dir / "ratings_long.csv")
    real = long_df[long_df["role"] == "item"].copy()
    real["item_id"] = real["item_id"].astype(str)
    counts = real.groupby("participant")["item_id"].nunique()
    completers = counts[counts >= MIN_REAL_ITEMS].index.tolist()
    excluded = {p: int(c) for p, c in counts.items() if c < MIN_REAL_ITEMS}
    kept = real[real["participant"].isin(completers)].copy()
    demo = (kept.groupby("participant")
            .agg(age_group=("age_group", "first"),
                 residence_country=("residence_country", "first")))
    meta = {
        "n_started": int(long_df["participant"].nunique()),
        "n_completers": len(completers),
        "excluded_partials": excluded,
        "bracket_counts": demo["age_group"].value_counts().to_dict(),
        "country_counts": demo["residence_country"].value_counts().to_dict(),
    }
    return kept, demo, meta


def model_item_stats(config: dict, item_ids: list[str]) -> pd.DataFrame:
    """Per model x condition x item: mean over parsed samples + within-cell SD."""
    runs_dir = HARNESS_DIR / config["runs_dir"]
    cached = load_run_lines(runs_dir, config["models"])
    rows = []
    for model in config["models"]:
        cells = cached[model]
        for persona_age in config["persona_ages"]:
            cond = "neutral" if persona_age is None else str(persona_age)
            for iid in item_ids:
                ratings = [rec["parsed"] for s in range(config["samples"])
                           if (rec := cells.get(cell_key(persona_age, iid, s)))
                           and rec.get("parsed") is not None]
                if ratings:
                    rows.append({
                        "model": model, "condition": cond, "item_id": iid,
                        "mean": float(np.mean(ratings)),
                        "sd": float(np.std(ratings, ddof=1)) if len(ratings) >= 2 else math.nan,
                    })
    return pd.DataFrame(rows)


def human_bracket_stats(item_ids: list[str]) -> pd.DataFrame:
    df = pd.read_csv(HARNESS_DIR / "results" / "human_age_profiles.csv")
    df["age"] = df["age"].astype(str).str.strip()
    df["instance_id"] = df["instance_id"].astype(str)
    return df[df["instance_id"].isin(item_ids)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-dir",
                    default=str(HARNESS_DIR.parent / "data-exports" / "2026-07-29-FROZEN"))
    args = ap.parse_args()
    pilot_dir = Path(args.pilot_dir)

    config = load_config()
    kept, demo, meta = load_pilot(pilot_dir)
    item_ids = sorted(kept["item_id"].unique())
    print(f"completers={meta['n_completers']} brackets={meta['bracket_counts']} "
          f"countries={meta['country_counts']} excluded={meta['excluded_partials']}")

    # Pilot per-item means/SDs (across completers) and per-bracket means.
    pilot_item = kept.groupby("item_id")["rating"].agg(["mean", "std", "count"])
    pilot_overall = kept.groupby("participant")["rating"].mean()  # per-participant mean
    out: dict = {"meta": meta}

    # ----- Q1: gradient resolution ---------------------------------------
    q1 = {}
    for bracket, grp in demo.groupby("age_group"):
        sub = kept[kept["participant"].isin(grp.index)]
        per_part = sub.groupby("participant")["rating"].mean().to_numpy()
        lo, hi = bootstrap_ci(per_part)
        q1[bracket] = {"n": len(per_part), "mean": round(float(per_part.mean()), 3),
                       "ci95": [round(lo, 3), round(hi, 3)]}
    pilot_all = pilot_overall.to_numpy()
    lo, hi = bootstrap_ci(pilot_all)
    q1["all_65plus"] = {"n": len(pilot_all), "mean": round(float(pilot_all.mean()), 3),
                        "ci95": [round(lo, 3), round(hi, 3)]}
    out["q1_pilot_brackets"] = q1

    hum = human_bracket_stats(item_ids)
    hum_brackets = {}
    for bracket in BRACKET_MIDPOINTS:
        sub = hum[hum["age"] == bracket]
        if not sub.empty:
            hum_brackets[bracket] = {"mean": round(float(sub["mean"].mean()), 3),
                                     "n_items": int(sub["instance_id"].nunique())}
    out["q1_popquorn_brackets_40items"] = hum_brackets

    # ----- Q2: variance flattening ----------------------------------------
    mstats = model_item_stats(config, item_ids)
    q2 = {}
    pilot_sd = pilot_item["std"].dropna()
    for model in config["models"]:
        for cond in ["65", "75", "85"]:
            msd = (mstats[(mstats["model"] == model) & (mstats["condition"] == cond)]
                   .set_index("item_id")["sd"].dropna())
            common = pilot_sd.index.intersection(msd.index)
            if len(common) < 10:
                continue
            a, b = pilot_sd.loc[common].to_numpy(), msd.loc[common].to_numpy()
            try:
                w = stats.wilcoxon(a, b)
                p = float(w.pvalue)
            except ValueError:
                p = math.nan
            q2[f"{model}|{cond}"] = {
                "n_items": int(len(common)),
                "pilot_median_sd": round(float(np.median(a)), 3),
                "persona_median_sd": round(float(np.median(b)), 3),
                "median_diff": round(float(np.median(a - b)), 3),
                "wilcoxon_p": None if math.isnan(p) else round(p, 5),
            }
    out["q2_variance"] = q2

    # ----- Q3: persona fidelity -------------------------------------------
    q3 = {}
    pilot_mean = pilot_item["mean"]
    for model in config["models"]:
        m75 = (mstats[(mstats["model"] == model) & (mstats["condition"] == "75")]
               .set_index("item_id")["mean"])
        mneu = (mstats[(mstats["model"] == model) & (mstats["condition"] == "neutral")]
                .set_index("item_id")["mean"])
        common = pilot_mean.index.intersection(m75.index).intersection(mneu.index)
        d_persona = (m75.loc[common] - pilot_mean.loc[common]).abs().to_numpy()
        d_neutral = (mneu.loc[common] - pilot_mean.loc[common]).abs().to_numpy()
        diff = d_persona - d_neutral  # negative -> persona closer to real elders
        try:
            w = stats.wilcoxon(d_persona, d_neutral)
            p = float(w.pvalue)
        except ValueError:
            p = math.nan
        n_pos = int((diff > 0).sum())
        n_neg = int((diff < 0).sum())
        q3[model] = {
            "n_items": int(len(common)),
            "median_abs_err_persona75": round(float(np.median(d_persona)), 3),
            "median_abs_err_neutral": round(float(np.median(d_neutral)), 3),
            "median_paired_diff": round(float(np.median(diff)), 3),
            "items_persona_closer": n_neg, "items_neutral_closer": n_pos,
            "wilcoxon_p": None if math.isnan(p) else round(p, 5),
        }
    out["q3_fidelity"] = q3

    # ----- Covariate split -------------------------------------------------
    cov = {}
    for country, grp in demo.groupby("residence_country"):
        sub = kept[kept["participant"].isin(grp.index)]
        per_part = sub.groupby("participant")["rating"].mean().to_numpy()
        cov[str(country)] = {"n": len(per_part), "mean": round(float(per_part.mean()), 3)}
    out["covariate_residence"] = cov

    # ----- Figure 2 ---------------------------------------------------------
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "pdf.fonttype": 42,
        "axes.linewidth": 0.6,
    })
    wide = bool(os.environ.get("WIDE_FIG"))
    fig = plt.figure(figsize=(5.5, 3.4) if wide else (3.3, 2.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[4.2, 1.0], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharex=ax)   # the two rulers

    PILOT_COLOR = "#0072B2"
    pop_edges = [18, 25, 30, 35, 40, 45, 50, 55, 60, 65]
    pilot_edges = [(65, 70, "65\u201369"), (70, 75, "70\u201374"), (75, 80, "75\u201379"),
                   (80, 85, "80\u201384"), (85, 100, "85+")]

    # ---- data panel -----------------------------------------------------
    for b, st in hum_brackets.items():
        if b == ">65":
            continue
        lo, hi = (float(s) for s in b.split("-"))
        ax.plot([lo, hi], [st["mean"], st["mean"]], "-", color="0.35",
                lw=2.0, solid_capstyle="butt", zorder=4)
    band = hum[hum["age"] == ">65"]
    band_means = band.groupby("instance_id")["mean"].mean()
    bmean = float(band_means.mean())
    bci = 1.96 * float(band_means.std(ddof=1)) / math.sqrt(len(band_means))
    ax.axvspan(BAND_LO, BAND_HI, color="0.93", lw=0, zorder=0)
    ax.plot([BAND_LO, BAND_HI], [bmean, bmean], "--", color="0.35", lw=1.8,
            dashes=(5, 3), zorder=4)
    ax.text(BAND_HI - 1, bmean + 0.015, "one value for the whole range",
            ha="right", va="bottom", fontsize=6, style="italic", color="0.4")

    for bracket, grp in demo.groupby("age_group"):
        lo, hi, _ = next(e for e in pilot_edges if e[2] == bracket)
        ys = sorted(pilot_overall.loc[p] for p in grp.index)
        xc = (lo + hi) / 2
        ax.plot([xc] * len(ys), ys, "s", color=PILOT_COLOR, ms=4.2, mec="white",
                mew=0.5, zorder=6, clip_on=False)
        ax.plot([lo + 0.7, hi - 0.7], [float(np.mean(ys))] * 2, "-",
                color=PILOT_COLOR, lw=1.4, alpha=0.55, solid_capstyle="butt",
                zorder=5)
    ax.set_ylabel("mean politeness rating (1\u20135)")
    ax.grid(True, axis="y", lw=0.3, color="0.92", zorder=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelbottom=False)

    # ---- ruler panel: what each instrument can distinguish ---------------
    filled = set(demo["age_group"].unique())

    def cell(lo, hi, y, color, height=0.30, dashed=False, fill=None):
        axr.add_patch(plt.Rectangle((lo, y), hi - lo, height, facecolor=fill or "none",
                                    edgecolor=color, lw=0.7, zorder=3,
                                    linestyle=(0, (2, 1.6)) if dashed else "solid"))

    raters = {}
    for b in hum_brackets:
        sub = hum[hum["age"] == b]
        raters[b] = int(sub.groupby("instance_id")["n"].mean().sum() / max(sub["instance_id"].nunique(), 1)) \
            if not sub.empty else 0
    ordered = [b for b in hum_brackets if b != ">65"]
    for i in range(len(pop_edges) - 1):
        lo, hi = pop_edges[i], pop_edges[i + 1]
        cell(lo, hi, 0.55, "0.35")
        if i < len(ordered):
            axr.text((lo + hi) / 2, 0.70, f"n={raters.get(ordered[i], '')}",
                     ha="center", va="center", fontsize=4.2, color="0.45")
    cell(BAND_LO, BAND_HI, 0.55, "0.35", fill="0.90")
    axr.text((BAND_LO + BAND_HI) / 2, 0.70,
             f"one cell, 35 years ({raters.get('>65', '')} rater per item)",
             ha="center", va="center", fontsize=5.6, style="italic", color="0.3")

    for lo, hi, label in pilot_edges:
        got = label in filled
        cell(lo, hi, 0.08, PILOT_COLOR, dashed=not got,
             fill="#DCEBF5" if got else None)

        lbl = f"n={sum(demo['age_group'] == label)}" if got else ("1*" if label == "80\u201384" else "0")
        axr.text((lo + hi) / 2, 0.23, lbl, ha="center", va="center", fontsize=5.5,
                 color=PILOT_COLOR if got else "0.55")
        if label == "85+":
            axr.text(hi - 0.6, 0.23, "85+", ha="right", va="center", fontsize=4.6,
                     color="0.55", style="italic")

    axr.text(17.4, 0.76, "POPQUORN", ha="right", va="center", fontsize=6, color="0.35")
    axr.text(17.4, 0.63, "raters/item", ha="right", va="center", fontsize=4.4, color="0.55")
    axr.text(17.4, 0.29, "this pilot", ha="right", va="center", fontsize=6,
             color=PILOT_COLOR)
    axr.text(17.4, 0.16, "people recruited", ha="right", va="center", fontsize=4.4,
             color="0.55")
    axr.set_ylim(0, 1.0)
    axr.set_xlim(18, BAND_HI)
    axr.set_xticks(pop_edges + [70, 75, 80, 85, 100])
    axr.set_xticklabels([str(e) for e in pop_edges] + ["70", "75", "80", "85", "100"],
                        fontsize=5.8)
    axr.set_yticks([])
    axr.set_xlabel("age (years); boxes show the age cells each instrument can distinguish")
    for s in ["top", "right", "left"]:
        axr.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(HARNESS_DIR / "results" / "fig2.pdf", bbox_inches="tight")
    plt.close(fig)

    # ----- Reports -----------------------------------------------------------
    (HARNESS_DIR / "results" / "pilot_stats.json").write_text(json.dumps(out, indent=1))
    md = ["# Pilot analysis (frozen 2026-07-29)", "",
          f"Started: {meta['n_started']}; completers (>= {MIN_REAL_ITEMS}/40 real items): "
          f"{meta['n_completers']}; excluded partials (real items rated): {meta['excluded_partials']}",
          f"Brackets: {meta['bracket_counts']} | Residence: {meta['country_counts']}", "",
          "## Q1 pilot bracket means (bootstrap 95% CI over participants)"]
    for k, v in q1.items():
        md.append(f"- {k}: n={v['n']} mean={v['mean']} CI=[{v['ci95'][0]}, {v['ci95'][1]}]")
    md += ["", f"POPQUORN >65 band on these 40 items: mean={bmean:.3f} ± {bci:.3f} (95% CI across items)",
           "", "## Q2 pilot per-item SD vs persona within-cell SD (paired Wilcoxon)"]
    for k, v in q2.items():
        md.append(f"- {k}: pilot med SD={v['pilot_median_sd']} vs persona {v['persona_median_sd']} "
                  f"(diff {v['median_diff']}, p={v['wilcoxon_p']}, n={v['n_items']})")
    md += ["", "## Q3 |persona-75 - pilot| vs |neutral - pilot| per item"]
    for k, v in q3.items():
        md.append(f"- {k}: med err persona={v['median_abs_err_persona75']} vs neutral="
                  f"{v['median_abs_err_neutral']}; persona closer on {v['items_persona_closer']}/"
                  f"{v['n_items']} items (p={v['wilcoxon_p']})")
    md += ["", f"## Residence covariate: {json.dumps(cov)}"]
    (HARNESS_DIR / "results" / "pilot_analysis.md").write_text("\n".join(md) + "\n")
    print("Wrote results/pilot_analysis.md, results/pilot_stats.json, results/fig2.pdf")


if __name__ == "__main__":
    main()
