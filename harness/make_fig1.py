#!/usr/bin/env python3
"""Build results/model_curves.csv and results/fig1.pdf (HARNESS_SPEC §4).

Panel A: mean predicted politeness vs persona age, one line per model with
direct end-of-line labels (no legend). Human ground truth: bracket means as
points at bracket midpoints through 60-64, then the '>65' bracket as a single
flat band spanning 65->100 (mean +/- 95% CI across items), labeled as the one
bin in which every annotator over 65 is indistinguishable.

Panel B: mean within-cell SD (across the 3 samples) per model vs persona age,
with the human per-bracket SD as points and the '>65' bracket as a band.

Grayscale-safe: identity is carried by linestyle + marker + direct labels,
not color. Column width (3.3in), two stacked panels.

Usage:
    python3 make_fig1.py                  # real cached runs in runs/
    python3 make_fig1.py --synthetic      # fabricate runs_synthetic/ and plot
    python3 make_fig1.py --runs-dir runs_synthetic
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from harness_common import (
    HARNESS_DIR,
    cell_key,
    condition_label,
    load_config,
    load_items,
    load_run_lines,
    model_slug_to_filename,
)

# Age-bracket midpoints. '54-59' overlaps '50-54' in the source data; we take
# its midpoint literally (56.5). '>65' is open-ended and is NOT plotted as a
# point: it becomes a flat band spanning BAND_LO..BAND_HI (its x_age in the
# CSV is the band's visual center).
BRACKET_MIDPOINTS = {
    "18-24": 21.0, "25-29": 27.0, "30-34": 32.0, "35-39": 37.0,
    "40-44": 42.0, "45-49": 47.0, "50-54": 52.0, "54-59": 56.5,
    "60-64": 62.0, ">65": 82.5,
}
BAND_LO, BAND_HI = 65.0, 100.0  # the '>65' bin drawn across 35 years of life
# Annotator count in the '>65' politeness bracket, verified from raw_data.csv
# (see results/selection_report.md).
BAND_N = 69

# Okabe-Ito colorblind-safe palette, paired with distinct linestyles and
# markers so the figure also survives grayscale printing.
MODEL_STYLES = [
    ("#E69F00", "-", "o"),   # orange
    ("#56B4E9", "--", "s"),  # sky blue
    ("#009E73", "-.", "^"),  # bluish green
    ("#D55E00", ":", "D"),   # vermilion
    ("#CC79A7", "-", "v"),   # reddish purple
    ("#0072B2", "--", "x"),  # blue
]


def short_model_name(slug: str) -> str:
    return slug.split("/", 1)[-1].replace("-instruct", "")


# ---------------------------------------------------------------------------
# Synthetic runs (so the figure/QC pipeline is testable before any API call)
# ---------------------------------------------------------------------------

def generate_synthetic_runs(config: dict, items: list[dict], out_dir: Path, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = config["models"]
    # Per-model behavior: bias, age slope (per decade), base SD, SD shrink with age.
    params = {
        m: {
            "bias": rng.uniform(-0.25, 0.25),
            "slope": rng.uniform(-0.05, 0.2),
            "sd": rng.uniform(0.45, 0.8),
            "shrink": rng.uniform(0.0, 0.6),  # variance collapse at high ages
        }
        for m in models
    }
    for model in models:
        p = params[model]
        path = out_dir / model_slug_to_filename(model)
        with open(path, "w", encoding="utf-8") as f:
            for persona_age in config["persona_ages"]:
                eff_age = 32 if persona_age is None else persona_age  # neutral ~ young default
                sd = p["sd"] * max(0.25, 1 - p["shrink"] * (eff_age - 25) / 60)
                for item in items:
                    for s in range(config["samples"]):
                        latent = (item["popquorn_mean"] + p["bias"]
                                  + p["slope"] * (eff_age - 45) / 10
                                  + rng.normal(0, sd))
                        rating = int(np.clip(round(latent), 1, 5))
                        roll = rng.random()
                        base = {"model": model, "persona_age": persona_age,
                                "item_id": str(item["item_id"]), "sample_idx": s,
                                "prompt_tokens": 100, "completion_tokens": 2,
                                "cost_usd": 3e-4, "error": None,
                                "ts": "1970-01-01T00:00:00"}
                        if roll < 0.015:  # format failure on first attempt
                            f.write(json.dumps({**base, "attempt": 0, "parsed": None,
                                                "raw_response": "I think this message sounds fairly polite overall."}) + "\n")
                            if rng.random() < 0.7:  # strict retry succeeds
                                f.write(json.dumps({**base, "attempt": 1, "parsed": rating,
                                                    "raw_response": str(rating)}) + "\n")
                        elif roll < 0.018:  # refusal, never recovers
                            f.write(json.dumps({**base, "attempt": 0, "parsed": None,
                                                "raw_response": "I'm sorry, but I can't role-play a specific age."}) + "\n")
                        else:
                            f.write(json.dumps({**base, "attempt": 0, "parsed": rating,
                                                "raw_response": f" {rating}."}) + "\n")
    print(f"Synthetic runs written to {out_dir}")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def model_curves(config: dict, items: list[dict], runs_dir: Path) -> list[dict]:
    """Per model x condition: mean rating over items, mean within-cell SD."""
    cached = load_run_lines(runs_dir, config["models"])
    rows = []
    for model in config["models"]:
        cells = cached[model]
        for persona_age in config["persona_ages"]:
            item_means, item_sds, n_parsed = [], [], 0
            for item in items:
                ratings = []
                for s in range(config["samples"]):
                    rec = cells.get(cell_key(persona_age, item["item_id"], s))
                    if rec is not None and rec.get("parsed") is not None:
                        ratings.append(rec["parsed"])
                n_parsed += len(ratings)
                if ratings:
                    item_means.append(float(np.mean(ratings)))
                if len(ratings) >= 2:
                    item_sds.append(float(np.std(ratings, ddof=1)))
            rows.append({
                "source": "model",
                "model": model,
                "condition": condition_label(persona_age),
                "x_age": "" if persona_age is None else persona_age,
                "age_bracket": "",
                "mean_rating": round(float(np.mean(item_means)), 4) if item_means else "",
                "mean_within_sd": round(float(np.mean(item_sds)), 4) if item_sds else "",
                "n_items": len(item_means),
                "n_ratings": n_parsed,
            })
    return rows


def human_curve(items: list[dict]) -> list[dict]:
    """Per-bracket mean politeness / mean per-item SD across the 200 items."""
    df = pd.read_csv(HARNESS_DIR / "results" / "human_age_profiles.csv")
    df["age"] = df["age"].astype(str).str.strip()
    item_ids = {str(i["item_id"]) for i in items}
    df = df[df["instance_id"].astype(str).isin(item_ids)]
    rows = []
    for bracket, mid in BRACKET_MIDPOINTS.items():
        sub = df[df["age"] == bracket]
        if sub.empty:
            continue
        item_means = sub["mean"].to_numpy(dtype=float)
        item_sds = sub["sd"].dropna().to_numpy(dtype=float)
        rows.append({
            "source": "human",
            "model": "human",
            "condition": bracket,
            "x_age": mid,
            "age_bracket": bracket,
            "mean_rating": round(float(item_means.mean()), 4),
            # per-item SD exists only where a bracket has >=2 raters on an item
            "mean_within_sd": round(float(item_sds.mean()), 4) if len(item_sds) else "",
            "ci95_mean": round(1.96 * float(item_means.std(ddof=1)) / math.sqrt(len(item_means)), 4)
                         if len(item_means) >= 2 else "",
            "ci95_sd": round(1.96 * float(item_sds.std(ddof=1)) / math.sqrt(len(item_sds)), 4)
                       if len(item_sds) >= 2 else "",
            "n_items": int(sub["instance_id"].nunique()),
            "n_ratings": int(sub["n"].sum()),
        })
    return rows


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def spread_labels(targets: list[float], lo: float, hi: float, min_gap: float) -> list[float]:
    """Nudge label y-positions apart so end-of-line labels don't collide."""
    order = sorted(range(len(targets)), key=lambda i: targets[i])
    ys = [targets[i] for i in order]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_gap)
    over = ys[-1] - hi
    if over > 0:  # push back down from the top, preserving gaps
        ys = [y - over for y in ys]
        for i in range(len(ys) - 2, -1, -1):
            ys[i] = min(ys[i], ys[i + 1] - min_gap)
    ys = [max(y, lo) for y in ys]
    out = [0.0] * len(targets)
    for pos, idx in enumerate(order):
        out[idx] = ys[pos]
    return out


UP_COLOR, DOWN_COLOR = "#D55E00", "#0072B2"  # slope direction in panel B
WIDE = False           # set by --wide: single-column (NeurIPS) proportions
WIDE_SIZE = (5.5, 4.6)
# Paired 65->85 contrasts surviving Bonferroni correction across six models.
SIG_MODELS = {"claude-sonnet-4.5", "gemini-2.5-flash", "llama-3.1-70b"}


def item_mean_cis(config, items, runs_dir, ages):
    """95% CI half-width of each (model, persona age) mean across item means."""
    cached = load_run_lines(runs_dir, config["models"])
    out = {}
    for model in config["models"]:
        cells = cached[model]
        for a in ages:
            means = []
            for it in items:
                r = [rec["parsed"] for s in range(config["samples"])
                     if (rec := cells.get(cell_key(a, it["item_id"], s)))
                     and rec.get("parsed") is not None]
                if r:
                    means.append(float(np.mean(r)))
            arr = np.array(means)
            out[(model, a)] = 1.96 * arr.std(ddof=1) / math.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return out


def paired_deltas(config, items, runs_dir, ages):
    """Per model/age: mean paired change from the age-65 baseline + 95% CI."""
    cached = load_run_lines(runs_dir, config["models"])
    out = {}
    for model in config["models"]:
        cells = cached[model]

        def means(a):
            d = {}
            for it in items:
                r = [rec["parsed"] for s in range(config["samples"])
                     if (rec := cells.get(cell_key(a, it["item_id"], s)))
                     and rec.get("parsed") is not None]
                if r:
                    d[str(it["item_id"])] = float(np.mean(r))
            return d

        base = means(ages[0])
        for a in ages:
            cur = means(a)
            ids = sorted(set(base) & set(cur))
            diff = np.array([cur[i] - base[i] for i in ids])
            ci = (1.96 * diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else 0.0
            out[(model, a)] = (float(diff.mean()), float(ci))
    return out


def make_figure(config: dict, rows: list[dict], out_pdf: Path,
                items_for_ci=None, runs_dir_for_ci=None) -> None:
    plt.rcParams.update({
        # Serif matching the paper's Times body, so the figure reads as part
        # of the document rather than a matplotlib default.
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "pdf.fonttype": 42,
        "axes.linewidth": 0.6,
    })
    fig = plt.figure(figsize=(WIDE_SIZE if WIDE else (3.3, 5.35)))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1.35], hspace=0.62)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    models = config["models"]
    persona_x = [a for a in config["persona_ages"] if a is not None]
    by_key = {(r["model"], r["condition"]): r for r in rows}
    human_rows = sorted((r for r in rows if r["source"] == "human"), key=lambda r: r["x_age"])
    h_pts = [r for r in human_rows if r["age_bracket"] != ">65"]
    h_band = next(r for r in human_rows if r["age_bracket"] == ">65")

    def val(r, field):
        return r[field] if r[field] != "" else math.nan

    def style(ax):
        ax.grid(True, axis="y", lw=0.3, color="0.9", zorder=1)
        ax.spines[["top", "right"]].set_visible(False)

    band_y, band_ci = val(h_band, "mean_rating"), val(h_band, "ci95_mean")

    # ----- Panel A: the ruler. Human bracket segments and the >65 band are
    # the protagonists; model curves are thin context only (detailed in B).
    model_vals, ends_a = [], []
    for mi, model in enumerate(models):
        color, ls, _ = MODEL_STYLES[mi % len(MODEL_STYLES)]
        ys = [val(by_key[(model, str(a))], "mean_rating") for a in persona_x]
        model_vals.extend(ys)
        ax_a.plot(persona_x, ys, ls, color=color, lw=0.8, alpha=0.5, zorder=2)
        ends_a.append((ys[-1], short_model_name(model), color))
    hvals = [val(r, "mean_rating") for r in h_pts]
    ax_a.fill_between([BAND_LO, BAND_HI], band_y - band_ci, band_y + band_ci,
                      color="0.85", lw=0, zorder=1)
    ax_a.plot([BAND_LO, BAND_HI], [band_y, band_y], "--", color="black",
              lw=1.5, dashes=(4, 2), zorder=4)
    for r, v in zip(h_pts, hvals):
        lo_x, hi_x = (float(s) for s in r["age_bracket"].split("-"))
        ax_a.plot([lo_x, hi_x], [v, v], "-", color="black", lw=1.8,
                  solid_capstyle="butt", zorder=4)
    lo = min(model_vals + hvals)
    hi = max(hvals + [band_y + band_ci])
    ax_a.set_ylim(lo - 0.26 * (hi - lo), hi + 0.08 * (hi - lo))
    ends_a.append((hvals[-1], "POPQUORN raters", "black"))
    ends_a.append((band_y, "human: >65 bin", "black"))
    ylo_a, yhi_a = ax_a.get_ylim()
    label_ys_a = spread_labels([e[0] for e in ends_a], ylo_a, yhi_a,
                               0.085 * (yhi_a - ylo_a))
    for (y_end, name, color), y_lab in zip(ends_a, label_ys_a):
        is_human = name.startswith("human")
        anchor = BAND_HI if name.startswith("human:") else (64 if name.startswith("POPQUORN") else persona_x[-1])
        ax_a.annotate(name, xy=(anchor, y_end),
                      xytext=(BAND_HI + 1.5, y_lab), fontsize=5.5, color=color,
                      va="center", ha="left",
                      fontweight="bold" if is_human else "normal",
                      annotation_clip=False,
                      arrowprops={"arrowstyle": "-", "lw": 0.4, "color": "0.6",
                                  "shrinkA": 1, "shrinkB": 1})
    ax_a.text((BAND_LO + BAND_HI) / 2, ax_a.get_ylim()[0] + 0.015 * (hi - lo),
              f"one value for ages 65–100 (n={BAND_N} annotators)",
              ha="center", va="bottom", fontsize=5.8, style="italic",
              color="0.25", zorder=5)
    ylo_a2, yhi_a2 = ax_a.get_ylim()
    ax_a.text(18.5, yhi_a2, f"axis shows {yhi_a2 - ylo_a2:.1f} of the 4-point range",
              fontsize=5.2, style="italic", color="0.45", va="center", ha="left")
    ax_a.set_xlim(17, BAND_HI)
    ax_a.set_xticks([25, 35, 45, 55, 65, 75, 85, 95])
    ax_a.set_xlabel("age (years): real rater age for segments, persona age for curves",
                    fontsize=6.5)
    ax_a.set_ylabel("mean rating (1–5)")
    ax_a.set_title("A. Human age resolution vs. simulated elders", loc="left")
    style(ax_a)

    # ----- Panel B: inside the bin. Slopegraph of persona means at 65/75/85,
    # colored by direction of travel; the one human number behind them.
    ages = [65, 75, 85]
    ends, n_up = [], 0
    # Panel B plots each model's change from its own 65 baseline, with paired
    # 95% CIs across the 200 items: the uncertainty that matters for the
    # direction claim. The human record cannot move here at all -- it has one
    # value for the whole span -- so its line is zero by construction.
    deltas = paired_deltas(config, items_for_ci, runs_dir_for_ci, ages)
    for model in models:
        ys = [deltas[(model, a)][0] for a in ages]
        errs = [deltas[(model, a)][1] for a in ages]
        up = ys[-1] >= 0
        n_up += up
        # Only Bonferroni survivors get direction color; the rest are gray,
        # so the panel cannot be skimmed as "three up, three down, all real".
        sig = short_model_name(model) in SIG_MODELS
        c = (UP_COLOR if up else DOWN_COLOR) if sig else "0.65"
        ax_b.errorbar(ages, ys, yerr=errs, fmt="-o", color=c, lw=1.2, ms=2.8,
                      elinewidth=0.9, capsize=2.5, zorder=3, clip_on=False)
        ends.append((ys[-1], short_model_name(model), c))
    x_lab = 86.5
    ax_b.axhline(0, color="black", lw=1.5, ls="--", dashes=(4, 2), zorder=2)
    ax_b.text(103.5, 0.006, "human record", fontsize=6, style="italic",
              color="0.25", va="bottom", ha="right")
    b_vals = [v for m in models for a in ages
              for v in (deltas[(m, a)][0] + deltas[(m, a)][1],
                        deltas[(m, a)][0] - deltas[(m, a)][1])]
    b_lo, b_hi = min(b_vals + [0.0]), max(b_vals + [0.0])
    pad = 0.18 * (b_hi - b_lo)
    ax_b.set_ylim(b_lo - pad, b_hi + pad)
    ylo, yhi = ax_b.get_ylim()
    label_ys = spread_labels([e[0] for e in ends], ylo, yhi, 0.105 * (yhi - ylo))
    for (y_end, name, c), y_lab in zip(ends, label_ys):
        is_h = name.startswith("human")
        ax_b.annotate(name, xy=(x_lab if is_h else 85, y_end),
                      xytext=(x_lab + 1.2, y_lab), fontsize=6,
                      color="black" if is_h else c, va="center", ha="left",
                      fontweight="bold" if is_h else "normal",
                      annotation_clip=False,
                      arrowprops={"arrowstyle": "-", "lw": 0.4, "color": "0.6",
                                  "shrinkA": 1, "shrinkB": 1})
    ax_b.set_ylabel("change from persona 65")
    ax_b.set_xlim(63, 104)
    ax_b.set_xticks([65, 70, 75, 80, 85])   # ticks stop where the data does
    ax_b.spines['bottom'].set_bounds(63, 86)
    ax_b.set_title("B. Change from persona 65 (paired 95% CI)", loc="left")
    style(ax_b)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure written to {out_pdf}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build model_curves.csv and fig1.pdf")
    ap.add_argument("--synthetic", action="store_true",
                    help="Fabricate plausible runs in runs_synthetic/ and plot from them")
    ap.add_argument("--runs-dir", default=None,
                    help="Runs directory to aggregate (default: config runs_dir)")
    ap.add_argument("--seed", type=int, default=7, help="Seed for --synthetic")
    ap.add_argument("--wide", action="store_true",
                    help="Single-column (NeurIPS) proportions instead of ACL column width")
    args = ap.parse_args()
    global WIDE
    WIDE = args.wide

    config = load_config()
    items = load_items(config)
    if args.synthetic:
        runs_dir = HARNESS_DIR / "runs_synthetic"
        generate_synthetic_runs(config, items, runs_dir, seed=args.seed)
        print("NOTE: plotting SYNTHETIC data; outputs overwrite results/model_curves.csv "
              "and results/fig1.pdf — re-run without --synthetic once real runs exist.")
    else:
        runs_dir = HARNESS_DIR / (args.runs_dir or config["runs_dir"])
        if not runs_dir.exists():
            raise SystemExit(f"ERROR: {runs_dir} does not exist. Run run_harness.py first, "
                             "or use --synthetic to test the pipeline.")

    rows = model_curves(config, items, runs_dir) + human_curve(items)
    results_dir = HARNESS_DIR / config["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "model_curves.csv"
    fieldnames = ["source", "model", "condition", "x_age", "age_bracket",
                  "mean_rating", "mean_within_sd", "ci95_mean", "ci95_sd",
                  "n_items", "n_ratings"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"Curves written to {csv_path} ({len(rows)} rows)")

    make_figure(config, rows, results_dir / "fig1.pdf", items, runs_dir)


if __name__ == "__main__":
    main()
