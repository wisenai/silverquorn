#!/usr/bin/env python3
"""How many raters per item per band would an elder validation target need?

Section 5 promises a recruitment number. The MHSC power analysis
(run_mhsc_val.py) shows contrast reliability of zero between every pair of
elder bands, but a reliability of zero is a verdict on the existing pool, not
a design target: true between-band variance is estimated at or below zero
there, so no required-n follows from it directly.

What does follow is a resolution budget. Fix a difference you want to be able
to detect between two elder bands, take the pooled within-band per-rater SD
this item actually shows, and solve for n. The minimum resolvable difference
for a two-band contrast with n raters each is

    MRD = 1.96 * sigma * sqrt(2 / n)   ->   n = 2 * sigma^2 * (1.96 / MRD)^2

sigma is pooled across the elder bands with (n-1) weights, over the 70-comment
calibration set the validation used. We deliberately do NOT reuse the median of
per-comment SEs that produces the reported 0.26--0.65 MRD figures: a majority of
elder cells there hold two or three raters who happen to agree, so their SE is
exactly zero and the median understates the spread. Pooling is the conservative
choice, which is the right direction for a number a study would budget against.

Usage: python3 power_target.py
Output: results/power_target.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
ELDER_BANDS = ["65-69", "70-74", "75-81"]
TARGETS = [0.05, 0.10, 0.15, 0.20, 0.25]
Z = 1.96


def main() -> None:
    val = json.loads((RESULTS / "mhsc_validation.json").read_text())
    targets = val["human_targets"]

    # Pooled within-band per-rater variance, (n-1)-weighted, elder bands only.
    num = den = 0.0
    cells = 0
    for bands in targets.values():
        for band in ELDER_BANDS:
            cell = bands.get(band)
            if not cell or cell.get("n", 0) < 2 or cell.get("sd") is None:
                continue
            num += (cell["n"] - 1) * cell["sd"] ** 2
            den += cell["n"] - 1
            cells += 1
    sigma2 = num / den
    sigma = math.sqrt(sigma2)

    out = {
        "pooled_within_band_per_rater_sd": round(sigma, 4),
        "df": int(den),
        "n_band_cells_pooled": cells,
        "formula": "n = 2 * sigma^2 * (1.96 / MRD)^2, per band, equal n",
        "raters_per_item_per_band": {
            f"{t:.2f}": math.ceil(2 * sigma2 * (Z / t) ** 2) for t in TARGETS
        },
        "reference_depth_in_corpus": {
            band: val["feasibility"]["core"]["bands"][band]["raters_per_comment_median"]
            for band in ["18-29", "45-54", "55-64", *ELDER_BANDS]
        },
        "resolved_human_gap_18_29_vs_70_74": val["human_age_gradient"]["pairs"][
            "18-29 vs 70-74"
        ]["mean_diff"],
    }
    (RESULTS / "power_target.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
