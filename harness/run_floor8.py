#!/usr/bin/env python3
"""Same-age noise floor at n=8, matched to the in-bin control.

The in-bin control (run_inbin.py) takes one draw at each of eight ages and
reports the SD across them. Comparing that to a floor measured over three
draws is invalid: median-SD-at-zero depends strongly on n. This script
resamples ONE fixed age eight times per item, so the floor is measured with
the same estimator at the same n, and the comparison is exact.

Usage: OPENROUTER_API_KEY=... python3 run_floor8.py
Output: runs_floor8/<model>.jsonl (resumable), results/floor8.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from harness_common import HARNESS_DIR, load_config, model_slug_to_filename, get_api_key
sys.path.insert(0, str(HARNESS_DIR))
from run_inbin import call  # reuse the exact request/retry path
from harness_common import build_messages, STRICT_SUFFIX, parse_rating

FIXED_AGE = 75      # mid-bin; the age the paper's fidelity claim uses
N_DRAWS = 8         # matched to the eight in-bin ages

def main() -> None:
    key = get_api_key(); cfg = load_config()
    items = [i for i in json.loads((HARNESS_DIR.parent / "app" / "src" / "items_40.json").read_text())
             if i["role"] == "item"]
    runs = HARNESS_DIR / "runs_floor8"; runs.mkdir(exist_ok=True)
    cost = 0.0
    for model in cfg["models"]:
        path = runs / model_slug_to_filename(model)
        done = {}
        if path.exists():
            for line in path.read_text().splitlines():
                r = json.loads(line)
                if r.get("parsed") is not None:
                    done.setdefault(r["item_id"], []).append(r["parsed"])
        todo = [(it, d) for it in items for d in range(N_DRAWS)
                if len(done.get(str(it["item_id"]), [])) <= d]
        if not todo:
            print(f"{model}: complete"); continue
        def work(job):
            it, d = job
            msgs = build_messages(FIXED_AGE, it["text"])
            raw, c = call(model, msgs, key, cfg["max_tokens"])
            v = parse_rating(raw) if raw else None
            if v is None and raw is not None:
                m2 = list(msgs); m2[-1] = {**m2[-1], "content": m2[-1]["content"] + STRICT_SUFFIX}
                raw, c2 = call(model, m2, key, cfg["max_tokens"])
                v = parse_rating(raw) if raw else None
                c = (c or 0) + (c2 or 0)
            return {"model": model, "persona_age": FIXED_AGE, "item_id": str(it["item_id"]),
                    "draw": d, "parsed": v, "cost_usd": c}
        with ThreadPoolExecutor(max_workers=8) as ex, open(path, "a") as f:
            for rec in ex.map(work, todo):
                f.write(json.dumps(rec) + "\n"); cost += rec.get("cost_usd") or 0
        print(f"{model}: {len(todo)} calls, ${cost:.2f}")

    ib = json.loads((HARNESS_DIR / "results" / "inbin_dispersion.json").read_text())
    out = {"fixed_age": FIXED_AGE, "n_draws": N_DRAWS, "models": {}}
    print(f"\n{'model':<24} {'floor n=8':>10} {'in-bin n=8':>11} {'excess':>8}")
    for model in cfg["models"]:
        recs = [json.loads(l) for l in (runs / model_slug_to_filename(model)).read_text().splitlines()]
        by = {}
        for r in recs:
            if r.get("parsed") is not None:
                by.setdefault(r["item_id"], []).append(r["parsed"])
        sds = [float(np.std(v[:N_DRAWS], ddof=1)) for v in by.values() if len(v) >= 2]
        med = float(np.median(sds)); mean = float(np.mean(sds))
        share = float(np.mean([s > 0 for s in sds]))
        inb = ib["models"][model]
        out["models"][model] = {"floor_median": round(med, 3), "floor_mean": round(mean, 3),
                                "floor_share_varying": round(share, 3),
                                "inbin_median": inb["median_per_item_sd"],
                                "inbin_mean": inb["mean_per_item_sd"],
                                "inbin_share_varying": inb["share_items_varying"]}
        print(f"{model.split('/')[-1]:<24} {med:>10.3f} {inb['median_per_item_sd']:>11.3f} "
              f"{inb['median_per_item_sd'] - med:>+8.3f}")
    (HARNESS_DIR / "results" / "floor8.json").write_text(json.dumps(out, indent=1))
    print(f"\ntotal cost: ${cost:.2f}")

if __name__ == "__main__":
    main()
