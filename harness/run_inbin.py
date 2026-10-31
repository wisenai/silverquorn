#!/usr/bin/env python3
"""In-bin dispersion control: does an ensemble of elder personas recover the
disagreement that real elders show?

The main harness asks each model for one persona age at a time and resamples it
(measuring decoding stochasticity). Reviewers rightly note that is not the same
quantity as disagreement across a bracket of people. This script asks the
complementary question with the resolution POPQUORN's >65 bin denies: prompt
each model at eight distinct ages *inside* the bin (66..88) on the 40 pilot
items, then measure per-item SD across those eight simulated elders and compare
it to the per-item SD across our five real elders on the same items.

Only the persona age varies. We deliberately do not invent personalities,
occupations, or traits: the comparison is about whether age resolution alone
produces population-like dispersion, which is exactly what the bin removes.

Usage: OPENROUTER_API_KEY=... python3 run_inbin.py [--limit-items N]
Output: runs_inbin/<model>.jsonl (resumable), results/inbin_dispersion.md/.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from harness_common import (
    HARNESS_DIR, STRICT_SUFFIX, build_messages, get_api_key, load_config,
    model_slug_to_filename, parse_rating,
)

IN_BIN_AGES = [66, 69, 72, 75, 78, 81, 85, 88]
URL = "https://openrouter.ai/api/v1/chat/completions"


def call(model: str, messages: list, key: str, max_tokens: int) -> tuple:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0.7, "usage": {"include": True}}
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=60))
            return r["choices"][0]["message"]["content"], r.get("usage", {}).get("cost")
        except urllib.error.HTTPError as e:
            if attempt == 3:
                return None, None
            time.sleep(2 ** attempt)
        except Exception:
            if attempt == 3:
                return None, None
            time.sleep(2 ** attempt)
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-items", type=int, default=None)
    args = ap.parse_args()

    key = get_api_key()
    cfg = load_config()
    items = [i for i in json.loads((HARNESS_DIR.parent / "app" / "src" / "items_40.json").read_text())
             if i["role"] == "item"]
    if args.limit_items:
        items = items[:args.limit_items]

    runs = HARNESS_DIR / "runs_inbin"
    runs.mkdir(exist_ok=True)
    total_cost = 0.0

    for model in cfg["models"]:
        path = runs / model_slug_to_filename(model)
        done = set()
        if path.exists():
            for line in path.read_text().splitlines():
                rec = json.loads(line)
                if rec.get("parsed") is not None:
                    done.add((rec["persona_age"], rec["item_id"]))
        todo = [(a, it) for a in IN_BIN_AGES for it in items
                if (a, str(it["item_id"])) not in done]
        if not todo:
            print(f"{model}: complete")
            continue

        def work(job):
            age, it = job
            msgs = build_messages(age, it["text"])
            raw, cost = call(model, msgs, key, cfg["max_tokens"])
            val = parse_rating(raw) if raw else None
            if val is None and raw is not None:
                msgs2 = list(msgs)
                msgs2[-1] = {**msgs2[-1], "content": msgs2[-1]["content"] + STRICT_SUFFIX}
                raw, c2 = call(model, msgs2, key, cfg["max_tokens"])
                val = parse_rating(raw) if raw else None
                cost = (cost or 0) + (c2 or 0)
            return {"model": model, "persona_age": age, "item_id": str(it["item_id"]),
                    "parsed": val, "raw_response": raw, "cost_usd": cost}

        with ThreadPoolExecutor(max_workers=8) as ex, open(path, "a") as f:
            for i, rec in enumerate(ex.map(work, todo), 1):
                f.write(json.dumps(rec) + "\n")
                total_cost += rec.get("cost_usd") or 0
                if i % 80 == 0:
                    print(f"  {model}: {i}/{len(todo)}  ${total_cost:.2f}")
        print(f"{model}: done ({len(todo)} cells)")

    # ---- analysis -----------------------------------------------------------
    pilot = HARNESS_DIR.parent / "data-exports" / "2026-07-29-FROZEN" / "ratings_long.csv"
    import pandas as pd
    df = pd.read_csv(pilot)
    real = df[df["role"] == "item"]
    counts = real.groupby("participant")["item_id"].nunique()
    keep = counts[counts >= 30].index
    real = real[real["participant"].isin(keep)]
    real_sd = real.groupby("item_id")["rating"].std().dropna()

    out = {"in_bin_ages": IN_BIN_AGES,
           "real_elder_median_per_item_sd": round(float(real_sd.median()), 3),
           "real_elder_n": int(real["participant"].nunique()), "models": {}}
    lines = ["# In-bin dispersion control", "",
             f"Eight simulated elders (ages {IN_BIN_AGES}) vs. "
             f"{out['real_elder_n']} real elders, per-item SD on the same 40 items.", "",
             f"Real elders: median per-item SD = **{out['real_elder_median_per_item_sd']}**", "",
             "| model | median per-item SD across 8 in-bin personas | ratio to real |", "|---|---|---|"]
    for model in cfg["models"]:
        path = runs / model_slug_to_filename(model)
        if not path.exists():
            continue
        recs = [json.loads(l) for l in path.read_text().splitlines()]
        by_item = {}
        for r in recs:
            if r.get("parsed") is not None:
                by_item.setdefault(r["item_id"], {})[r["persona_age"]] = r["parsed"]
        sds = [float(np.std(list(v.values()), ddof=1)) for v in by_item.values() if len(v) >= 2]
        med = float(np.median(sds)) if sds else math.nan
        ratio = med / out["real_elder_median_per_item_sd"] if med == med else math.nan
        out["models"][model] = {"median_per_item_sd": round(med, 3),
                                "n_items": len(sds), "ratio_to_real": round(ratio, 2)}
        lines.append(f"| {model} | {med:.3f} | {ratio:.2f}x |")
    (HARNESS_DIR / "results" / "inbin_dispersion.json").write_text(json.dumps(out, indent=1))
    (HARNESS_DIR / "results" / "inbin_dispersion.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[3:]))
    print(f"\ntotal cost this run: ${total_cost:.2f}")


if __name__ == "__main__":
    main()
