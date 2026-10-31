#!/usr/bin/env python3
"""QC report for harness runs: parse rate and refusal rate per model x condition.

Reads runs/*.jsonl (last record per cell wins, matching run_harness resume
semantics), prints a markdown table to stdout, writes results/qc.md, and lists
cells that still need a re-run (missing entirely, or final parse failure).

Usage:
    python3 qc.py                     # against runs/
    python3 qc.py --runs-dir runs_synthetic
"""

from __future__ import annotations

import argparse
from pathlib import Path

from harness_common import (
    HARNESS_DIR,
    cell_key,
    condition_label,
    load_config,
    load_items,
    load_run_lines,
    looks_like_refusal,
)

MAX_LISTED_CELLS = 40  # cap the per-cell rerun listing in qc.md


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse/refusal QC for harness runs")
    ap.add_argument("--runs-dir", default=None,
                    help="Runs directory (default: config runs_dir, i.e. runs/)")
    args = ap.parse_args()

    config = load_config()
    runs_dir = HARNESS_DIR / (args.runs_dir or config["runs_dir"])
    items = load_items(config)
    models = config["models"]
    samples = config["samples"]
    cached = load_run_lines(runs_dir, models)

    rows = []
    rerun: dict[tuple[str, str], list] = {}
    for model in models:
        cells = cached[model]
        for persona_age in config["persona_ages"]:
            cond = condition_label(persona_age)
            expected = len(items) * samples
            attempted = parsed_ok = refusals = fmt_fail = errors = 0
            missing = []
            for item in items:
                for s in range(samples):
                    key = cell_key(persona_age, item["item_id"], s)
                    rec = cells.get(key)
                    if rec is None:
                        missing.append((item["item_id"], s))
                        continue
                    attempted += 1
                    if rec.get("parsed") is not None:
                        parsed_ok += 1
                    else:
                        missing.append((item["item_id"], s))  # needs re-run too
                        if rec.get("raw_response") is None:
                            errors += 1
                        elif looks_like_refusal(rec.get("raw_response")):
                            refusals += 1
                        else:
                            fmt_fail += 1
            parse_rate = parsed_ok / attempted if attempted else float("nan")
            refusal_rate = refusals / attempted if attempted else float("nan")
            rows.append({
                "model": model, "condition": cond, "expected": expected,
                "attempted": attempted, "parsed_ok": parsed_ok,
                "parse_rate": parse_rate, "refusals": refusals,
                "refusal_rate": refusal_rate, "format_fail": fmt_fail,
                "errors": errors, "needs_rerun": len(missing),
            })
            if missing:
                rerun[(model, cond)] = missing

    def fmt(x):
        return "n/a" if x != x else f"{x:.1%}"  # NaN check

    lines = [
        "# Harness QC report",
        "",
        f"Runs dir: `{runs_dir.relative_to(HARNESS_DIR)}` | items: {len(items)} | "
        f"samples/cell: {samples}",
        "",
        "Refusals are unparseable responses matching refusal phrasing; format "
        "failures are other unparseable responses; errors are cells with no "
        "response content (network/API). All three plus missing cells count as "
        "needs-rerun.",
        "",
        "| model | condition | expected | attempted | parsed | parse rate | "
        "refusals | refusal rate | format fail | errors | needs rerun |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['condition']} | {r['expected']} | {r['attempted']} "
            f"| {r['parsed_ok']} | {fmt(r['parse_rate'])} | {r['refusals']} "
            f"| {fmt(r['refusal_rate'])} | {r['format_fail']} | {r['errors']} "
            f"| {r['needs_rerun']} |"
        )

    lines += ["", "## Cells needing re-run", ""]
    total_rerun = sum(len(v) for v in rerun.values())
    if not rerun:
        lines.append("None. All cells have a valid parse.")
    else:
        lines.append(f"Total: {total_rerun} cells. Re-running `run_harness.py` "
                     "retries exactly these (cache skips valid parses).")
        lines.append("")
        for (model, cond), cells in sorted(rerun.items()):
            shown = ", ".join(f"{iid}#{s}" for iid, s in cells[:MAX_LISTED_CELLS])
            more = f" … +{len(cells) - MAX_LISTED_CELLS} more" if len(cells) > MAX_LISTED_CELLS else ""
            lines.append(f"- **{model} / {cond}** ({len(cells)}): {shown}{more}")

    report = "\n".join(lines) + "\n"
    print(report)
    out = HARNESS_DIR / config["results_dir"] / "qc.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"[written to {out}]")


if __name__ == "__main__":
    main()
