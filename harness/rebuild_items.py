#!/usr/bin/env python3
"""Restore item text into the item files from your own POPQUORN download.

The item files in this release carry item IDs, POPQUORN statistics, and the
selection metadata, but no item text. POPQUORN's politeness items are drawn
from the Enron email corpus and contain real people's names, email addresses
and phone numbers. Those strings are already public in POPQUORN, so withholding
them protects nobody -- but re-publishing them in a second place, keyed and
tidied, spreads them further for no scientific gain. The IDs are sufficient:
this script joins them back to the file get_data.py downloaded, which restores
the exact text every experiment ran on.

Run this after harness/data/get_data.py and before any run_*.py script.

Usage: python3 rebuild_items.py [--check]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
RAW = HARNESS_DIR / "data" / "popquorn_politeness_raw.csv"
TARGETS = [HARNESS_DIR / "items_200.json",
           HARNESS_DIR.parent / "app" / "src" / "items_40.json"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report status without writing")
    args = ap.parse_args()

    if not RAW.exists():
        print(f"ERROR: {RAW} not found. Run harness/data/get_data.py first.")
        return 1

    import pandas as pd
    raw = pd.read_csv(RAW, usecols=["instance_id", "text"])
    text_by_id = {str(i): t for i, t in zip(raw["instance_id"], raw["text"])}
    # POPQUORN rates each instance by several annotators, so IDs repeat; the
    # text is identical across those rows and dict assignment collapses them.

    rc = 0
    for path in TARGETS:
        if not path.exists():
            print(f"skip {path.name}: not present")
            continue
        recs = json.loads(path.read_text())
        have = sum(1 for r in recs if r.get("text"))
        missing = [str(r["item_id"]) for r in recs
                   if str(r["item_id"]) not in text_by_id]
        if missing:
            print(f"ERROR {path.name}: {len(missing)} IDs absent from POPQUORN "
                  f"(first: {missing[:5]})")
            rc = 1
            continue
        if args.check:
            print(f"{path.name}: {len(recs)} records, {have} already have text")
            continue
        for r in recs:
            r["text"] = text_by_id[str(r["item_id"])]
        path.write_text(json.dumps(recs, indent=1) + "\n")
        print(f"{path.name}: restored text for {len(recs)} records")
    return rc


if __name__ == "__main__":
    sys.exit(main())
