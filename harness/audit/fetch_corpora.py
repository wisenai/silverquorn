#!/usr/bin/env python3
"""Fetch the annotator tables behind the dataset audit (Table 2).

Each dataset stays with its authors; this downloads their released files so
audit_age.py can recompute the table from source instead of trusting a
transcription. Nothing here is redistributed.

One source is dead. NLPositionality's official endpoints
(delphi-litw / toxicity-litw .apps.allenai.org) stopped responding before the
paper was written, and the mirror used instead (HuggingFace
`asaakyan/culture-safety`, documented as an exact download and cross-checked
759/759 against the authors' own processed release) has since gone private.
The project page is also down. That row therefore cannot currently be
reverified by anyone, which audit_age.py reports rather than hides.

Usage: python3 fetch_corpora.py [--force]
"""

from __future__ import annotations

import argparse
import io
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

GH = "https://raw.githubusercontent.com"
SOURCES = {
    "popquorn_politeness.csv":
        f"{GH}/Jiaxin-Pei/potato-prolific-dataset/main/dataset/politeness_rating/raw_data.csv",
    "popquorn_offensiveness.csv":
        f"{GH}/Jiaxin-Pei/potato-prolific-dataset/main/dataset/offensiveness/raw_data.csv",
    "d3code_raters.csv":
        f"{GH}/google-research-datasets/d3code/main/dataset/d3-raters.csv",
    "dices350.csv":
        f"{GH}/google-research-datasets/dices-dataset/main/350/"
        "diverse_safety_adversarial_dialog_350.csv",
    "prism_survey.jsonl":
        "https://huggingface.co/datasets/HannahRoseKirk/prism-alignment/"
        "resolve/main/survey.jsonl",
    "mhsc.parquet":
        "https://huggingface.co/api/datasets/ucberkeley-dlab/"
        "measuring-hate-speech/parquet/default/train/0.parquet",
}

# Shipped as a tarball; we want one member out of it.
SBIC_URL = "https://homes.cs.washington.edu/~msap/social-bias-frames/SBIC.v2.tgz"
SBIC_MEMBERS = ["SBIC.v2.trn.csv", "SBIC.v2.dev.csv", "SBIC.v2.tst.csv"]

DEAD = {
    "nlpositionality": [
        "https://delphi-litw.apps.allenai.org/api/v1/dataset?type=raw",
        "https://toxicity-litw.apps.allenai.org/api/v1/dataset?type=raw",
        "https://huggingface.co/datasets/asaakyan/culture-safety (mirror, now private)",
    ]
}


def get(url: str, dest: Path, force: bool) -> bool:
    if dest.exists() and not force:
        print(f"  {dest.name}: present ({dest.stat().st_size:,} B)")
        return True
    try:
        print(f"  {dest.name}: downloading ...")
        with urllib.request.urlopen(url, timeout=300) as r:
            dest.write_bytes(r.read())
        print(f"    ok ({dest.stat().st_size:,} B)")
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        print(f"    FAILED: {e}")
        return False


def get_sbic(force: bool) -> bool:
    if (DATA / SBIC_MEMBERS[0]).exists() and not force:
        print(f"  SBIC: present")
        return True
    try:
        print("  SBIC: downloading tarball ...")
        with urllib.request.urlopen(SBIC_URL, timeout=600) as r:
            blob = r.read()
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
            for m in SBIC_MEMBERS:
                src = tf.extractfile(m)
                if src is None:
                    print(f"    FAILED: {m} not in archive")
                    return False
                (DATA / m).write_bytes(src.read())
        print(f"    ok ({len(SBIC_MEMBERS)} splits)")
        return True
    except Exception as e:
        print(f"    FAILED: {e}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download everything")
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)

    print("Fetching annotator tables")
    ok = [get(u, DATA / n, args.force) for n, u in SOURCES.items()]
    ok.append(get_sbic(args.force))

    print("\nKnown-dead source (reported, not fetched):")
    for name, urls in DEAD.items():
        print(f"  {name}")
        for u in urls:
            print(f"    - {u}")

    failed = ok.count(False)
    print(f"\n{len(ok) - failed}/{len(ok)} sources available.")
    print("Next: python3 audit_age.py")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
