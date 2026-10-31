"""Shared helpers for the SilverQuorn persona age-extrapolation harness.

Used by run_harness.py, qc.py, and make_fig1.py so that prompt construction,
parsing, cache-key semantics, and JSONL loading are defined exactly once.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent

RATING_INSTRUCTION = (
    "Rate how polite the following message sounds to you, from 1 (not polite "
    "at all) to 5 (very polite). Reply with only the number."
)
STRICT_SUFFIX = "Reply with ONLY a single digit from 1 to 5."

# Accept exactly one integer 1-5, allowing surrounding whitespace/punctuation
# (e.g. "4", " 4.", "**4**") but rejecting anything with letters, extra
# digits, or other word characters ("rating: 4", "45", "4 out of 5").
PARSE_RE = re.compile(r"^[\s\W]*([1-5])[\s\W]*$")

REFUSAL_PATTERNS = re.compile(
    r"(i can'?t|i cannot|i'?m sorry|i am sorry|as an ai|i won'?t|unable to|"
    r"i'?m not able|i am not able|cannot assist|can'?t assist)",
    re.IGNORECASE,
)


def load_config(path: Path | None = None) -> dict:
    with open(path or HARNESS_DIR / "config.json", encoding="utf-8") as f:
        return json.load(f)


def load_items(config: dict, limit: int | None = None) -> list[dict]:
    with open(HARNESS_DIR / config["items_file"], encoding="utf-8") as f:
        items = json.load(f)
    if limit is not None:
        items = items[:limit]
    return items


def persona_system_message(persona_age: int | None) -> str | None:
    """Persona sentence per HARNESS_SPEC §3; None means no system message."""
    if persona_age is None:
        return None
    article = "an" if str(persona_age).startswith("8") else "a"
    return f"You are {article} {persona_age}-year-old adult."


def build_messages(persona_age: int | None, item_text: str, strict: bool = False) -> list[dict]:
    user = f"{RATING_INSTRUCTION}\n\n{item_text}"
    if strict:
        user = f"{user}\n\n{STRICT_SUFFIX}"
    messages = []
    system = persona_system_message(persona_age)
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return messages


def parse_rating(raw: str | None) -> int | None:
    if raw is None:
        return None
    m = PARSE_RE.match(raw)
    return int(m.group(1)) if m else None


def looks_like_refusal(raw: str | None) -> bool:
    return bool(raw) and bool(REFUSAL_PATTERNS.search(raw))


def cell_key(persona_age: int | None, item_id: str, sample_idx: int) -> tuple:
    """Cache key within a model file. persona_age None == neutral condition."""
    return ("neutral" if persona_age is None else int(persona_age), str(item_id), int(sample_idx))


def model_slug_to_filename(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", model) + ".jsonl"


def condition_label(persona_age) -> str:
    return "neutral" if persona_age is None else str(persona_age)


def load_run_lines(runs_dir: Path, models: list[str]) -> dict[str, dict[tuple, dict]]:
    """Load JSONL caches. Returns {model: {cell_key: last_record_for_cell}}.

    Later lines override earlier ones for the same cell, so re-runs of
    previously failed cells supersede the failure record.
    """
    out: dict[str, dict[tuple, dict]] = {}
    for model in models:
        path = runs_dir / model_slug_to_filename(model)
        cells: dict[tuple, dict] = {}
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = cell_key(rec.get("persona_age"), rec.get("item_id"), rec.get("sample_idx", 0))
                    prev = cells.get(key)
                    # Prefer a parsed record over an unparsed one; otherwise last wins.
                    if prev is None or rec.get("parsed") is not None or prev.get("parsed") is None:
                        cells[key] = rec
        out[model] = cells
    return out


def sum_cached_cost(run_cells: dict[str, dict[tuple, dict]]) -> float:
    total = 0.0
    for cells in run_cells.values():
        for rec in cells.values():
            c = rec.get("cost_usd")
            if isinstance(c, (int, float)):
                total += c
    return total


def get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "ERROR: OPENROUTER_API_KEY is not set.\n"
            "Export it before running, e.g.:\n"
            "  export OPENROUTER_API_KEY=sk-or-...\n"
            "(Get a key at https://openrouter.ai/keys)"
        )
    return key
