#!/usr/bin/env python3
"""SilverQuorn persona age-extrapolation harness (HARNESS_SPEC §3).

Calls OpenRouter chat completions over the matrix:
    models x persona_ages x items x samples
with a resumable per-model JSONL cache under runs/, a strict 1-5 parser with
up to 2 stricter-instruction retries, and a cumulative-cost budget guardrail
fed by OpenRouter usage fields.

Transport: aiohttp (async) if importable, otherwise urllib in a thread pool.

Usage:
    python3 run_harness.py --dry-run
    python3 run_harness.py --limit-items 3 --models openai/gpt-4o
    python3 run_harness.py                     # full 28,800-call matrix
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from harness_common import (
    HARNESS_DIR,
    STRICT_SUFFIX,
    build_messages,
    cell_key,
    condition_label,
    get_api_key,
    load_config,
    load_items,
    load_run_lines,
    model_slug_to_filename,
    parse_rating,
    sum_cached_cost,
)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

try:
    import aiohttp  # type: ignore

    HAVE_AIOHTTP = True
except ImportError:
    HAVE_AIOHTTP = False


class BudgetExceeded(Exception):
    pass


class CostTracker:
    """Thread-safe cumulative cost from OpenRouter usage fields."""

    def __init__(self, budget_usd: float, initial_cost: float, state_file: Path):
        self.budget_usd = budget_usd
        self.total = initial_cost
        self.state_file = state_file
        self._lock = threading.Lock()
        self.stop_event = threading.Event()

    def add(self, cost: float | None) -> None:
        with self._lock:
            if isinstance(cost, (int, float)):
                self.total += cost
            if self.total >= self.budget_usd:
                self.stop_event.set()

    def exceeded(self) -> bool:
        return self.stop_event.is_set()

    def save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = {"cumulative_cost_usd": round(self.total, 6), "budget_usd": self.budget_usd,
                       "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1)


class CacheWriter:
    """Thread-safe append-only JSONL writer, one file per model."""

    def __init__(self, runs_dir: Path):
        self.runs_dir = runs_dir
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._handles: dict[str, object] = {}

    def write(self, model: str, record: dict) -> None:
        with self._lock:
            fh = self._handles.get(model)
            if fh is None:
                fh = open(self.runs_dir / model_slug_to_filename(model), "a", encoding="utf-8")
                self._handles[model] = fh
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()

    def close(self) -> None:
        with self._lock:
            for fh in self._handles.values():
                fh.close()
            self._handles.clear()


def request_once_urllib(api_key: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://silverquorn.local",
            "X-Title": "SilverQuorn persona harness",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_with_network_retry(api_key: str, body: dict, cfg_req: dict) -> tuple[dict | None, str | None]:
    """Returns (response_json, error_string). Retries 429/5xx/timeouts with backoff."""
    retries = int(cfg_req.get("network_retries", 5))
    base = float(cfg_req.get("backoff_base_seconds", 2.0))
    timeout = float(cfg_req.get("timeout_seconds", 60))
    last_err = None
    for attempt in range(retries):
        try:
            resp = request_once_urllib(api_key, body, timeout)
            # OpenRouter sometimes returns 200 with an error payload.
            if isinstance(resp, dict) and "error" in resp and "choices" not in resp:
                last_err = f"api_error: {resp['error']}"
            else:
                return resp, None
        except urllib.error.HTTPError as e:
            last_err = f"http_{e.code}"
            if e.code not in (408, 425, 429, 500, 502, 503, 504):
                return None, last_err  # non-retryable
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = f"network: {e}"
        time.sleep(base * (2 ** attempt) + random.uniform(0, 0.5))
    return None, last_err


def extract_response_fields(resp: dict) -> tuple[str | None, dict]:
    """Pull (content, meta) out of an OpenRouter response."""
    content = None
    try:
        content = resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        pass
    usage = resp.get("usage") or {}
    meta = {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cost_usd": usage.get("cost"),
        "provider": resp.get("provider"),
        "response_model": resp.get("model"),
    }
    return content, meta


def process_cell(api_key: str, config: dict, model: str, persona_age: int | None,
                 item: dict, sample_idx: int, tracker: CostTracker, writer: CacheWriter) -> dict:
    """Run one matrix cell: up to 1 + parse_retries API calls until parseable.

    Every attempt is appended to the cache; the last line per cell wins on load.
    """
    parse_retries = int(config.get("parse_retries", 2))
    record = {}
    for attempt in range(parse_retries + 1):
        if tracker.exceeded():
            raise BudgetExceeded()
        body = {
            "model": model,
            "messages": build_messages(persona_age, item["text"], strict=attempt > 0),
            "temperature": config["temperature"],
            "max_tokens": config.get("max_tokens", 8),
            "usage": {"include": True},
        }
        resp, err = call_with_network_retry(api_key, body, config.get("request", {}))
        content, meta = (None, {}) if resp is None else extract_response_fields(resp)
        parsed = parse_rating(content)
        record = {
            "model": model,
            "persona_age": persona_age,
            "item_id": str(item["item_id"]),
            "sample_idx": sample_idx,
            "attempt": attempt,
            "raw_response": content,
            "parsed": parsed,
            "error": err,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **meta,
        }
        tracker.add(meta.get("cost_usd"))
        writer.write(model, record)
        if parsed is not None:
            break
    return record


def build_pending(config: dict, models: list[str], items: list[dict], runs_dir: Path):
    cached = load_run_lines(runs_dir, models)
    pending = []
    n_cached_ok = 0
    for model in models:
        cells = cached[model]
        for persona_age in config["persona_ages"]:
            for item in items:
                for s in range(config["samples"]):
                    key = cell_key(persona_age, item["item_id"], s)
                    rec = cells.get(key)
                    if rec is not None and rec.get("parsed") is not None:
                        n_cached_ok += 1
                        continue
                    pending.append((model, persona_age, item, s))
    return pending, n_cached_ok, cached


def estimate_cost(config: dict, models: list[str], n_calls_per_model: int, avg_item_tokens: float) -> float:
    prices = config.get("cost_tracking", {}).get("pricing_estimates_per_mtok", {})
    in_tok = 55 + avg_item_tokens  # instruction + persona + chat overhead + item
    out_tok = 3.0
    total = 0.0
    for m in models:
        p = prices.get(m, {"input": 3.0, "output": 15.0})
        total += n_calls_per_model * (in_tok * p["input"] + out_tok * p["output"]) / 1e6
    return total


def dry_run(config: dict, models: list[str], items: list[dict]) -> None:
    print("=== DRY RUN (no network calls) ===\n")
    examples = [(None, items[0]), (65, items[1 % len(items)]), (85, items[2 % len(items)])]
    for i, (age, item) in enumerate(examples, 1):
        print(f"--- Example prompt {i}: condition={condition_label(age)}, item_id={item['item_id']} ---")
        for msg in build_messages(age, item["text"]):
            print(f"[{msg['role']}]\n{msg['content']}\n")
        print(f"(on parse failure, retried with appended: {STRICT_SUFFIX!r})\n")
    n_conditions = len(config["persona_ages"])
    n_per_model = n_conditions * len(items) * config["samples"]
    n_total = n_per_model * len(models)
    avg_tokens = sum(len(it["text"]) for it in items) / len(items) / 4.0
    est = estimate_cost(config, models, n_per_model, avg_tokens)
    print("=== Call matrix ===")
    print(f"models:            {len(models)}  ({', '.join(models)})")
    print(f"persona conditions: {n_conditions}  ({', '.join(condition_label(a) for a in config['persona_ages'])})")
    print(f"items:             {len(items)}")
    print(f"samples/cell:      {config['samples']}")
    print(f"TOTAL CALLS:       {n_total}  ({n_per_model} per model; excludes parse-failure retries)")
    print(f"est. cost:         ~${est:.2f} (rough, from config pricing estimates; "
          f"avg item ~{avg_tokens:.0f} tok)  budget: ${config['budget_usd']:.2f}")


def run_threaded(api_key, config, pending, tracker, writer, max_workers):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    done = failed = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(process_cell, api_key, config, m, a, it, s, tracker, writer): (m, a, it, s)
                   for (m, a, it, s) in pending}
        try:
            for fut in as_completed(futures):
                try:
                    rec = fut.result()
                    done += 1
                    if rec.get("parsed") is None:
                        failed += 1
                except BudgetExceeded:
                    done += 1
                if done % 50 == 0 or done == len(pending):
                    rate = done / max(time.time() - t0, 1e-9)
                    print(f"  {done}/{len(pending)} cells  ({failed} unparsed)  "
                          f"${tracker.total:.2f} spent  {rate:.1f} cells/s", flush=True)
                    tracker.save()
                if tracker.exceeded():
                    for f2 in futures:
                        f2.cancel()
                    break
        except KeyboardInterrupt:
            print("\nInterrupted; cache is consistent, re-run to resume.")
            for f2 in futures:
                f2.cancel()
    return done, failed


async def run_async(api_key, config, pending, tracker, writer, max_workers):
    """aiohttp transport: same cell semantics, cooperative concurrency."""
    import asyncio

    cfg_req = config.get("request", {})
    timeout = aiohttp.ClientTimeout(total=float(cfg_req.get("timeout_seconds", 60)))
    sem = asyncio.Semaphore(max_workers)
    counters = {"done": 0, "failed": 0}
    t0 = time.time()

    async def call_api(session, body):
        retries = int(cfg_req.get("network_retries", 5))
        base = float(cfg_req.get("backoff_base_seconds", 2.0))
        last_err = None
        for attempt in range(retries):
            try:
                async with session.post(OPENROUTER_URL, json=body) as resp:
                    if resp.status in (408, 425, 429, 500, 502, 503, 504):
                        last_err = f"http_{resp.status}"
                    elif resp.status != 200:
                        return None, f"http_{resp.status}"
                    else:
                        data = await resp.json()
                        if isinstance(data, dict) and "error" in data and "choices" not in data:
                            last_err = f"api_error: {data['error']}"
                        else:
                            return data, None
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_err = f"network: {e}"
            await asyncio.sleep(base * (2 ** attempt) + random.uniform(0, 0.5))
        return None, last_err

    async def do_cell(session, model, persona_age, item, sample_idx):
        async with sem:
            for attempt in range(int(config.get("parse_retries", 2)) + 1):
                if tracker.exceeded():
                    return
                body = {
                    "model": model,
                    "messages": build_messages(persona_age, item["text"], strict=attempt > 0),
                    "temperature": config["temperature"],
                    "max_tokens": config.get("max_tokens", 8),
                    "usage": {"include": True},
                }
                resp, err = await call_api(session, body)
                content, meta = (None, {}) if resp is None else extract_response_fields(resp)
                parsed = parse_rating(content)
                writer.write(model, {
                    "model": model, "persona_age": persona_age, "item_id": str(item["item_id"]),
                    "sample_idx": sample_idx, "attempt": attempt, "raw_response": content,
                    "parsed": parsed, "error": err, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **meta,
                })
                tracker.add(meta.get("cost_usd"))
                if parsed is not None:
                    counters["done"] += 1
                    return
            counters["done"] += 1
            counters["failed"] += 1

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
               "HTTP-Referer": "https://silverquorn.local", "X-Title": "SilverQuorn persona harness"}
    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
        tasks = [asyncio.create_task(do_cell(session, m, a, it, s)) for (m, a, it, s) in pending]
        for i, t in enumerate(asyncio.as_completed(tasks), 1):
            await t
            if i % 50 == 0 or i == len(tasks):
                rate = i / max(time.time() - t0, 1e-9)
                print(f"  {i}/{len(tasks)} cells  ({counters['failed']} unparsed)  "
                      f"${tracker.total:.2f} spent  {rate:.1f} cells/s", flush=True)
                tracker.save()
            if tracker.exceeded():
                for t2 in tasks:
                    t2.cancel()
                break
    return counters["done"], counters["failed"]


def main() -> None:
    ap = argparse.ArgumentParser(description="SilverQuorn OpenRouter persona harness")
    ap.add_argument("--models", type=str, default=None,
                    help="Comma-separated subset of config models (default: all)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print 3 example prompts + call matrix count; no network")
    ap.add_argument("--limit-items", type=int, default=None, metavar="N",
                    help="Use only the first N items (smoke tests)")
    args = ap.parse_args()

    config = load_config()
    models = config["models"]
    if args.models:
        requested = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = [m for m in requested if m not in models]
        if unknown:
            sys.exit(f"ERROR: model(s) not in config.json: {unknown}\nConfigured: {models}")
        models = requested
    items = load_items(config, args.limit_items)

    if args.dry_run:
        dry_run(config, models, items)
        return

    api_key = get_api_key()
    runs_dir = HARNESS_DIR / config["runs_dir"]
    pending, n_cached_ok, cached = build_pending(config, models, items, runs_dir)
    initial_cost = sum_cached_cost(cached)
    total_cells = n_cached_ok + len(pending)
    print(f"Matrix: {total_cells} cells; {n_cached_ok} already cached with valid parse; "
          f"{len(pending)} to run. Cached spend so far: ${initial_cost:.2f} "
          f"(budget ${config['budget_usd']:.2f}).")
    if not pending:
        print("Nothing to do.")
        return
    if initial_cost >= config["budget_usd"]:
        sys.exit(f"ABORT: cached cumulative cost ${initial_cost:.2f} already exceeds "
                 f"budget ${config['budget_usd']:.2f}. Raise budget_usd in config.json to continue.")

    tracker = CostTracker(config["budget_usd"], initial_cost,
                          HARNESS_DIR / config["cost_tracking"]["cost_state_file"])
    writer = CacheWriter(runs_dir)
    max_workers = int(config.get("max_concurrent_requests", 8))
    print(f"Transport: {'aiohttp (async)' if HAVE_AIOHTTP else 'urllib in threads'}; "
          f"max concurrent = {max_workers}")
    try:
        if HAVE_AIOHTTP:
            import asyncio
            done, failed = asyncio.run(run_async(api_key, config, pending, tracker, writer, max_workers))
        else:
            done, failed = run_threaded(api_key, config, pending, tracker, writer, max_workers)
    finally:
        tracker.save()
        writer.close()

    if tracker.exceeded():
        print(f"\nSTOPPED: budget guardrail hit (${tracker.total:.2f} >= "
              f"${config['budget_usd']:.2f}). Cache is consistent; raise budget_usd and re-run to resume.")
    else:
        print(f"\nDone: {done} cells processed, {failed} final parse failures "
              f"(see qc.py), total spend ${tracker.total:.2f}.")


if __name__ == "__main__":
    main()
