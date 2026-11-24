"""Send every prompt in both languages to the model and write one CSV row per call."""
from __future__ import annotations

import csv
import re
from pathlib import Path

from .config import CATEGORIES, DATASET, LANGUAGES, SYSTEM_PROMPT
from .counting import count_call

COLUMNS = ["id", "category", "language", "prompt", "output", "input_tokens", "output_tokens",
           "cost_usd", "latency_s", "load_s", "done_reason", "truncated"]
BANGLA_CHARS = re.compile(r"[\u0980-\u09FF]")


def validate_dataset(rows: list[dict], expected: int = 50) -> bool:
    if len(rows) != expected:
        raise ValueError(f"expected {expected} pairs, found {len(rows)}")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("ids must be unique")
    for r in rows:
        if None in r:
            raise ValueError(f"pair {r.get('id')} has an extra tab character")
        for key in ("id", "category", "english", "bangla"):
            if not (r.get(key) or "").strip():
                raise ValueError(f"empty '{key}' in pair {r.get('id')}")
        if r["category"] not in CATEGORIES:
            raise ValueError(f"unknown category '{r['category']}' in pair {r['id']}")
        if not BANGLA_CHARS.search(r["bangla"]):
            raise ValueError(f"pair {r['id']}: the bangla column has no Bangla characters")
    return True


def load_dataset(path: Path = DATASET) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    validate_dataset(rows)
    return rows


def run(rows: list[dict], backend, out_path, system: str = SYSTEM_PROMPT, fresh: bool = True,
        resume: bool = True, progress=print) -> int:
    """Run all rows in both languages. Returns the number of new calls written.

    Order alternates per pair (English first on even rows, Bangla first on odd rows) so that
    any drift in server speed does not favour one language. Already-finished calls are skipped
    when resume=True, so an interrupted run can be continued.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    exists = out_path.exists() and out_path.stat().st_size > 0
    done = set()
    if resume and exists:
        with open(out_path, encoding="utf-8", newline="") as f:
            done = {(r["id"], r["language"]) for r in csv.DictReader(f)}
    append = resume and exists
    written = 0
    with open(out_path, "a" if append else "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if not append:
            writer.writeheader()
        for index, row in enumerate(rows):
            order = LANGUAGES if index % 2 == 0 else tuple(reversed(LANGUAGES))
            for lang in order:
                if (row["id"], lang) in done:
                    continue
                r = count_call(backend, row[lang], system=system, fresh=fresh)
                writer.writerow({
                    "id": row["id"], "category": row["category"], "language": lang,
                    "prompt": row[lang], "output": r["output"],
                    "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"],
                    "cost_usd": f"{r['cost_usd']:.8f}", "latency_s": f"{r['latency_s']:.3f}",
                    "load_s": f"{r['load_s']:.3f}", "done_reason": r["done_reason"] or "",
                    "truncated": r["done_reason"] == "length",
                })
                f.flush()
                written += 1
                if progress:
                    progress(f"{row['id']} {lang:<8} in={r['input_tokens']:>4} out={r['output_tokens']:>4} "
                             f"latency={r['latency_s']:.2f}s")
    return written
