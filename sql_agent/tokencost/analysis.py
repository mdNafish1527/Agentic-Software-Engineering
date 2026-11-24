"""Bangla vs English: per-pair ratios, totals, bootstrap interval, categories, latency and chart."""
from __future__ import annotations

import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from .config import CATEGORIES, RESULTS_DIR


def read_results(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            r["input_tokens"] = int(r["input_tokens"])
            r["output_tokens"] = int(r["output_tokens"])
            r["cost_usd"] = float(r["cost_usd"])
            r["latency_s"] = float(r["latency_s"])
            rows.append(r)
    return rows


def ratio(numerator: float, denominator: float):
    return numerator / denominator if denominator else None


def pair_table(results: list[dict]) -> list[dict]:
    by_id = defaultdict(dict)
    for r in results:
        by_id[r["id"]][r["language"]] = r
    pairs = []
    for pid in sorted(by_id):
        d = by_id[pid]
        if set(d) != {"english", "bangla"}:
            continue  # incomplete pair: skipped, and reported by the caller if needed
        e, b = d["english"], d["bangla"]
        pairs.append({
            "id": pid, "category": e["category"],
            "en_input_tokens": e["input_tokens"], "bn_input_tokens": b["input_tokens"],
            "input_ratio": ratio(b["input_tokens"], e["input_tokens"]),
            "en_output_tokens": e["output_tokens"], "bn_output_tokens": b["output_tokens"],
            "output_ratio": ratio(b["output_tokens"], e["output_tokens"]),
            "en_cost_usd": e["cost_usd"], "bn_cost_usd": b["cost_usd"],
            "cost_ratio": ratio(b["cost_usd"], e["cost_usd"]),
            "en_latency_s": e["latency_s"], "bn_latency_s": b["latency_s"],
            "latency_ratio": ratio(b["latency_s"], e["latency_s"]),
        })
    return pairs


def bootstrap_ci(values: list[float], n_boot: int = 10_000, alpha: float = 0.05, seed: int = 12345):
    """Percentile bootstrap 95% interval for the MEAN of the values (fixed seed, reproducible)."""
    if len(values) < 2:
        return None, None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(statistics.fmean(rng.choices(values, k=n)) for _ in range(n_boot))
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return lo, hi


def describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    lo, hi = bootstrap_ci(values)
    return {"n": len(values), "mean": statistics.fmean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values), "ci95_of_mean": [lo, hi]}


def summarize(pairs: list[dict], results: list[dict]) -> dict:
    en = [r for r in results if r["language"] == "english"]
    bn = [r for r in results if r["language"] == "bangla"]

    def total(rows, key):
        return sum(r[key] for r in rows)

    summary = {
        "pairs": len(pairs),
        "ratio_per_pair": {
            "input": describe([p["input_ratio"] for p in pairs]),
            "output": describe([p["output_ratio"] for p in pairs if p["output_ratio"] is not None]),
            "cost": describe([p["cost_ratio"] for p in pairs if p["cost_ratio"] is not None]),
            "latency": describe([p["latency_ratio"] for p in pairs if p["latency_ratio"] is not None]),
        },
        "totals": {
            "english_input_tokens": total(en, "input_tokens"), "bangla_input_tokens": total(bn, "input_tokens"),
            "english_output_tokens": total(en, "output_tokens"), "bangla_output_tokens": total(bn, "output_tokens"),
            "english_cost_usd": total(en, "cost_usd"), "bangla_cost_usd": total(bn, "cost_usd"),
        },
        "overall_ratio": {
            "input": ratio(total(bn, "input_tokens"), total(en, "input_tokens")),
            "output": ratio(total(bn, "output_tokens"), total(en, "output_tokens")),
            "cost": ratio(total(bn, "cost_usd"), total(en, "cost_usd")),
        },
        "latency_s": {
            "english_mean": statistics.fmean(r["latency_s"] for r in en) if en else None,
            "english_median": statistics.median(r["latency_s"] for r in en) if en else None,
            "bangla_mean": statistics.fmean(r["latency_s"] for r in bn) if bn else None,
            "bangla_median": statistics.median(r["latency_s"] for r in bn) if bn else None,
        },
        "truncated_calls": {"english": sum(r["truncated"] == "True" for r in en),
                            "bangla": sum(r["truncated"] == "True" for r in bn)},
        "by_category": {},
    }
    for cat in CATEGORIES:
        ps = [p for p in pairs if p["category"] == cat]
        if not ps:
            continue
        ids = {p["id"] for p in ps}
        e_rows = [r for r in en if r["id"] in ids]
        b_rows = [r for r in bn if r["id"] in ids]
        summary["by_category"][cat] = {
            "pairs": len(ps),
            "input_ratio_of_totals": total(b_rows, "input_tokens") / total(e_rows, "input_tokens"),
            "cost_ratio_of_totals": total(b_rows, "cost_usd") / total(e_rows, "cost_usd"),
            "mean_pair_input_ratio": statistics.fmean(p["input_ratio"] for p in ps),
        }
    return summary


def _round(row: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}


def write_comparison(pairs: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pairs[0].keys()))
        writer.writeheader()
        for p in pairs:
            writer.writerow(_round(p))


def write_chart(pairs: list[dict], summary: dict, path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed, so the chart was skipped (pip install matplotlib).")
        return None
    cats = list(summary["by_category"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.bar(cats, [summary["by_category"][c]["input_ratio_of_totals"] for c in cats], color="#1d4e89")
    ax1.axhline(1.0, color="grey", linestyle="--", linewidth=1)
    ax1.set_title("Bangla / English input tokens by category")
    ax1.tick_params(axis="x", rotation=30)
    ordered = sorted(pairs, key=lambda p: p["input_ratio"])
    ax2.bar([p["id"] for p in ordered], [p["input_ratio"] for p in ordered], color="#2a9d8f")
    ax2.axhline(1.0, color="grey", linestyle="--", linewidth=1)
    ax2.set_title("Bangla / English input tokens per pair")
    ax2.tick_params(axis="x", rotation=90, labelsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(results_path: Path = RESULTS_DIR / "results.csv", out_dir: Path | None = None) -> dict:
    results_path = Path(results_path)
    out_dir = Path(out_dir) if out_dir else results_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    results = read_results(results_path)
    pairs = pair_table(results)
    if not pairs:
        raise ValueError("no complete English/Bangla pairs found in results")
    summary = summarize(pairs, results)
    write_comparison(pairs, out_dir / "comparison.csv")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_chart(pairs, summary, out_dir / "chart.png")
    return summary
