"""Command line.  python -m tokencost probe | run [--limit N] | analyze"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

from . import analysis, runner
from .backend import BackendError, OllamaBackend
from .config import DATASET, MODEL, OPTIONS, PRICE_INPUT_PER_M, PRICE_OUTPUT_PER_M, RESULTS_DIR, SYSTEM_PROMPT
from .counting import cache_check, estimate_overhead


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def cmd_probe(args):
    backend = OllamaBackend()
    overhead = estimate_overhead(backend)
    cache = cache_check(backend)
    data = {"time": datetime.now().isoformat(timespec="seconds"), "model": MODEL,
            "system_prompt": SYSTEM_PROMPT, "overhead": overhead, "cache_check": cache}
    write_json(RESULTS_DIR / "probe.json", data)
    print(json.dumps(data, indent=2, ensure_ascii=False))


def cmd_run(args):
    backend = OllamaBackend()
    rows = runner.load_dataset(DATASET)
    if args.limit:
        rows = rows[: args.limit]
    overhead = estimate_overhead(backend)
    write_json(RESULTS_DIR / "run_config.json", {
        "time": datetime.now().isoformat(timespec="seconds"), "model": MODEL, "options": OPTIONS,
        "system_prompt": SYSTEM_PROMPT, "price_input_per_million": PRICE_INPUT_PER_M,
        "price_output_per_million": PRICE_OUTPUT_PER_M, "fresh_cache_per_call": not args.no_fresh,
        "pairs": len(rows), "overhead": overhead})
    written = runner.run(rows, backend, RESULTS_DIR / "results.csv",
                         fresh=not args.no_fresh, resume=not args.restart)
    print(f"done: {written} new calls written to {RESULTS_DIR / 'results.csv'}")


def cmd_analyze(args):
    summary = analysis.main()
    o = summary["overall_ratio"]
    print(f"pairs: {summary['pairs']}")
    print(f"overall input ratio (BN/EN): {o['input']:.3f}")
    print(f"overall output ratio (BN/EN): {o['output']:.3f}")
    print(f"overall cost ratio (BN/EN): {o['cost']:.3f}")
    ci = summary["ratio_per_pair"]["input"]["ci95_of_mean"]
    print(f"per-pair input ratio mean {summary['ratio_per_pair']['input']['mean']:.3f}, "
          f"95% bootstrap CI of mean [{ci[0]:.3f}, {ci[1]:.3f}]")
    print(f"written: {RESULTS_DIR / 'comparison.csv'}, summary.json, chart.png")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m tokencost", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe", help="measure template overhead and check prompt caching").set_defaults(func=cmd_probe)
    p_run = sub.add_parser("run", help="run all pairs in both languages")
    p_run.add_argument("--limit", type=int, help="only the first N pairs (quick trial)")
    p_run.add_argument("--no-fresh", action="store_true", help="keep the cache between calls (faster, less clean)")
    p_run.add_argument("--restart", action="store_true", help="ignore and overwrite results.csv")
    p_run.set_defaults(func=cmd_run)
    sub.add_parser("analyze", help="build comparison.csv, summary.json and chart.png").set_defaults(func=cmd_analyze)
    args = ap.parse_args(argv)
    try:
        args.func(args)
    except BackendError as exc:
        print(f"Error: {exc}\nStart Ollama (ollama serve) and check: ollama pull {MODEL}")
        sys.exit(1)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
