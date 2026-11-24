"""Score raw outputs: invalid rate, field accuracy, cost; spread over runs.
python src/score.py [--dir results]"""
import argparse, json, pathlib, re, statistics as st, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from validator import validate

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIELDS = ["domain", "notice_type", "registrar", "issue_date", "deadline_date", "amount", "currency", "action_required"]


def parse(raw):
    """Parse whatever came back: strict JSON, else strip a code fence / surrounding prose."""
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


def same(f, a, b):
    if f == "amount":
        return a is not None and b is not None and isinstance(a, (int, float)) and abs(a - b) < 0.005 or a == b
    if f == "domain" and isinstance(a, str):
        return a.strip().lower() == str(b).lower()
    if isinstance(a, str) and isinstance(b, str):
        return a.strip() == b.strip()
    return a == b and type(a) == type(b)


def spread(xs):
    return {"mean": st.mean(xs), "stdev": st.stdev(xs) if len(xs) > 1 else 0.0, "min": min(xs), "max": max(xs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    a = ap.parse_args()
    d = ROOT / a.dir
    schema = json.loads((ROOT / "schema/notice.schema.json").read_text())
    keys = {r["id"]: r["fields"] for r in map(json.loads, (ROOT / "corpus/answer_key.jsonl").read_text().splitlines()) if r}
    summary = {}
    for method in ("A", "B"):
        runs = []
        for p in sorted((d / "raw").glob(f"{method}_run*.jsonl")):
            rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
            invalid = correct = 0
            fcorrect = {f: 0 for f in FIELDS}
            for r in rows:
                obj = parse(r["raw"])
                if obj is None or validate(obj, schema):
                    invalid += 1
                    continue
                for f in FIELDS:
                    if same(f, obj.get(f), keys[r["id"]][f]):
                        correct += 1
                        fcorrect[f] += 1
            n = len(rows)
            runs.append({"run": p.stem, "n": n, "invalid_rate": invalid / n,
                         "field_accuracy": correct / (n * len(FIELDS)),
                         "per_field_accuracy": {f: fcorrect[f] / n for f in FIELDS},
                         "cost_usd": sum(r["cost_usd"] for r in rows),
                         "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
                         "completion_tokens": sum(r["completion_tokens"] for r in rows)})
        summary[method] = {"runs": runs,
                           "invalid_rate": spread([r["invalid_rate"] for r in runs]),
                           "field_accuracy": spread([r["field_accuracy"] for r in runs]),
                           "cost_usd_per_run": spread([r["cost_usd"] for r in runs]),
                           "cost_usd_total": sum(r["cost_usd"] for r in runs)}
    (d / "summary.json").write_text(json.dumps(summary, indent=2))
    names = {"A": "A: prompt-only", "B": "B: grammar-constrained"}
    L = ["# Task 2 results", "", "| Method | Invalid rate (mean ± sd) | [min, max] | Field accuracy (mean ± sd) | [min, max] | Cost per run USD (mean ± sd) | Total cost USD (3 runs) |",
         "|---|---|---|---|---|---|---|"]
    for m in ("A", "B"):
        s = summary[m]; i, f, c = s["invalid_rate"], s["field_accuracy"], s["cost_usd_per_run"]
        L.append(f"| {names[m]} | {i['mean']:.3f} ± {i['stdev']:.3f} | [{i['min']:.3f}, {i['max']:.3f}] | "
                 f"{f['mean']:.3f} ± {f['stdev']:.3f} | [{f['min']:.3f}, {f['max']:.3f}] | "
                 f"{c['mean']:.5f} ± {c['stdev']:.5f} | {s['cost_usd_total']:.5f} |")
    L += ["", "## Per run", "", "| Run | Invalid rate | Field accuracy | Cost USD |", "|---|---|---|---|"]
    for m in ("A", "B"):
        for r in summary[m]["runs"]:
            L.append(f"| {m}/{r['run']} | {r['invalid_rate']:.3f} | {r['field_accuracy']:.3f} | {r['cost_usd']:.5f} |")
    (d / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
