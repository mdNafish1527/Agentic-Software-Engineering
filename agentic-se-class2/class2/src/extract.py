"""Run Method A (prompt-only) and Method B (grammar-constrained) over the corpus, 3 runs each.
python src/extract.py --model qwen2.5:7b --runs 3 [--mock]"""
import argparse, json, pathlib, random, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ledger import Ledger

ROOT = pathlib.Path(__file__).resolve().parent.parent

SYSTEM = "You extract structured data from domain-name notices. Output JSON only."
PROMPT = """Extract the fields below from the domain notice and answer with one JSON object and nothing else.

JSON format (schema version {ver}):
{{
  "domain": string,                      // the domain name concerned
  "notice_type": one of "expiry_reminder" | "renewal_confirmation" | "transfer_request" | "suspension_notice" | "dispute_notice" | "verification_request",
  "registrar": string,                   // sender / registrar name
  "issue_date": "YYYY-MM-DD",            // date of the notice
  "deadline_date": "YYYY-MM-DD" or null, // date by which the recipient must act or the domain expires
  "amount": number or null,              // fee or payment amount, no thousands separator
  "currency": 3-letter ISO code or null,
  "action_required": true or false       // true if the recipient must do something
}}
Use null when a value is not stated. Dates must be ISO format.

Notice:
{text}
"""


def load_schema():
    return json.loads((ROOT / "schema/notice.schema.json").read_text())


def server_schema(schema):
    """Schema passed to the server: drop non-standard x- keys."""
    return {k: v for k, v in schema.items() if not k.startswith("x-") and k != "$schema"}


def make_mock(keys):
    import re
    rng = random.Random(7)

    def mock(tag, prompt):
        nid = next(n for n, k in keys.items() if k["domain"] in prompt)
        d = dict(keys[nid])
        method = tag.split("|")[1]
        if rng.random() < 0.15:
            d["amount"] = None if d["amount"] else 1.0
        if method == "A":
            r = rng.random()
            txt = json.dumps(d)
            if r < 0.15: txt = "Here is the JSON:\n```json\n" + txt + "\n```"
            elif r < 0.25: txt = txt.replace('"action_required": true', '"action_required": "yes"')
            elif r < 0.30: txt = txt[:-5]
        else:
            txt = json.dumps(d)
        return txt, 520 + len(prompt) // 4, 90 + rng.randint(0, 20)
    return mock


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-predict", type=int, default=300)
    ap.add_argument("--out", default="results")
    ap.add_argument("--mock", action="store_true", help="pipeline self-test, no server")
    a = ap.parse_args()

    out = ROOT / (a.out + ("_mock" if a.mock else ""))
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "ledger.jsonl").unlink(missing_ok=True)
    schema = load_schema()
    ver = schema["x-schema-version"]
    notices = [json.loads(l) for l in (ROOT / "corpus/notices.jsonl").read_text().splitlines() if l.strip()]
    keys = {r["id"]: r["fields"] for r in map(json.loads, (ROOT / "corpus/answer_key.jsonl").read_text().splitlines()) if r}
    assert len(notices) == 50
    led = Ledger(out / "ledger.jsonl", a.host, make_mock(keys) if a.mock else None)

    for run in range(1, a.runs + 1):
        for method, fmt in (("A", None), ("B", server_schema(schema))):
            path = out / "raw" / f"{method}_run{run}.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for n in notices:
                    r = led.call(tag=f"{n['id']}|{method}|run{run}", model=a.model,
                                 prompt=PROMPT.format(ver=ver, text=n["text"]), system=SYSTEM, fmt=fmt,
                                 temperature=a.temperature, seed=a.seed, num_predict=a.num_predict)
                    r.update(id=n["id"], method=method, run=run, raw=r.pop("content"))
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"done {method} run {run}")


if __name__ == "__main__":
    main()
