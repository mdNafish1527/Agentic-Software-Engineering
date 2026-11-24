"""Class 1 ledger wrapper (drop-in). Every model call goes through `Ledger.call`,
which records tokens and notional cost per call to results/ledger.jsonl.
If your class 1 wrapper has a different module/name, replace the body of `call`
with a call to it and keep the returned dict keys."""
import json, time, pathlib, urllib.request

# Notional prices, USD per 1M tokens (same constants as class 1 ledger).
PRICE_IN = 0.15
PRICE_OUT = 0.60


class Ledger:
    def __init__(self, path, host="http://localhost:11434", mock=None):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.host, self.mock = host, mock

    def call(self, *, tag, model, prompt, system, fmt, temperature, seed, num_predict):
        t0 = time.time()
        err = None
        if self.mock:
            content, pin, pout = self.mock(tag, prompt)
        else:
            body = {"model": model, "stream": False,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                    "options": {"temperature": temperature, "seed": seed, "num_predict": num_predict}}
            if fmt is not None:
                body["format"] = fmt
            try:
                req = urllib.request.Request(self.host + "/api/chat", json.dumps(body).encode(),
                                             {"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=300) as r:
                    d = json.loads(r.read())
                content = d["message"]["content"]
                pin, pout = d.get("prompt_eval_count", 0), d.get("eval_count", 0)
            except Exception as e:  # recorded as an invalid output
                content, pin, pout, err = "", 0, 0, repr(e)
        cost = pin * PRICE_IN / 1e6 + pout * PRICE_OUT / 1e6
        row = {"tag": tag, "model": model, "prompt_tokens": pin, "completion_tokens": pout,
               "cost_usd": cost, "latency_s": round(time.time() - t0, 3), "error": err}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        return {"content": content, **row}
