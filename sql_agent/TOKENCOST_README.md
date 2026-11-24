# Assignment 2: English vs Bangla token cost (Ollama, qwen2.5:3b)

Measures how many more input and output tokens, and how much more notional cost, a Bangla prompt
needs compared with an English prompt that has the same meaning. Uses 50 aligned pairs, a local model,
and the course prices ($0.15 per million input tokens, $0.60 per million output tokens).

## Run (from the `sql_agent` folder)

```bash
pip install matplotlib                               # optional, for chart.png
ollama pull qwen2.5:3b                               # once; Ollama must be running
python -m unittest discover -s tests -t .            # tests (no model needed)
python -m tokencost probe                            # template overhead + cache check
python -m tokencost run                              # 50 pairs x 2 languages = 100 calls
python -m tokencost analyze                          # comparison.csv, summary.json, chart.png
```

Quick trial first: `python -m tokencost run --limit 3`, then delete `tokencost/results/results.csv`
before the full run (or use `--restart`).

## Files

| Path | Purpose |
|---|---|
| `tokencost/data/dataset.tsv` | 50 aligned pairs: `id, category, english, bangla` (tab separated) |
| `tokencost/data/quality_check.csv` | 10 chosen pairs for the manual quality check (fill after reading the outputs) |
| `tokencost/config.py` | every setting shared by both languages (model, options, prompt, prices) |
| `tokencost/backend.py` | `OllamaBackend` (real calls) and `FakeBackend` (tests) |
| `tokencost/counting.py` | token counts, cost, template overhead, cache check |
| `tokencost/runner.py` | runs all calls, writes `results.csv` |
| `tokencost/analysis.py` | per-pair ratios, totals, bootstrap interval, categories, chart |
| `tokencost/__main__.py` | the command line (`probe`, `run`, `analyze`) |
| `tests/test_tokencost_*.py` | tests for counting, cost, runner, dataset and analysis |
| `tokencost/results/` | outputs: `probe.json`, `run_config.json`, `results.csv`, `comparison.csv`, `summary.json`, `chart.png` |

## Method (state these in the report)

- **Token counts:** taken from Ollama's `prompt_eval_count` (input) and `eval_count` (output). A missing count raises an error; it is never replaced by 0.
- **Template and system prompt overhead:** one fixed system prompt is used for every call. `probe` measures the fixed overhead by sending "hi" and "hi hi hi hi" and subtracting the per-word cost. The result is stored in `probe.json` and `run_config.json`.
- **Cache:** Ollama can reuse a cached prompt prefix, which lowers `prompt_eval_count` on repeat calls. `probe` checks this (`cache_reduced_second_call`). `run` unloads the model before every call (`keep_alive: 0`), so every call starts with an empty cache. Latency excludes the model load time, which is recorded separately as `load_s`.
- **Same settings for both languages:** temperature 0, seed 42, `num_predict` 256 (max output), `num_ctx` 4096. The system prompt does not mention language.
- **Order:** the language order alternates from pair to pair to reduce drift bias.
- **Truncation:** a call that reaches `num_predict` is flagged `truncated=True`. Check the count in `summary.json`; truncated outputs make the output ratio less reliable.
- **Cost:** `input × 0.15 / 1e6 + output × 0.60 / 1e6`, computed per call and totalled per language.
- **Interval:** the 95% interval is a percentile bootstrap (10,000 resamples, fixed seed) for the **mean** per-pair ratio.
- **Output language:** the model may answer a Bangla prompt in English. That is allowed, but it changes the output token count, so note it in the report.

## Caveats

- The Bangla text was written for this assignment. Have a native speaker check the 50 pairs before submission.
- Each call reloads the model, so `run` takes much longer than a normal chat session. `--no-fresh` is faster but lets the cache affect the counts.
- Token counts depend on the tokenizer of qwen2.5:3b. The notional prices are for comparison only, not real provider prices.
