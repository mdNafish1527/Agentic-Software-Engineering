# Assignment 1: SQL calculation agent on a local model (Ollama, qwen2.5:3b)

An agent that answers calculation questions about a small shop database by calling tools. The model never
does the arithmetic: numbers come from SQL (`run_sql`) or a calculator, and a run only counts as a pass when
the expected values were actually returned by a tool. No hosted APIs, no agent framework.

## Run everything (5 commands)

```bash
ollama pull qwen2.5:3b                                            # 1. the model (once; Ollama must be running)
python -m unittest discover -s tests -t .                         # 2. 100+ unit tests, no model needed
python build_db.py && python make_questions.py                    # 3. data/shop.db (seed 42) + data/questions.json
python run_eval.py --variant baseline && python run_eval.py --variant improved   # 4. 22 questions x 3 runs each
python make_report.py                                             # 5. report.pdf + report.md from the logs
```

* Python 3.10+; only step 5's PDF needs `pip install reportlab` (without it you still get `report.md`).
* Step 4 is the slow one: 66 runs per variant, each with 3 to 6 model calls. Expect tens of minutes per variant on a
  laptop CPU and much less with a GPU. Try `python run_eval.py --limit 3 --reps 1` first.
* `python smoke_test.py` checks that your Ollama model follows JSON mode and shows whether native tool calls work
  (the assignment's "test with a tiny tool first" hint).
* A larger model for the bonus comparison: `python run_eval.py --model qwen2.5:7b --out results/qwen7b`
  (use `--variant improved`; the report generator reads `results/baseline` and `results/improved`, so compare the
  larger model by reading its `summary.json`).

## What is in the folder

| Path | What it is |
|---|---|
| `agentlab/agent.py` | the agent loop (step cap, token budget, final status) and the `generate_sql` tool |
| `agentlab/sqltools.py` | read-only SQL executor, schema helpers, safe calculator |
| `agentlab/prompts.py` | every prompt, including the baseline vs improved SQL prompt |
| `agentlab/lm.py` | Ollama wrapper with a token ledger (stand-in for the Class 1 `agentlab/lm.py`) |
| `agentlab/dbbuild.py`, `build_db.py` | creates and fills the database from a fixed seed |
| `agentlab/questions.py`, `make_questions.py` | the 22 questions and their gold SQL; writes `data/questions.json` and `data/questions_expected.md` |
| `agentlab/grader.py`, `agentlab/metrics.py` | pass/fail rules, failure classification, pass@1 and pass^3 |
| `run_eval.py` | runs every question 3 times and writes `results/<variant>/runs.jsonl` and `summary.json` |
| `make_report.py` | builds the 1-2 page report from the logs |
| `tests/` | unit tests (written first; `docs/tdd_log.txt` keeps the failing run) |
| `data/shop.db` | the database (rebuilt by step 3 anyway) |
| `docs/harness_selftest_oracle_mock/` | log-format example from a scripted stand-in, NOT model results |

## Design in brief

* **Protocol.** The model returns one JSON object per turn (Ollama `format: json`): a tool call or `{"final": "..."}`.
  I parse it myself, and malformed replies are sent back as errors. Tools: `generate_sql`, `run_sql`, `calculator`, `describe_schema`.
* **SQL tool = a separate model call** (schema + business rules + question, temperature 0). Reason: it keeps the agent prompt short for a 3B model
  and lets the SQL prompt be tuned and measured alone. Cost: one more model call per query.
* **Read-only.** One statement, SELECT/WITH only, forbidden keywords checked after literals and comments are blanked;
  file opened `mode=ro` with `query_only`; SQLite authorizer allows only SELECT/READ/FUNCTION; 50-row and 5-second limits.
* **Limits and status.** 8 agent steps and 10,000 tokens per question (`--max-steps`, `--token-budget`).
  Status is one of `ok`, `step_cap`, `token_cap`, `error`.
* **Grading.** Expected answers come from gold SQL. A run passes if status is `ok`, every expected number (tolerance 0.01 or 0.05%)
  and name is in the answer and was returned by a tool. Unanswerable questions pass with a refusal and no invented number.
  `pass@1` = mean per-question success rate; `pass^3` = share of questions that passed all 3 runs.
* **The improvement experiment.** `--variant improved` adds live data facts and worked examples to the SQL prompt. Nothing else changes,
  so the before/after in the report is a single-variable comparison. No few-shot question appears in the test set (unit-tested).
* **Repeatability.** Database seed 42; each run gets its own model seed (`1000 + 100*rep + question index`); temperature 0.2 for the agent, 0 for the SQL tool.

## Run log format

One JSON line per run in `results/<variant>/runs.jsonl`: question, tool calls, SQL generated and executed, final answer, status,
tokens (total and by purpose), time, pass/fail, grounding flag and failure type.

## Known limits

* The grader checks expected values rather than whole sentences.
* Ollama may report only uncached prompt tokens, so token counts can be slightly low on repeated prompts.
* 22 questions x 3 runs is a small sample; read the spread, not just the average.

---

## Assignment 2: English vs Bangla token cost

This part is self-contained in the `tokencost/` folder. See **TOKENCOST_README.md** for the commands, method and files. Quick start: `python -m tokencost run` after `ollama pull qwen2.5:3b`, then `python -m tokencost analyze`.
