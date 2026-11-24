"""Settings shared by both languages. Change them only here so the comparison stays fair."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET = HERE / "data" / "dataset.tsv"
QUALITY_FILE = HERE / "data" / "quality_check.csv"
RESULTS_DIR = HERE / "results"

MODEL = "qwen2.5:3b"
HOST = "http://localhost:11434"

# Notional course prices in USD per 1,000,000 tokens.
PRICE_INPUT_PER_M = 0.15
PRICE_OUTPUT_PER_M = 0.60

# One fixed system prompt for every call. It is in English and does NOT mention language,
# so the model is not told to answer in one language and not the other.
SYSTEM_PROMPT = "You are a helpful assistant. Answer the user's request concisely."

# Identical for both languages. num_predict is Ollama's name for max output tokens.
OPTIONS = {"temperature": 0, "seed": 42, "num_predict": 256, "num_ctx": 4096}

LANGUAGES = ("english", "bangla")
CATEGORIES = ("question", "instruction", "explanation", "arithmetic", "summary", "long")
