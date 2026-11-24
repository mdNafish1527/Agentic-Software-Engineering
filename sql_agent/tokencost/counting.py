"""Token counting, cost, template-overhead measurement and cache check."""
from __future__ import annotations

from .backend import ChatReply
from .config import PRICE_INPUT_PER_M, PRICE_OUTPUT_PER_M, SYSTEM_PROMPT


def cost_usd(input_tokens: int, output_tokens: int,
             price_in: float = PRICE_INPUT_PER_M, price_out: float = PRICE_OUTPUT_PER_M) -> float:
    """Notional cost: input tokens x $0.15 per million + output tokens x $0.60 per million."""
    return input_tokens * price_in / 1_000_000 + output_tokens * price_out / 1_000_000


def count_call(backend, user_text: str, system: str = SYSTEM_PROMPT, fresh: bool = True) -> dict:
    """One model call with its counts and cost. fresh=True empties the prompt cache first."""
    if fresh:
        backend.unload()
    reply: ChatReply = backend.chat(user_text, system=system)
    return {
        "output": reply.text,
        "input_tokens": reply.prompt_tokens,
        "output_tokens": reply.completion_tokens,
        "cost_usd": cost_usd(reply.prompt_tokens, reply.completion_tokens),
        "latency_s": reply.latency_s,
        "load_s": reply.load_s,
        "done_reason": reply.done_reason,
    }


def estimate_overhead(backend, system: str = SYSTEM_PROMPT) -> dict:
    """Fixed tokens added around every user message (system prompt + chat template).

    Probes "hi" and "hi hi hi hi" with the cache emptied before each. The extra tokens for the
    3 extra words give the cost of one word; overhead = tokens(one word probe) - tokens per word.
    """
    backend.unload()
    short = backend.chat("hi", system=system).prompt_tokens
    backend.unload()
    long = backend.chat("hi hi hi hi", system=system).prompt_tokens
    per_word = (long - short) / 3
    return {"probe_1_word_tokens": short, "probe_4_word_tokens": long,
            "tokens_per_extra_word": per_word, "overhead_tokens": round(short - per_word, 2)}


def cache_check(backend, text: str = "Explain what a database index is in one sentence.",
                system: str = SYSTEM_PROMPT) -> dict:
    """Send the same prompt twice WITHOUT unloading. If the second count is smaller, Ollama reused a cache."""
    backend.unload()
    first = backend.chat(text, system=system).prompt_tokens
    second = backend.chat(text, system=system).prompt_tokens
    return {"first_call_prompt_tokens": first, "second_call_prompt_tokens": second,
            "cache_reduced_second_call": second < first}
