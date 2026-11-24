"""Model backends. OllamaBackend calls the local Ollama server; FakeBackend is for tests only."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from .config import HOST, MODEL, OPTIONS, SYSTEM_PROMPT


class BackendError(RuntimeError):
    """The model server failed or returned a reply without token counts."""


@dataclass
class ChatReply:
    text: str
    prompt_tokens: int       # Ollama prompt_eval_count (input tokens, incl. template and system prompt)
    completion_tokens: int   # Ollama eval_count (output tokens)
    latency_s: float         # total time minus model loading time
    load_s: float            # time spent loading the model into memory
    done_reason: str | None  # "stop", or "length" when num_predict was reached


class OllamaBackend:
    def __init__(self, model: str = MODEL, host: str = HOST, options: dict | None = None, timeout: int = 600):
        self.model = model
        self.host = host.rstrip("/")
        self.options = dict(OPTIONS if options is None else options)
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(self.host + path, data=json.dumps(payload).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            raise BackendError(f"Ollama request to {path} failed: {exc}") from exc

    def chat(self, user_text: str, system: str = SYSTEM_PROMPT) -> ChatReply:
        payload = {
            "model": self.model,
            "stream": False,
            "options": self.options,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_text}],
        }
        body = self._post("/api/chat", payload)
        try:
            text = body["message"]["content"]
            prompt_tokens = int(body["prompt_eval_count"])
            completion_tokens = int(body["eval_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BackendError(f"Ollama reply has no usable token count: {exc}") from exc
        load_s = body.get("load_duration", 0) / 1e9
        total_s = body.get("total_duration", 0) / 1e9
        return ChatReply(text=text, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                         latency_s=max(total_s - load_s, 0.0), load_s=load_s,
                         done_reason=body.get("done_reason"))

    def unload(self) -> None:
        """Unload the model. The next call starts with an empty prompt cache."""
        self._post("/api/generate", {"model": self.model, "keep_alive": 0})


class FakeBackend:
    """Deterministic stand-in for tests: 1 token per word plus a fixed template overhead."""

    def __init__(self, template_overhead: int = 20, answer_words: int = 5):
        self.template_overhead = template_overhead
        self.answer_words = answer_words
        self.calls: list[tuple[str, str]] = []
        self.unloads = 0

    def chat(self, user_text: str, system: str = SYSTEM_PROMPT) -> ChatReply:
        self.calls.append((system, user_text))
        return ChatReply(text=" ".join(["w"] * self.answer_words),
                         prompt_tokens=self.template_overhead + len(user_text.split()),
                         completion_tokens=self.answer_words, latency_s=0.5, load_s=0.0, done_reason="stop")

    def unload(self) -> None:
        self.unloads += 1
