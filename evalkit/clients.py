"""LLM client abstractions + stub providers.

Real eval frameworks treat the model as a pluggable interface. We do
the same. All providers expose `complete(prompt, ...)` and return a
string. In `dry_run=True` mode they return deterministic canned outputs
so the whole harness runs offline with zero API keys.
"""

from __future__ import annotations

import hashlib
import os
import random
import re
from dataclasses import dataclass
from typing import Protocol


class LLMClient(Protocol):
    provider: str
    model: str

    def complete(self, prompt: str, system: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 512) -> str:
        ...


# ---------------------------------------------------------------------------
# Canned-output engine used by all dry-run providers
# ---------------------------------------------------------------------------


def _canned_output(prompt: str, model: str) -> str:
    """Deterministic fake output based on prompt content.

    The goal isn't realism - it's reproducibility. Same prompt + same
    model always gives the same answer, so regression tests are stable.
    """
    p = prompt.lower()

    # judge heuristic (must come first — judge prompts often embed text
    # from the task under grading, which would trip downstream branches)
    if "rate the model" in p or "impartial grader" in p or "on a scale of 1-5" in p \
            or "choosing the better" in p or '"winner"' in p:
        import hashlib as _h
        h = int(_h.md5(prompt.encode()).hexdigest(), 16)
        if "winner" in p or "choosing the better" in p:
            winner = ["A", "B", "tie"][h % 3]
            return f'{{"winner": "{winner}", "rationale": "stub judge"}}'
        rating = 3 + (h % 3)  # 3..5
        return f'{{"rating": {rating}, "rationale": "stub judge"}}'

    # sentiment heuristic
    if "sentiment" in p or "positive" in p or "negative" in p:
        pos = sum(p.count(w) for w in ("love", "great", "amazing", "good", "best", "fantastic", "wonderful", "excellent"))
        neg = sum(p.count(w) for w in ("hate", "bad", "terrible", "awful", "worst", "horrible", "disappointing"))
        if pos > neg:
            return "positive"
        if neg > pos:
            return "negative"
        return "neutral"

    # fallback: echo + hash so outputs differ but are deterministic
    h = hashlib.md5((model + prompt).encode()).hexdigest()[:8]
    return f"[stub:{model}:{h}] {prompt[:80]}"


@dataclass
class OpenAIClient:
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    dry_run: bool = True
    api_key_env: str = "OPENAI_API_KEY"

    def complete(self, prompt: str, system: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 512) -> str:
        if self.dry_run:
            return _canned_output(prompt, self.model)
        # Real-call shape - kept here so the integration point is obvious.
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} not set; use --dry-run to skip real calls")
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise RuntimeError("openai package not installed") from e
        client = OpenAI(api_key=api_key)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""


@dataclass
class LocalClient:
    """A 'local' provider - in practice you'd wire this to llama.cpp, vLLM,
    or an Ollama HTTP endpoint. For this harness it's always canned."""
    provider: str = "local"
    model: str = "local-stub-7b"
    dry_run: bool = True

    def complete(self, prompt: str, system: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 512) -> str:
        out = _canned_output(prompt, self.model)
        if temperature > 0:
            rng = random.Random(hash(prompt) ^ int(temperature * 1000))
            if rng.random() < 0.1:
                out = out + " "  # trivial perturbation
        return out
