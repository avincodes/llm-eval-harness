"""LLM client abstractions + stub providers.

Real eval frameworks treat the model as a pluggable interface. We do the
same. All three providers below expose `complete(prompt, ...)` and return
a string. In `dry_run=True` mode they return deterministic canned outputs
so the whole harness runs offline with zero API keys - which is what the
tests and CI rely on.

Swapping in a real provider is a ~10-line change: drop `dry_run`, import
the real SDK, call it inside `complete`.
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

    The goal isn't realism - it's reproducibility. Same prompt + same model
    always gives the same answer, so regression tests are stable.
    """
    p = prompt.lower()

    # judge heuristic (must come first — judge prompts often embed SQL or text
    # from the task under grading, which would otherwise trip the SQL branch)
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

    # SQL heuristic - extract table/column names loosely
    if "sql" in p or "select" in p or "query" in p:
        # find a plausible table name
        m = re.search(r"from\s+(?:the\s+)?(\w+)\s+table", p)
        table = m.group(1) if m else "users"
        if "count" in p:
            return f"SELECT COUNT(*) FROM {table};"
        if "average" in p or "avg" in p:
            col_m = re.search(r"average\s+(\w+)", p)
            col = col_m.group(1) if col_m else "value"
            return f"SELECT AVG({col}) FROM {table};"
        if "all" in p:
            return f"SELECT * FROM {table};"
        return f"SELECT * FROM {table} LIMIT 10;"

    # fallback: echo + hash so outputs differ but are deterministic
    h = hashlib.md5((model + prompt).encode()).hexdigest()[:8]
    return f"[stub:{model}:{h}] {prompt[:80]}"


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


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
class AnthropicClient:
    provider: str = "anthropic"
    model: str = "claude-3-5-sonnet-latest"
    dry_run: bool = True
    api_key_env: str = "ANTHROPIC_API_KEY"

    def complete(self, prompt: str, system: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 512) -> str:
        if self.dry_run:
            return _canned_output(prompt, self.model)
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} not set; use --dry-run to skip real calls")
        try:
            import anthropic  # type: ignore
        except ImportError as e:
            raise RuntimeError("anthropic package not installed") from e
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        # first text block
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                return block.text
        return ""


@dataclass
class LocalClient:
    """A 'local' provider - in practice you'd wire this to llama.cpp, vLLM,
    or an Ollama HTTP endpoint. For this harness it's always canned."""
    provider: str = "local"
    model: str = "local-stub-7b"
    dry_run: bool = True

    def complete(self, prompt: str, system: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 512) -> str:
        # add a tiny bit of noise keyed on temperature so two variants differ
        out = _canned_output(prompt, self.model)
        if temperature > 0:
            rng = random.Random(hash(prompt) ^ int(temperature * 1000))
            if rng.random() < 0.1:
                out = out + " "  # trivial perturbation
        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_client(provider: str, model: str, dry_run: bool = True) -> LLMClient:
    provider = provider.lower()
    if provider == "openai":
        return OpenAIClient(model=model, dry_run=dry_run)
    if provider == "anthropic":
        return AnthropicClient(model=model, dry_run=dry_run)
    if provider == "local":
        return LocalClient(model=model, dry_run=dry_run)
    raise ValueError(f"Unknown provider: {provider}")
