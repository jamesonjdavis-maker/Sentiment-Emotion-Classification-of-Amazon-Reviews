"""Shared OpenAI-compatible client.

Reads LLM_BASE_URL / LLM_API_KEY / LLM_MODEL from the environment or from a
local .env file (kept out of git). Falls back to a local Ollama server.

The class model (Qwen3.x) is a *thinking* model: by default it spends ~170
tokens in a hidden reasoning block before answering, so a short max_tokens
returns an empty answer. We switch thinking off via the chat template and
also strip any <think>...</think> block defensively.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from openai import OpenAI

_ENV = Path(__file__).parent / ".env"
if _ENV.exists():
    for line in _ENV.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

client = OpenAI(base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
                api_key=os.environ.get("LLM_API_KEY", "ollama"))  # Ollama ignores the key
MODEL = os.environ.get("LLM_MODEL", "llama3.2:3b")
SETTINGS = {"temperature": 0, "enable_thinking": False}

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def chat(system: str, user: str, max_tokens: int = 20, response_format: dict | None = None) -> str:
    extra = {"response_format": response_format} if response_format else {}
    resp = client.chat.completions.create(
        **extra,
        model=MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=SETTINGS["temperature"],
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": SETTINGS["enable_thinking"]}},
    )
    return _THINK_RE.sub("", resp.choices[0].message.content or "").strip()
