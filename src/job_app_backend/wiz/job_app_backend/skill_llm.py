"""
LLM-based skill extractor using a local Ollama model (qwen2.5:7b).

Returns the same entity format as skill_model.predict() so the two
backends are interchangeable in the UI.
"""

from __future__ import annotations

import json
import re

import requests

from .skill_model import TAXONOMY

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"

_VALID_LABELS = frozenset(TAXONOMY)

_SYSTEM_PROMPT = """\
You are a skill extraction engine. Given a job posting, identify every skill, \
technology, tool, domain-knowledge area, and soft skill mentioned.

For each skill found, return a JSON object with:
- "text": the exact substring as it appears in the original text
- "label": one of TECH, TOOL, DOMAIN, SOFT
  - TECH: programming languages, methodologies, practices (e.g. Python, DevOps, machine learning)
  - TOOL: specific platforms, products, services (e.g. Docker, AWS, Airflow)
  - DOMAIN: domain knowledge areas (e.g. risk management, NLP, forecasting)
  - SOFT: interpersonal / soft skills (e.g. collaboration, leadership)

Return ONLY a JSON array. No explanation, no markdown fences. Example:
[{"text": "Python", "label": "TECH"}, {"text": "Docker", "label": "TOOL"}]"""


def _ollama_generate(prompt: str, model: str = DEFAULT_MODEL) -> str:
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "system": _SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_ctx": 16384},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def _parse_response(raw: str) -> list[dict]:
    """Extract a JSON array from the LLM response, tolerating markdown fences."""
    idx = raw.find("[")
    if idx == -1:
        return []
    try:
        result, _ = json.JSONDecoder().raw_decode(raw, idx)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        return []


def _locate_spans(text: str, items: list[dict]) -> list[dict]:
    """Find character offsets for each extracted skill in the original text."""
    entities: list[dict] = []
    text_lower = text.lower()
    occupied: list[tuple[int, int]] = []
    for item in items:
        skill = item.get("text", "")
        label = item.get("label", "TECH")
        if label not in _VALID_LABELS:
            label = "TECH"
        for m in re.finditer(re.escape(skill.lower()), text_lower):
            start, end = m.start(), m.end()
            if any(s < end and start < e for s, e in occupied):
                continue
            occupied.append((start, end))
            entities.append({
                "start": start,
                "end": end,
                "label": label,
                "text": text[start:end],
            })
    return entities


def is_available(model: str = DEFAULT_MODEL) -> bool:
    """Check whether Ollama is running and the model is pulled."""
    try:
        data = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3).json()
        names = {m["name"] for m in data.get("models", [])}
        return model in names or any(n.startswith(model) for n in names)
    except (requests.RequestException, KeyError, ValueError):
        return False


def predict(text: str, model: str = DEFAULT_MODEL) -> list[dict]:
    """Extract skills from text using the local LLM.

    Returns the same format as skill_model.predict():
    list of {start, end, label, text} dicts.
    """
    raw = _ollama_generate(text, model=model)
    items = _parse_response(raw)
    return _locate_spans(text, items)
