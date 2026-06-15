"""LLM-powered entity resolution using Claude.

Normalises supplier name variants into a canonical form:
  "Apple Inc" / "Apple Computer" / "APPLE INC." → "Apple Inc."

This is the documented silent failure point of GraphRAG systems —
handling it explicitly is a strong portfolio signal.
"""

from __future__ import annotations

import json
from typing import Optional

import anthropic
from loguru import logger

from ..config import settings

_client: Optional[anthropic.Anthropic] = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


SYSTEM_PROMPT = """You are an entity resolution system specialising in company and vendor names.

Given vendor name variants, identify which refer to the same entity and return the most
formal, complete canonical name for each.

Return ONLY a valid JSON object mapping each input variant to its canonical name.
Example input: ["Apple Inc", "Apple Computer", "APPLE", "Google LLC", "Google"]
Example output: {"Apple Inc": "Apple Inc.", "Apple Computer": "Apple Inc.", "APPLE": "Apple Inc.", "Google LLC": "Google LLC", "Google": "Google LLC"}

Rules:
- Same company → same canonical name  
- Different companies → different canonical names
- Use the most formal/complete version (prefer "Inc." over "Inc", "LLC" over abbreviations)
- For receipts/stores (e.g. "WALMART #1234"), strip location codes → "Walmart"
- Return ONLY the JSON object, no explanation, no markdown"""


def resolve_batch(names: list[str]) -> dict[str, str]:
    """Resolve a list of vendor names to canonical forms in one Claude call.

    Args:
        names: Raw vendor name strings from invoice extractions

    Returns:
        Dict mapping each input name to its canonical form.
        Falls back to identity mapping if Claude call fails.
    """
    unique = list({n.strip() for n in names if n and len(n.strip()) >= 2})
    if not unique:
        return {}

    logger.info(f"Resolving {len(unique)} entity names via Claude...")

    names_json = json.dumps(unique)
    try:
        response = _get_client().messages.create(
            model=settings.claude_model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Normalise these vendor names:\n{names_json}",
            }],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:-1]).strip()

        result: dict[str, str] = json.loads(raw)

        # Log any resolutions that changed the name
        changed = {k: v for k, v in result.items() if k != v}
        if changed:
            logger.info(f"Entity resolutions: {changed}")

        return result

    except Exception as exc:
        logger.warning(f"Batch entity resolution failed: {exc} — using original names")
        return {n: n for n in unique}


def resolve_single(name: str) -> str:
    """Resolve a single vendor name. Useful for one-off lookups."""
    if not name or len(name.strip()) < 2:
        return name
    result = resolve_batch([name])
    return result.get(name, name)
