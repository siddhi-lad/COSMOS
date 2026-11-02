"""Stage 6 – Ling Sante: LLM-assisted evidence validation.

After the rule-based normalise / de-duplicate / rank step (processing.py), the Ling Sante
model (LING_SANTE_MODEL, default inclusionai/ling-3.0-flash-sante:free on OpenRouter) reads
each retrieved study title + abstract excerpt and says whether it is actually relevant to the
ingredient's use on skin. Irrelevant papers are flagged and pushed down so they are not cited.

Design for low latency / no new failure modes: ONE batched call per retrieval, short timeout,
and any failure silently keeps the rule-based ranking."""
from __future__ import annotations

import json
import re

from . import llm
from .config import OPENROUTER_BASE, Settings

MAX_ITEMS = 14
VERDICTS = ("relevant", "weak", "irrelevant")
PENALTY = {"relevant": 0.15, "weak": -0.1, "irrelevant": -0.6}

SYSTEM = ("You validate scientific evidence for a cosmetic-ingredient information tool. For each numbered record decide "
          "whether it is relevant to the use or safety of the named ingredient in skin-care / cosmetic products. "
          "Verdicts: relevant, weak (only loosely related), irrelevant (different topic, e.g. oral drug trial or unrelated "
          "condition). Reply with JSON only: [{\"id\":1,\"verdict\":\"relevant\"}, ...]")


def _parse(text: str) -> dict[int, str]:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return {}
    try:
        rows = json.loads(m.group(0))
    except ValueError:
        return {}
    out = {}
    for r in rows if isinstance(rows, list) else []:
        try:
            v = str(r.get("verdict", "")).lower().strip()
            if v in VERDICTS:
                out[int(r["id"])] = v
        except (AttributeError, KeyError, ValueError, TypeError):
            continue
    return out


def validate(evidence: dict[str, list[dict]], settings: Settings) -> bool:
    """Mutates items in place (adds ai_verdict, adjusts score/flags, re-sorts). Returns True if the model ran."""
    if not (settings.ling_enabled and settings.has_openrouter):
        return False
    todo: list[tuple[str, dict]] = []
    for name, items in evidence.items():
        for it in items:
            if it["source"] == "PubMed" and it.get("snippet") and "ai_verdict" not in it:
                todo.append((name, it))
    todo = todo[:MAX_ITEMS]
    if not todo:
        return False
    lines = [f"[{n}] Ingredient: {name}\nTitle: {it['title']}\nAbstract: {it['snippet'][:350]}"
             for n, (name, it) in enumerate(todo, 1)]
    try:
        text = llm.complete(OPENROUTER_BASE, settings.openrouter_key, settings.ling_model,
                            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(lines)}],
                            temperature=0.0, max_tokens=40 * len(todo) + 40, timeout=min(settings.llm_timeout, 25))
    except llm.LLMError:
        return False
    verdicts = _parse(text)
    if not verdicts:
        return False
    for n, (_, it) in enumerate(todo, 1):
        v = verdicts.get(n)
        if not v:
            continue
        it["ai_verdict"] = v
        it["score"] = round(it["score"] + PENALTY[v], 2)
        if v == "irrelevant":
            it["flags"] = list(it["flags"]) + ["Ling Sante: not relevant to cosmetic use"]
            it["validated"] = False
    for items in evidence.values():
        items.sort(key=lambda x: (-x["validated"], -x["score"]))
    return True
