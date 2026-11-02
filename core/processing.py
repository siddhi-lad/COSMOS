"""Stage 6 – 'Ling Sante' evidence processing and validation:
normalise → validate → de-duplicate → rank. Pure Python, no network."""
from __future__ import annotations

import re
from datetime import date

from .knowledge import normalize

SOURCE_WEIGHT = {"FDA": 1.0, "CosIng": 1.0, "PubChem": 0.6, "PubMed": 0.7}
SKIN_WORDS = ("skin", "topical", "dermat", "cosmetic", "barrier", "irritation", "sensitization", "sensitisation")


def _validate(item: dict) -> list[str]:
    problems = []
    if not item.get("title"):
        problems.append("missing title")
    if not str(item.get("url", "")).startswith("https://"):
        problems.append("no secure source link")
    if item["source"] == "PubMed" and not item.get("snippet"):
        problems.append("no abstract available")
    return problems


def process(ingredient: str, items: list[dict], aliases: list[str] | None = None) -> list[dict]:
    """Return cleaned, validated, ranked evidence for one ingredient."""
    names = [normalize(ingredient)] + [normalize(a) for a in (aliases or [])]
    seen, out = set(), []
    this_year = date.today().year
    for raw in items:
        it = dict(raw)
        it["title"] = re.sub(r"\s+", " ", (it.get("title") or "")).strip()
        it["snippet"] = re.sub(r"\s+", " ", (it.get("snippet") or "")).strip()
        key = it.get("url") or it["title"].lower()
        if key in seen:
            continue
        seen.add(key)
        problems = _validate(it)
        hay = normalize(it["title"] + " " + it["snippet"])
        relevance = any(n and n in hay for n in names) or it["source"] in ("FDA", "CosIng", "PubChem")
        title_norm = normalize(it["title"])
        it["title_hit"] = any(n and n in title_norm for n in names)  # ingredient named in the title
        score = SOURCE_WEIGHT.get(it["source"], 0.5)
        if it.get("kind") == "review":
            score += 0.2
        if it.get("year"):
            score += 0.1 if it["year"] >= this_year - 6 else 0.0
        score += 0.05 * sum(w in hay for w in SKIN_WORDS[:4])
        if it["source"] == "PubMed":
            score += 0.25 if it["title_hit"] else -0.15
        if not relevance:
            score -= 0.3
            problems.append("ingredient not named in title/abstract")
        it["validated"] = not [p for p in problems if p != "no abstract available"]
        it["flags"] = problems
        it["score"] = round(score, 2)
        out.append(it)
    out.sort(key=lambda x: (-x["validated"], -x["score"]))
    return out


def count_sources(items: list[dict]) -> dict[str, int]:
    c: dict[str, int] = {}
    for it in items:
        c[it["source"]] = c.get(it["source"], 0) + 1
    return c
