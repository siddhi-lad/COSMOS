"""Stage 4 – Intent classifier: rule-based first (instant, free), with an optional
LLM call only when the rules are unsure."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import llm
from .config import Settings
from .knowledge import INDEX, normalize

INTENTS = [
    "ingredient_function", "ingredient_safety", "ingredient_identification",
    "regulatory_status", "product_suitability", "evidence_lookup", "general_cosmetic",
]

_PATTERNS: dict[str, list[str]] = {
    "ingredient_function": [r"\bfunction", r"\brole\b", r"\bused for\b", r"\bpurpose\b", r"\bwhat does .* do\b",
                            r"\bwhy is .* (in|added)\b", r"\bbenefit", r"\bdoes .* work\b"],
    "ingredient_safety": [r"\bsafe", r"\bsafety\b", r"\brisk", r"\birritat", r"\ballerg", r"\btoxic", r"\bharm",
                          r"\bside effect", r"\bpregnan", r"\bdanger", r"\bcause", r"\bsting", r"\bburn"],
    "ingredient_identification": [r"\bidentify\b", r"\bwhat is\b", r"\bwhat'?s\b", r"\bwhich ingredient", r"\blist\b",
                                  r"\binci\b", r"\balso known as\b", r"\bsynonym", r"\bwhat are the ingredients"],
    "regulatory_status": [r"\bbann", r"\blegal", r"\bregulat", r"\brestrict", r"\bapprov", r"\b(eu|fda|cosing)\b",
                          r"\ballowed\b", r"\blimit", r"\bmax(imum)? concentration", r"\bprohibit"],
    "product_suitability": [r"\bsuitable\b", r"\bgood for\b", r"\brecommend", r"\bskin type\b",
                            r"\bthis product\b", r"\bok for\b", r"\bokay for\b", r"\bfor (my )?(oily|dry|sensitive|acne)"],
    "evidence_lookup": [r"\bstud(y|ies)\b", r"\bevidence\b", r"\bresearch\b", r"\bpubmed\b", r"\bpaper", r"\bclinical\b",
                        r"\bcitation", r"\bsources?\b", r"\bproof\b", r"\bpublished\b"],
}
_COMPILED = {k: [re.compile(p, re.I) for p in v] for k, v in _PATTERNS.items()}


@dataclass
class IntentResult:
    intent: str
    confidence: float
    method: str  # rules | llm | default
    ingredients: list[str] = field(default_factory=list)


def find_mentions(query: str, scan_names: list[str] | None = None) -> list[str]:
    """Ingredient names mentioned in the query (KB aliases + names in the current scan)."""
    qn = f" {normalize(query)} "
    found: list[str] = []
    cands = {normalize(n): n for n in (scan_names or [])}
    for alias, canon in INDEX.items():
        if len(alias) >= 4 or alias in ("mit", "mci", "sls"):
            cands.setdefault(alias, canon)
    for alias, canon in sorted(cands.items(), key=lambda kv: -len(kv[0])):
        if f" {alias} " in qn and canon not in found:
            found.append(canon)
    return found


def classify_rules(query: str) -> tuple[str, float]:
    scores = {k: sum(bool(p.search(query)) for p in pats) for k, pats in _COMPILED.items()}
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top, second = ranked[0], ranked[1]
    if top[1] == 0:
        return "general_cosmetic", 0.3
    if top[1] == second[1]:
        return top[0], 0.45
    return top[0], min(0.95, 0.6 + 0.15 * (top[1] - second[1]))


def classify(query: str, settings: Settings | None = None, scan_names: list[str] | None = None) -> IntentResult:
    mentions = find_mentions(query, scan_names)
    intent, conf = classify_rules(query)
    if conf >= 0.55 or settings is None or not settings.llm_intent_fallback or not settings.has_llm:
        return IntentResult(intent, conf, "rules" if conf >= 0.55 else "default", mentions)
    try:
        prompt = ("Classify the user's cosmetic-science question into exactly one label from: "
                  + ", ".join(INTENTS) + ".\nQuestion: " + query + "\nAnswer with the label only.")
        text, _ = llm.chat(settings, [{"role": "user", "content": prompt + " Do not explain."}],
                           temperature=0.0, max_tokens=300)
        hits = [(text.lower().rfind(lab), lab) for lab in INTENTS if lab in text.lower()]
        label = max(hits)[1] if hits else ""  # last label mentioned = the model's conclusion
        if label in INTENTS:
            return IntentResult(label, 0.7, "llm", mentions)
    except llm.LLMError:
        pass
    return IntentResult(intent, conf, "default", mentions)
