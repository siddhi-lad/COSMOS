"""Stage 3 – Ingredient NLP pipeline:
text cleaning → ingredient identification → function mapping → risk assessment.
Deterministic and instant (no network)."""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field, asdict

from .knowledge import INDEX, KB, normalize

_MARKER = re.compile(r"(?:ingredients?|ingr[eé]dients?|inci|composition|zutaten|ingredientes)\s*[:\-–]?", re.I)
_STOP = re.compile(r"\b(may contain|\+/-|warning|caution|directions?|how to use|made in|manufactured|distributed|"
                   r"net ?wt|best before|expiry|batch|www\.|\.com|keep out|for external use|store in|customer care)\b", re.I)

RISK_ORDER = {"High": 0, "Medium": 1, "Low": 2, "Unknown": 3}

PROFILE_RULES = {
    "Sensitive skin": ({"irritant", "allergen"}, "May irritate or sensitise reactive skin"),
    "Pregnant or nursing": ({"pregnancy_caution"}, "Ask a clinician before use during pregnancy"),
    "Acne-prone skin": ({"comedogenic"}, "May clog pores"),
    "Known fragrance allergy": ({"fragrance"}, "Fragrance or fragrance allergen"),
}


@dataclass
class Ingredient:
    position: int
    raw: str
    name: str
    matched: bool
    match_type: str  # exact | fuzzy | unmatched
    confidence: float
    functions: list[str] = field(default_factory=list)
    risk: str = "Unknown"
    risk_note: str = "Not in the curated knowledge base – no risk assessment made."
    eu_note: str = ""
    tags: list[str] = field(default_factory=list)
    chem: str | None = None
    pm_terms: list[str] | None = None

    def flags(self, profile: list[str]) -> list[str]:
        out = []
        for p in profile:
            tags, msg = PROFILE_RULES.get(p, (set(), ""))
            if tags & set(self.tags):
                out.append(msg)
        return list(dict.fromkeys(out))

    def to_dict(self) -> dict:
        return asdict(self)


# ── cleaning ─────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.replace("\r", "\n").replace("•", ",").replace("·", ",").replace("；", ";").replace("、", ",")
    text = re.sub(r"[|*_#`]+", " ", text)
    text = re.sub(r"-\s*\n\s*", "", text)  # hyphenated line-breaks
    text = re.sub(r"\s*\n\s*", " ", text)  # OCR line wraps inside the list
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def extract_segment(text: str) -> str:
    """Keep only the text after 'Ingredients:' and before warnings/other sections."""
    lines = [l.strip() for l in text.replace("\r", "\n").split("\n") if l.strip()]
    if len(lines) >= 3 and "," not in text and ";" not in text:
        text = ", ".join(lines)  # one ingredient per line, no commas anywhere -> the line breaks ARE the separators
    t = clean_text(text)
    m = None
    for m in _MARKER.finditer(t):  # use the last marker (some labels repeat it)
        pass
    if m:
        t = t[m.end():]
    s = _STOP.search(t)
    if s:
        t = t[:s.start()]
    return t.strip(" .,:;-")


def split_ingredients(segment: str) -> list[str]:
    """Split on commas / semicolons at parenthesis depth 0.
    A comma between two digits belongs to the name (e.g. '1,2-Hexanediol') and is not a separator."""
    parts, buf, depth = [], [], 0
    for i, ch in enumerate(segment):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        in_number = (ch == "," and 0 < i < len(segment) - 1
                     and segment[i - 1].isdigit() and segment[i + 1].isdigit())
        if ch in ",;" and depth == 0 and not in_number:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    cleaned = []
    for p in parts:
        p = re.sub(r"^\s*(and|&)\s+", "", p.strip(" .:-\t"), flags=re.I).strip()
        if len(re.sub(r"[^A-Za-z]", "", p)) >= 2:
            cleaned.append(p)
    return cleaned


# ── identification ───────────────────────────────────────────────────────────
def _candidates(token: str) -> list[str]:
    cands = [normalize(token)]
    no_paren = normalize(re.sub(r"\([^)]*\)", " ", token))
    cands.append(no_paren)
    cands += [normalize(x) for x in re.findall(r"\(([^)]*)\)", token)]
    if "/" in token:
        cands += [normalize(x) for x in token.split("/")]
    seen, out = set(), []
    for c in cands:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


_PREFIXES = ("di", "tri", "tetra", "poly", "bis", "iso", "peg", "ppg")


def _is_homologue(a: str, b: str) -> bool:
    """True when two names differ only by a chemical prefix on one word
    (e.g. 'dipropylene glycol' vs 'propylene glycol') - different substances, so never fuzzy-match.
    Ordinary OCR typos (extra/missing letters elsewhere) are still allowed."""
    wa, wb = a.split(), b.split()
    if len(wa) != len(wb):
        return False
    for x, y in zip(wa, wb):
        if x == y:
            continue
        short, long_ = sorted((x, y), key=len)
        if long_.endswith(short) and long_[: len(long_) - len(short)] in _PREFIXES:
            return True
    return False


def identify(token: str, position: int = 0) -> Ingredient:
    cands = _candidates(token)
    for c in cands:
        if c in INDEX:
            return _from_kb(position, token, INDEX[c], "exact", 1.0)
    keys = list(INDEX)
    best, best_score = None, 0.0
    for c in cands:
        if len(c) < 5:
            continue
        for k in difflib.get_close_matches(c, keys, n=1, cutoff=0.86):
            if _is_homologue(c, k):
                continue
            score = difflib.SequenceMatcher(None, c, k).ratio()
            if score > best_score:
                best, best_score = k, score
    if best:
        return _from_kb(position, token, INDEX[best], "fuzzy", round(best_score, 2))
    display = re.sub(r"\s+", " ", re.sub(r"\([^)]*\)", "", token)).strip().title() or token.strip()
    return Ingredient(position, token.strip(), display, False, "unmatched", 0.0)


def _from_kb(position: int, raw: str, canonical: str, how: str, conf: float) -> Ingredient:
    e = KB[canonical]
    return Ingredient(position, raw.strip(), e.name, True, how, conf, list(e.functions), e.risk, e.note,
                      e.cosing_note, list(e.tags), e.chem, list(e.pm_terms) if e.pm_terms else None)


def analyze_text(text: str) -> list[Ingredient]:
    """Full NLP pipeline on raw OCR / manual text. Duplicates are merged."""
    out: list[Ingredient] = []
    seen: set[str] = set()
    for tok in split_ingredients(extract_segment(text)):
        ing = identify(tok, len(out) + 1)
        key = normalize(ing.name)
        if key in seen:
            continue
        seen.add(key)
        out.append(ing)
    for i, ing in enumerate(out, 1):
        ing.position = i
    return out


def summarize(ings: list[Ingredient]) -> dict:
    return {
        "total": len(ings),
        "recognised": sum(i.matched for i in ings),
        "high": sum(i.risk == "High" for i in ings),
        "medium": sum(i.risk == "Medium" for i in ings),
        "low": sum(i.risk == "Low" for i in ings),
        "unknown": sum(i.risk == "Unknown" for i in ings),
    }
