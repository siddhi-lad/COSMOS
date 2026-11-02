"""Stages 5 + 7 – LLM layer and response synthesis.

The LLM only ever sees (a) the ingredient analysis and (b) numbered evidence items,
and must cite them as [n]. Citations are validated in code, the source list is built
in code (so links can never be invented), and a deterministic answer is used if every
LLM is unavailable."""
from __future__ import annotations

import re
from typing import Iterator

from . import llm
from .config import Settings
from .nlp import Ingredient, RISK_ORDER

SYSTEM = (
    "You are COSMOS, a careful cosmetic-ingredient information assistant for consumers. "
    "Use ONLY the INGREDIENT DATA and EVIDENCE provided. Cite evidence as [n] right after the claim it supports; "
    "never invent a citation number, study, or statistic. If the evidence does not cover something, say so plainly. "
    "You are not a doctor: do not diagnose; suggest a patch test or a dermatologist for persistent reactions. "
    "Write in clear, friendly, plain English. Be concise. "
    "Begin your reply with the line [[ANSWER]] and then give only the final answer (no analysis, no drafts)."
)
MARKER = "[[ANSWER]]"
SUMMARY_FORMAT = (
    "Format in Markdown with exactly these sections:\n"
    "### Key takeaways\n(3-4 short bullets)\n"
    "### Ingredient-wise analysis\n(one bullet per ingredient that has medium/high risk or relevant evidence: "
    "**Name** – function, risk, and what the evidence says, with [n] citations)\n"
    "### Limitations & uncertainty\n(2-3 bullets)\n"
    "### In plain words\n(2 sentences)"
)


def build_context(ings: list[Ingredient], evidence: dict[str, list[dict]], profile: list[str],
                  per_ing: int = 3, max_items: int = 12) -> tuple[str, list[dict]]:
    """Returns (prompt text, numbered evidence list). Index in list + 1 == citation number."""
    lines = ["INGREDIENT DATA:"]
    for i in ings[:40]:
        flags = i.flags(profile)
        lines.append(f"- {i.name} | functions: {', '.join(i.functions) or 'unknown'} | risk: {i.risk} | "
                     f"{i.risk_note}" + (f" | EU: {i.eu_note}" if i.eu_note and i.risk in ("Medium", "High") else "")
                     + (f" | user-profile flags: {'; '.join(flags)}" if flags else ""))
    if profile:
        lines.append(f"USER PROFILE: {', '.join(profile)}")
    numbered: list[dict] = []
    lines.append("\nEVIDENCE:")
    for name, items in evidence.items():
        useful = [it for it in items if it["source"] != "FDA" and not it["snippet"].startswith("No specific entry")
                  and it.get("validated", True) and (it["source"] != "PubMed" or it.get("title_hit"))]
        for it in useful[:per_ing]:
            if len(numbered) >= max_items:
                break
            numbered.append({**it, "ingredient": name})
            n = len(numbered)
            yr = f", {it['year']}" if it.get("year") else ""
            lines.append(f"[{n}] ({it['source']}{yr}) about {name}: {it['title']} — {it['snippet'][:420]}")
    if not numbered:
        lines.append("(none retrieved)")
    return "\n".join(lines), numbered


def validate_citations(text: str, n_items: int) -> tuple[str, list[int]]:
    """Remove citation numbers that don't exist; return the cleaned text and used numbers."""
    used: list[int] = []

    def fix(m: re.Match) -> str:
        nums = [int(x) for x in re.findall(r"\d+", m.group(0))]
        good = [x for x in nums if 1 <= x <= n_items]
        for g in good:
            if g not in used:
                used.append(g)
        return "[" + ", ".join(map(str, good)) + "]" if good else ""

    cleaned = re.sub(r"\[(?:\d+\s*,?\s*)+\]", fix, text)
    return cleaned, sorted(used)


def sources_markdown(numbered: list[dict], used: list[int]) -> str:
    if not used:
        return ""
    rows = []
    for n in used:
        it = numbered[n - 1]
        yr = f" ({it['year']})" if it.get("year") else ""
        tag = " · curated pointer" if it.get("curated") else ""
        rows.append(f"{n}. [{it['title']}]({it['url']}) — {it['source']}{yr}, *{it['ingredient']}*{tag}")
    return "\n\n**Sources**\n" + "\n".join(rows)


def fallback_summary(ings: list[Ingredient], profile: list[str], reason: str = "") -> str:
    """Deterministic summary used when no LLM is reachable. Based only on the curated KB."""
    flagged = sorted([i for i in ings if i.risk in ("High", "Medium")], key=lambda i: RISK_ORDER[i.risk])
    unknown = [i.name for i in ings if not i.matched]
    out = ["### Key takeaways"]
    out.append(f"- {len(ings)} ingredients found; {sum(i.matched for i in ings)} recognised in the curated knowledge base.")
    out.append(f"- {len(flagged)} ingredient(s) deserve attention." if flagged else "- No medium- or high-concern ingredients were identified among the recognised ones.")
    prof = [(i.name, f) for i in ings for f in i.flags(profile)]
    for n, f in prof[:4]:
        out.append(f"- **{n}** – {f} (based on your skin profile).")
    out.append("\n### Ingredient-wise analysis")
    for i in flagged:
        out.append(f"- **{i.name}** ({', '.join(i.functions)}) – {i.risk} concern. {i.risk_note}")
    if not flagged:
        out.append("- All recognised ingredients are widely used and generally well tolerated.")
    out.append("\n### Limitations & uncertainty")
    if unknown:
        out.append(f"- Not assessed (not in the knowledge base): {', '.join(unknown[:8])}.")
    out.append("- Risk depends on concentration, which labels do not show.")
    out.append("- This is general information, not medical advice; patch-test new products.")
    if reason:
        out.append(f"\n*AI narrative unavailable ({reason}); showing the rule-based analysis.*")
    return "\n".join(out)


def summary_messages(context: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{context}\n\nTask: Summarise this product's ingredient list.\n{SUMMARY_FORMAT}"}]


def question_messages(question: str, context: str, intent: str, history: list[dict]) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM}]
    msgs += [{"role": h["role"], "content": h["content"][:1200]} for h in history[-4:]]
    msgs.append({"role": "user", "content": (
        f"{context}\n\nDetected intent: {intent}\nQuestion: {question}\n"
        "Answer in under 220 words. Start with a direct answer, then support it with cited evidence, "
        "then one line of caveat.")})
    return msgs


_LEAK = re.compile(r"thinking process|analy[sz]e user input|let'?s draft|check word count|constraint:", re.I)


def stream_answer(settings: Settings, messages: list[dict], meta: dict) -> Iterator[str]:
    """Token stream with any leaked reasoning removed.
    - Normal case: the reply starts with [[ANSWER]] -> stream everything after it live.
    - If a long preamble comes first (the model is 'thinking out loud', and may even quote the marker
      while doing so) we buffer to the end and keep only the text after the LAST marker.
    - If there is no marker and the text looks like reasoning, show nothing and set meta['error']
      so the caller falls back to the deterministic answer."""
    buf, released = "", False
    try:
        for tok in llm.chat_stream(settings, messages, meta, max_tokens=2000):
            if released:
                yield tok
                continue
            buf += tok
            if MARKER in buf and len(buf.split(MARKER, 1)[0].strip()) < 40:
                released = True
                rest = buf.split(MARKER, 1)[1].lstrip("\n: ")
                if rest:
                    yield rest
        if not released and buf:
            if MARKER in buf:
                final = buf.rsplit(MARKER, 1)[1].strip(" \n:")
                if final:
                    yield final
                else:
                    meta["error"] = "model produced no final answer"
            elif _LEAK.search(buf[:600]):
                meta["error"] = "model returned its reasoning instead of an answer"
            else:
                yield buf  # well-behaved model that ignored the marker
    except llm.LLMError as e:
        meta["error"] = str(e)