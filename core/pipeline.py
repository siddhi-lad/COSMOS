"""Orchestrates stages 1→7 for one scan. The UI calls `run_scan` and then streams the summary."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from . import evidence, ling_sante, nlp, ocr, processing, synthesis
from .config import Settings
from .knowledge import KB
from .nlp import Ingredient


@dataclass
class ScanResult:
    id: str
    label: str
    created: datetime
    source: str  # image | manual
    raw_text: str
    ingredients: list[Ingredient]
    ocr: ocr.OCRResult | None = None
    evidence: dict[str, list[dict]] = field(default_factory=dict)  # processed, ranked
    evidence_errors: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    summary: str = ""
    summary_model: str = ""
    context_numbered: list[dict] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return [i.name for i in self.ingredients]


def run_scan(label: str, settings: Settings, depth: int, *, image_bytes: bytes | None = None,
             manual_text: str = "", progress=None) -> ScanResult:
    """`progress(stage:str, detail:str)` is an optional callback for live status."""
    say = progress or (lambda *_: None)
    timings: dict[str, float] = {}
    ocr_res = None
    text = manual_text
    if image_bytes is not None:
        say("ocr", "Reading the label with cloud OCR…")
        t0 = time.time()
        ocr_res = ocr.run_ocr(image_bytes, settings)
        timings["OCR"] = round(time.time() - t0, 1)
        text = ocr_res.text
    scan = ScanResult(uuid_str(), label, datetime.now(), "image" if image_bytes is not None else "manual", text, [], ocr_res)
    if not text:
        scan.timings = timings
        return scan

    say("nlp", "Cleaning text and identifying ingredients…")
    t0 = time.time()
    scan.ingredients = nlp.analyze_text(text)
    timings["NLP"] = round(time.time() - t0, 2)

    if scan.ingredients and depth > 0:
        say("evidence", "Retrieving evidence from PubMed, PubChem, FDA and CosIng…")
        t0 = time.time()
        add_evidence(scan, settings, depth)
        timings["Evidence"] = round(time.time() - t0, 1)
    scan.timings = timings
    return scan


def add_evidence(scan: ScanResult, settings: Settings, depth: int, only: list[Ingredient] | None = None,
                 on_done=None) -> None:
    targets = only if only is not None else scan.ingredients
    raw = evidence.gather(targets, settings, depth=depth, on_done=on_done)
    for name, res in raw.items():
        e = KB.get(name)
        scan.evidence[name] = processing.process(name, res["items"], list(e.aliases) if e else None)
        scan.evidence_errors += res["errors"]
    ling_sante.validate({n: scan.evidence[n] for n in raw}, settings)  # one batched LLM check; safe no-op on failure


def prepare_summary(scan: ScanResult, profile: list[str]) -> list[dict]:
    """Builds the LLM messages for the product summary and remembers the numbered evidence."""
    ctx, numbered = synthesis.build_context(scan.ingredients, scan.evidence, profile)
    scan.context_numbered = numbered
    return synthesis.summary_messages(ctx)


def finalize_summary(scan: ScanResult, text: str, model: str, profile: list[str], error: str = "") -> str:
    if not text.strip():
        scan.summary = synthesis.fallback_summary(scan.ingredients, profile, error or "no response")
        scan.summary_model = "rule-based"
        return scan.summary
    cleaned, used = synthesis.validate_citations(text, len(scan.context_numbered))
    scan.summary = cleaned + synthesis.sources_markdown(scan.context_numbered, used)
    scan.summary_model = model
    return scan.summary


def uuid_str() -> str:
    return uuid.uuid4().hex[:8]
