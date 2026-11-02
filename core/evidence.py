"""Stage 6 – Evidence layer: retrieval from trusted sources.

LIVE APIs (free, no key required):
  • PubMed  – NCBI E-utilities (esearch + efetch) → real abstracts
  • PubChem – PUG REST → molecular formula / weight / IUPAC name
CURATED pointers (these sources have no free public ingredient API):
  • FDA     – regulatory framing + link to FDA's prohibited/restricted list
  • CosIng  – EU Annex status from the curated knowledge base + link to the database
Every item is a plain dict so it can be shown, ranked and cited uniformly."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote

from . import http
from .config import Settings
from .knowledge import FDA_DEFAULT, KB, normalize
from .nlp import RISK_ORDER, Ingredient

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name"
FDA_URL = "https://www.fda.gov/cosmetics/cosmetics-laws-regulations/prohibited-restricted-ingredients-cosmetics"
COSING_URL = "https://single-market-economy.ec.europa.eu/sectors/cosmetics/cosmetic-ingredient-database_en"

_cache = http.TTLCache(ttl=3600, max_items=1024)


def _pm_limiter(settings: Settings) -> http.RateLimiter:
    # NCBI allows 3 req/s without a key, 10 req/s with one.
    return _PM_FAST if settings.ncbi_key else _PM_SLOW


_PM_SLOW = http.RateLimiter(0.36)
_PM_FAST = http.RateLimiter(0.11)
_CHEM_LIMITER = http.RateLimiter(0.22)  # PubChem: max 5 req/s


def _base_params(settings: Settings) -> dict:
    p = {"tool": "cosmos-app"}
    if settings.ncbi_email:
        p["email"] = settings.ncbi_email
    if settings.ncbi_key:
        p["api_key"] = settings.ncbi_key
    return p


# ── PubMed ───────────────────────────────────────────────────────────────────
def fetch_pubmed(terms: list[str] | tuple[str, ...], settings: Settings, retmax: int = 3) -> dict:
    key = ("pm", tuple(terms), retmax)
    if (hit := _cache.get(key)) is not None:
        return hit
    expr = ("(" + " OR ".join(f'"{t}"[Title/Abstract]' for t in terms) + ") "
            "AND (skin OR cosmetic OR dermatolog* OR topical)")
    try:
        data = http.get_json(f"{EUTILS}/esearch.fcgi",
                             {**_base_params(settings), "db": "pubmed", "term": expr, "retmax": retmax,
                              "retmode": "json", "sort": "relevance"},
                             timeout=settings.evidence_timeout, limiter=_pm_limiter(settings))
        ids = (data or {}).get("esearchresult", {}).get("idlist", [])
        if not ids:
            out = {"items": [], "error": None}
            _cache.set(key, out)
            return out
        xml = http.get_text(f"{EUTILS}/efetch.fcgi",
                            {**_base_params(settings), "db": "pubmed", "id": ",".join(ids), "retmode": "xml"},
                            timeout=settings.evidence_timeout, limiter=_pm_limiter(settings))
        items = _parse_pubmed_xml(xml)
        out = {"items": items, "error": None}
        _cache.set(key, out)
        return out
    except Exception as e:
        return {"items": [], "error": f"PubMed: {type(e).__name__}"}


def _parse_pubmed_xml(xml: str) -> list[dict]:
    items = []
    root = ET.fromstring(xml)
    for art in root.findall(".//PubmedArticle"):
        pmid = (art.findtext(".//MedlineCitation/PMID") or "").strip()
        title = "".join(art.find(".//ArticleTitle").itertext()).strip() if art.find(".//ArticleTitle") is not None else ""
        journal = (art.findtext(".//Journal/Title") or "").strip()
        year = art.findtext(".//JournalIssue/PubDate/Year") or ""
        if not year:
            md = art.findtext(".//JournalIssue/PubDate/MedlineDate") or ""
            year = md[:4] if md[:4].isdigit() else ""
        parts = []
        for ab in art.findall(".//Abstract/AbstractText"):
            label = ab.get("Label")
            txt = "".join(ab.itertext()).strip()
            if txt:
                parts.append(f"{label}: {txt}" if label else txt)
        abstract = " ".join(parts)
        pubtypes = [p.text or "" for p in art.findall(".//PublicationType")]
        if not pmid or not title:
            continue
        items.append({
            "source": "PubMed", "id": f"PMID {pmid}", "title": title, "year": int(year) if year.isdigit() else None,
            "venue": journal, "snippet": abstract[:700], "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "kind": "review" if any("Review" in p or "Meta-Analysis" in p for p in pubtypes) else "article",
            "curated": False,
        })
    return items


# ── PubChem ──────────────────────────────────────────────────────────────────
def fetch_pubchem(chem_name: str, settings: Settings) -> dict:
    key = ("pc", chem_name)
    if (hit := _cache.get(key)) is not None:
        return hit
    try:
        data = http.get_json(f"{PUBCHEM}/{quote(chem_name)}/property/MolecularFormula,MolecularWeight,IUPACName/JSON",
                             timeout=settings.evidence_timeout, limiter=_CHEM_LIMITER)
        props = ((data or {}).get("PropertyTable", {}).get("Properties") or [])
        if not props:
            out = {"items": [], "error": None}
            _cache.set(key, out)
            return out
        p = props[0]
        cid = p.get("CID")
        snippet = (f"Formula {p.get('MolecularFormula', '?')} · molecular weight {p.get('MolecularWeight', '?')} g/mol"
                   f" · IUPAC name: {p.get('IUPACName', 'n/a')}")
        out = {"items": [{
            "source": "PubChem", "id": f"CID {cid}", "title": f"PubChem compound record: {chem_name.title()}",
            "year": None, "venue": "PubChem", "snippet": snippet, "url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
            "kind": "database", "curated": False}], "error": None}
        _cache.set(key, out)
        return out
    except Exception as e:
        return {"items": [], "error": f"PubChem: {type(e).__name__}"}


# ── Regulatory (curated) ─────────────────────────────────────────────────────
def regulatory_items(name: str) -> list[dict]:
    e = KB.get(name)
    eu = e.cosing_note if e else "Not in the curated list – search the CosIng database directly."
    return [
        {"source": "FDA", "id": "FDA cosmetics", "title": "FDA – prohibited & restricted cosmetic ingredients", "year": None,
         "venue": "U.S. FDA", "snippet": FDA_DEFAULT, "url": FDA_URL, "kind": "regulatory", "curated": True},
        {"source": "CosIng", "id": "EU CosIng", "title": f"EU CosIng – regulatory status of {name}", "year": None,
         "venue": "European Commission", "snippet": eu, "url": COSING_URL, "kind": "regulatory", "curated": True},
    ]


# ── orchestration ────────────────────────────────────────────────────────────
def fetch_ingredient(ing: Ingredient, settings: Settings, retmax: int = 3) -> dict:
    """All sources for one ingredient. Network calls are skipped when the KB says
    there is nothing sensible to search (e.g. water, polymers)."""
    items, errors = [], []
    terms = ing.pm_terms or ([ing.name.lower()] if not ing.matched else None)
    if terms:
        r = fetch_pubmed(terms, settings, retmax)
        items += r["items"]
        if r["error"]:
            errors.append(r["error"])
    chem = ing.chem if ing.matched else ing.name.lower()
    if chem:
        r = fetch_pubchem(chem, settings)
        items += r["items"]
        if r["error"]:
            errors.append(r["error"])
    items += regulatory_items(ing.name)
    return {"items": items, "errors": errors}


def prioritise(ings: list[Ingredient]) -> list[Ingredient]:
    return sorted(ings, key=lambda i: (RISK_ORDER.get(i.risk, 3), i.position))


def gather(ings: list[Ingredient], settings: Settings, depth: int = 4, workers: int = 4,
           on_done=None) -> dict[str, dict]:
    """Fetch evidence for the `depth` highest-priority ingredients in parallel.
    One slow or failing source never blocks the others."""
    chosen = prioritise(ings)[:depth]
    results: dict[str, dict] = {}
    if not chosen:
        return results
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_ingredient, i, settings): i for i in chosen}
        try:
            for fut in as_completed(futures, timeout=settings.evidence_timeout * 3):
                ing = futures[fut]
                try:
                    results[ing.name] = fut.result()
                except Exception as e:
                    results[ing.name] = {"items": regulatory_items(ing.name), "errors": [type(e).__name__]}
                if on_done:
                    on_done(len(results), len(chosen), ing.name)
        except Exception:
            pass  # overall timeout: keep what we have
    for i in chosen:
        results.setdefault(i.name, {"items": regulatory_items(i.name), "errors": ["timed out"]})
    return results
