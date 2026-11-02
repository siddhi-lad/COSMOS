"""Human-annotation workflow for the parts of COSMOS that need human judgement
(intent classifier, evidence relevance / ranking / Ling Sante, answer faithfulness).

  python evaluation/annotation_tools.py make-intents                 # writes annotation/intent_questions.csv
  python evaluation/annotation_tools.py collect-evidence --n 30       # needs internet (PubMed) [+ OpenRouter key for Ling Sante]
  python evaluation/annotation_tools.py collect-answers  --n 20       # needs internet + LLM keys
  ... two people fill the rater columns (see EVALUATION.md) ...
  python evaluation/annotation_tools.py score                         # computes every metric + tables for the paper

Gold labels come ONLY from the humans. The intent file deliberately contains no suggested label, so the results are
not circular. Rater columns are left empty; `adjudicated` is the agreed final label (if empty, rater 1 is used)."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stats import (boot_ci, cohen_kappa, fmt_ci, macro_f1, mrr, ndcg_at_k, precision_at_k,  # noqa: E402
                   write_manifest)

ANN = ROOT / "evaluation" / "annotation"
OUT = ROOT / "results"

QUESTIONS = """What does niacinamide do in a moisturiser?|What is the function of glycerin in skincare?|Why is phenoxyethanol added to creams?|What role does hyaluronic acid play in a serum?|What is cetearyl alcohol used for?|Does retinol actually work for fine lines?|What is the purpose of tocopherol in a formula?|Why do shampoos contain cocamidopropyl betaine?|What benefit does panthenol give the skin?|What does salicylic acid do in a face wash?
Is phenoxyethanol safe for daily use?|Can methylparaben irritate sensitive skin?|Is fragrance a common allergen?|Are parabens harmful?|Is retinol safe during pregnancy?|Does sodium lauryl sulfate cause dryness?|Can limonene cause an allergic reaction?|Is benzyl alcohol toxic in cosmetics?|What are the side effects of salicylic acid?|Is triclosan dangerous to use on skin?
What is INCI name for vitamin E?|What is aqua on an ingredient list?|Which ingredient on this label is a preservative?|What is parfum?|Identify the ingredients in this list.|What is another name for sodium hyaluronate?|What are the ingredients of this shampoo?|Is tocopheryl acetate the same as tocopherol?|What is cetyl alcohol also known as?|What is CI 77891?
Is methylisothiazolinone banned in the EU?|What is the maximum concentration of salicylic acid allowed?|Is triclosan restricted in cosmetics?|Is butylparaben approved for use in Europe?|Which EU regulation covers cosmetic preservatives?|Is hydroquinone legal in skin creams?|Does the FDA regulate sunscreen filters?|Are phthalates prohibited in cosmetics?|Is formaldehyde allowed in nail products?|What limits apply to retinol in the EU?
Is this product suitable for oily skin?|Would this moisturiser be good for dry skin?|Can I use this cream if I have acne?|Is this shampoo ok for a sensitive scalp?|Should someone with rosacea use this product?|Is this serum recommended for pregnant women?|Is this sunscreen right for children?|Is this product okay for eczema-prone skin?|Would this cleanser suit combination skin?|Is this lotion fine for daily use on my face?
Are there studies on niacinamide for acne?|Is there clinical evidence that hyaluronic acid hydrates skin?|What research exists on phenoxyethanol safety?|Can you cite papers on retinol and photoaging?|Show me published evidence about parabens.|Is there any proof that vitamin C brightens skin?|What does PubMed say about salicylic acid?|Are there clinical trials on ceramides?|Which studies looked at fragrance allergy?|Give me sources on sunscreen filter absorption.
How should I layer skincare products?|What order do I apply serum and moisturiser?|How often should I exfoliate?|Does the pH of a cleanser matter?|What is a patch test?|How long does a moisturiser last once opened?|What does non-comedogenic mean?|Is natural better than synthetic in cosmetics?|What is the difference between a serum and a lotion?|How do I read a cosmetic label?"""


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig so Excel opens it correctly
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ───────────────────────── make-intents ─────────────────────────
def make_intents(a) -> None:
    from core.intent import INTENTS
    groups = [[q.strip() for q in line.split("|") if q.strip()] for line in QUESTIONS.splitlines()]
    n = max(7, min(a.n, sum(map(len, groups))))
    per = -(-n // len(groups))  # ceil: same number from every question group, so all topics are covered
    rng = random.Random(7)
    qs = [q for g in groups for q in rng.sample(g, min(per, len(g)))][:n]
    rng.shuffle(qs)  # order carries no information about the category
    rows = [{"id": f"q{i:03d}", "question": q, "annotator1": "", "annotator2": "", "adjudicated": ""}
            for i, q in enumerate(qs, 1)]
    write_csv(ANN / "intent_questions.csv", rows, ["id", "question", "annotator1", "annotator2", "adjudicated"])
    (ANN / "intent_labels.txt").write_text(
        "Allowed labels (copy exactly):\n" + "\n".join(INTENTS) + "\n\n"
        "ingredient_function = what an ingredient does/why it is used\n"
        "ingredient_safety = safety, irritation, allergy, pregnancy, toxicity\n"
        "ingredient_identification = what something is / INCI / synonyms / identify from a list\n"
        "regulatory_status = bans, limits, approvals, laws\n"
        "product_suitability = is this product right for a skin type / person\n"
        "evidence_lookup = asks for studies, papers, evidence, sources\n"
        "general_cosmetic = anything else about cosmetics\n", encoding="utf-8")
    print(f"Wrote {ANN / 'intent_questions.csv'} ({len(rows)} questions) and intent_labels.txt\n"
          "Give the file to 2 people independently: each fills ONE column (annotator1 / annotator2) using the labels in "
          "intent_labels.txt, without seeing each other's work. Then agree the disputed ones in `adjudicated`.")


# ───────────────────────── collect-evidence ─────────────────────────
def collect_evidence(a) -> None:
    from core import evidence, ling_sante, nlp, processing, synthesis
    from core.config import get_settings
    from core.knowledge import KB
    s = get_settings()
    names = [n for n, e in KB.items() if e.pm_terms or e.chem]
    random.Random(a.seed).shuffle(names)
    rows = []
    for name in names[: a.n]:
        ing = nlp.identify(name)
        res = evidence.fetch_ingredient(ing, s, retmax=a.retmax)
        pm = [it for it in res["items"] if it["source"] == "PubMed" and it.get("snippet")]
        if not pm:
            continue
        for i, it in enumerate(pm, 1):
            it["raw_rank"] = i
        proc = processing.process(name, pm, list(KB[name].aliases))
        for i, it in enumerate(proc, 1):
            it["rule_rank"] = i
        ev_ling = {name: copy.deepcopy(proc)}
        ran = ling_sante.validate(ev_ling, s)
        ling = ev_ling[name]
        for i, it in enumerate(ling, 1):
            it["ling_rank"] = i
        by_url = {x["url"]: x for x in ling}
        ctx_rule, num_rule = synthesis.build_context([ing], {name: proc}, [])
        ctx_ling, num_ling = synthesis.build_context([ing], {name: ling}, [])
        cited_rule = {x["url"] for x in num_rule}
        cited_ling = {x["url"] for x in num_ling}
        for it in proc:
            L = by_url.get(it["url"], {})
            rows.append({"ingredient": name, "pmid_url": it["url"], "title": it["title"], "year": it.get("year", ""),
                         "abstract_excerpt": it["snippet"][:400], "raw_rank": it["raw_rank"], "rule_rank": it["rule_rank"],
                         "ling_rank": L.get("ling_rank", ""), "ling_verdict": L.get("ai_verdict", "") if ran else "",
                         "cited_by_rules": int(it["url"] in cited_rule), "cited_by_rules_plus_ling": int(it["url"] in cited_ling),
                         "rater1": "", "rater2": "", "adjudicated": ""})
        time.sleep(0.4)
    if not rows:
        raise SystemExit("No evidence retrieved - check your internet connection (PubMed).")
    write_csv(ANN / "evidence_relevance.csv", rows,
              ["ingredient", "pmid_url", "title", "year", "abstract_excerpt", "raw_rank", "rule_rank", "ling_rank",
               "ling_verdict", "cited_by_rules", "cited_by_rules_plus_ling", "rater1", "rater2", "adjudicated"])
    print(f"Wrote {ANN / 'evidence_relevance.csv'} ({len(rows)} papers, {len({r['ingredient'] for r in rows})} ingredients).\n"
          "Raters: for every row put 1 if the paper is relevant to the use or safety of THAT ingredient in cosmetic / skin "
          "products, else 0 (rater1 / rater2 independently; agree disputed ones in `adjudicated`). "
          "Do not look at the rank or verdict columns while rating.")


# ───────────────────────── collect-answers ─────────────────────────
def collect_answers(a) -> None:
    from core import evidence, intent, ling_sante, nlp, processing, synthesis
    from core.config import get_settings
    from core.knowledge import KB
    s = get_settings()
    if not s.has_llm:
        raise SystemExit("Put OPENROUTER_API_KEY / NVIDIA_API_KEY in .env first.")
    src = ANN / "intent_questions.csv"
    if not src.is_file():
        raise SystemExit("Run make-intents first.")
    qs = [r["question"] for r in read_csv(src)]
    qs = [q for q in qs if intent.find_mentions(q)]  # questions that name an ingredient the KB knows
    random.Random(a.seed).shuffle(qs)
    rows = []
    for k, q in enumerate(qs[: a.n], 1):
        t0 = time.time()
        res = intent.classify(q, s, [])
        ings = [nlp.identify(n) for n in res.ingredients[:3]]
        raw = evidence.gather(ings, s, depth=len(ings) or 1)
        ev = {n: processing.process(n, r["items"], list(KB[n].aliases) if n in KB else None) for n, r in raw.items()}
        ling_sante.validate(ev, s)
        ctx, numbered = synthesis.build_context(ings, ev, [])
        meta: dict = {}
        text = "".join(synthesis.stream_answer(s, synthesis.question_messages(q, ctx, res.intent, []), meta))
        cleaned, used = synthesis.validate_citations(text, len(numbered)) if text else ("", [])
        raw_cites = {int(x) for m in re.findall(r"\[((?:\d+\s*,?\s*)+)\]", text) for x in re.findall(r"\d+", m)}
        rows.append({"id": f"a{k:03d}", "question": q, "answer": cleaned, "n_evidence_items": len(numbered),
                     "n_valid_citations": len(used), "n_invalid_citations_removed": len([c for c in raw_cites if c > len(numbered) or c < 1]),
                     "fallback_used": int(not text), "reasoning_leak": int(bool(synthesis._LEAK.search(text))),
                     "marker_in_output": int(synthesis.MARKER in cleaned), "seconds": round(time.time() - t0, 1),
                     "evidence_shown_to_rater": " || ".join(f"[{i}] {x['title']} ({x['source']})" for i, x in enumerate(numbered, 1)),
                     "faithful_1to5_r1": "", "faithful_1to5_r2": "", "unsupported_claims_r1": "", "unsupported_claims_r2": "",
                     "helpful_1to5_r1": "", "helpful_1to5_r2": ""})
        print(f"  {k}/{min(a.n, len(qs))}")
    write_csv(ANN / "answer_quality.csv", rows, list(rows[0]))
    print(f"Wrote {ANN / 'answer_quality.csv'}.\nRaters: faithful 1-5 (are all claims supported by the evidence/knowledge shown? 5 = fully), "
          "unsupported_claims = number of claims NOT supported, helpful 1-5. Two raters, independently.")


# ───────────────────────── score ─────────────────────────
def label(r: dict, *cols: str) -> str:
    for c in ("adjudicated", *cols):
        if (r.get(c) or "").strip():
            return r[c].strip()
    return ""


def score(a) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {}
    md: list[str] = []
    f = ANN / "intent_questions.csv"
    if f.is_file():
        rows = [r for r in read_csv(f) if label(r, "annotator1")]
        if rows:
            from core import intent
            from core.config import get_settings
            st = get_settings() if a.llm else None
            gold = [label(r, "annotator1") for r in rows]
            pred_rules = [intent.classify_rules(r["question"])[0] for r in rows]
            res = {"n": len(rows)}
            for name, pred in (("rules", pred_rules),
                               *((("rules+LLM", [intent.classify(r["question"], st, []).intent for r in rows]),) if a.llm else ())):
                acc = boot_ci(list(zip(gold, pred)), lambda ps: statistics.fmean(g == p for g, p in ps))
                mf, per = macro_f1(gold, pred)
                res[name] = {"accuracy": fmt_ci(acc), "macro_F1_%": round(100 * mf, 1),
                             "per_class_P_R_F1_n": {k: [round(x, 2) for x in v[:3]] + [v[3]] for k, v in per.items()}}
            both = [(r["annotator1"], r["annotator2"]) for r in rows if r["annotator1"].strip() and r["annotator2"].strip()]
            if both:
                res["inter_annotator_kappa"] = round(cohen_kappa([x for x, _ in both], [y for _, y in both]), 3)
            report["intent"] = res
            md.append(f"**Intent classifier** (n={res['n']}): " + ", ".join(
                f"{k} accuracy {res[k]['accuracy']}%, macro-F1 {res[k]['macro_F1_%']}%" for k in ("rules", "rules+LLM") if k in res)
                + (f"; annotator kappa {res['inter_annotator_kappa']}" if "inter_annotator_kappa" in res else ""))
    f = ANN / "evidence_relevance.csv"
    if f.is_file():
        rows = [r for r in read_csv(f) if label(r, "rater1") != ""]
        if rows:
            by_ing: dict[str, list[dict]] = {}
            for r in rows:
                by_ing.setdefault(r["ingredient"], []).append(r)
            res = {"n_papers": len(rows), "n_ingredients": len(by_ing)}
            table = []
            for name, key in (("Raw PubMed order", "raw_rank"), ("Rule-based ranking", "rule_rank"),
                              ("Rules + Ling Sante", "ling_rank")):
                lists = []
                for items in by_ing.values():
                    ranked = [i for i in items if str(i.get(key, "")).strip()]
                    ranked.sort(key=lambda i: int(i[key]))
                    lists.append([int(label(i, "rater1") == "1") for i in ranked])
                lists = [l for l in lists if l]
                row = {"system": name}
                for lab, fn in (("P@3", lambda l: precision_at_k(l, 3)), ("MRR", mrr), ("nDCG@5", lambda l: ndcg_at_k(l, 5))):
                    row[lab] = fmt_ci(boot_ci(lists, lambda ls, fn=fn: statistics.fmean(fn(l) for l in ls)), pct=False, nd=3)
                table.append(row)
            res["ranking"] = table
            for name, col in (("rules", "cited_by_rules"), ("rules+ling", "cited_by_rules_plus_ling")):
                cited = [int(label(r, "rater1") == "1") for r in rows if r.get(col) == "1"]
                res[f"citation_precision_{name}"] = f"{100 * sum(cited) / len(cited):.1f}% ({sum(cited)}/{len(cited)})" if cited else "n/a"
            lv = [(r["ling_verdict"], int(label(r, "rater1") == "1")) for r in rows if r.get("ling_verdict")]
            if lv:
                tp = sum(v == "irrelevant" and h == 0 for v, h in lv)
                fp = sum(v == "irrelevant" and h == 1 for v, h in lv)
                fn_ = sum(v != "irrelevant" and h == 0 for v, h in lv)
                p, r_, f1 = (tp / (tp + fp) if tp + fp else 0, tp / (tp + fn_) if tp + fn_ else 0, 0)
                f1 = 2 * p * r_ / (p + r_) if p + r_ else 0
                res["ling_sante_detects_irrelevant"] = {"precision": round(p, 3), "recall": round(r_, 3), "F1": round(f1, 3), "n": len(lv)}
            both = [(r["rater1"], r["rater2"]) for r in rows if r["rater1"].strip() and r["rater2"].strip()]
            if both:
                res["rater_kappa"] = round(cohen_kappa([x for x, _ in both], [y for _, y in both]), 3)
            report["evidence"] = res
            md.append("**Evidence layer**: " + json.dumps(res["ranking"]) + f"; citation precision rules {res.get('citation_precision_rules')}, "
                      f"rules+Ling {res.get('citation_precision_rules+ling')}")
    f = ANN / "answer_quality.csv"
    if f.is_file():
        rows = read_csv(f)
        res = {"n_answers": len(rows)}
        num = lambda r, c: float(r[c]) if str(r.get(c, "")).strip() else None
        res["with_valid_citation_%"] = round(100 * statistics.fmean(int(r["n_valid_citations"]) > 0 for r in rows), 1)
        res["invalid_citation_rate_%"] = round(100 * statistics.fmean(int(r["n_invalid_citations_removed"]) > 0 for r in rows), 1)
        res["fallback_rate_%"] = round(100 * statistics.fmean(int(r["fallback_used"]) for r in rows), 1)
        res["reasoning_leak_rate_%"] = round(100 * statistics.fmean(int(r["reasoning_leak"]) or int(r["marker_in_output"]) for r in rows), 1)
        res["median_seconds"] = round(statistics.median(float(r["seconds"]) for r in rows), 1)
        for key in ("faithful_1to5", "unsupported_claims", "helpful_1to5"):
            vals = []
            for r in rows:
                v = [num(r, f"{key}_r1"), num(r, f"{key}_r2")]
                v = [x for x in v if x is not None]
                if v:
                    vals.append(statistics.fmean(v))
            if vals:
                res[key + "_mean"] = fmt_ci(boot_ci(vals, statistics.fmean), pct=False, nd=2)
            k_pairs = [(r[f"{key}_r1"], r[f"{key}_r2"]) for r in rows if str(r.get(f"{key}_r1", "")).strip() and str(r.get(f"{key}_r2", "")).strip()]
            if k_pairs:
                res[key + "_kappa_linear"] = round(cohen_kappa([x for x, _ in k_pairs], [y for _, y in k_pairs], "linear"), 3)
        if res.get("unsupported_claims_mean"):
            report["answers"] = res
        else:
            report["answers_auto_only"] = res
        md.append("**Answers**: " + json.dumps(res))
    if not report:
        raise SystemExit("Nothing to score yet: fill in the annotation CSVs first (see EVALUATION.md).")
    if (ANN / "LLM_LABELS.txt").is_file():
        note = ("NOTE for the paper: the intent labels were assigned by an LLM annotator (Claude), independently of the rule-based "
                "classifier, not by human raters; report this as an LLM-annotated pilot. Suggested sentence: "
                "\"Gold intent labels for n questions were produced by an LLM annotator following written label definitions; "
                "no human inter-annotator agreement was measured.\"")
        md.append(note)
    (OUT / "annotation_scores.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "annotation_scores.md").write_text("\n\n".join(md), encoding="utf-8")
    write_manifest(OUT, {"annotation_scoring": True})
    print("\n\n".join(md), f"\n\nSaved {OUT / 'annotation_scores.json'} and annotation_scores.md")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("make-intents")
    p.add_argument("--n", type=int, default=70, help="number of questions (7-70); 30 is a good short version")
    p.set_defaults(fn=make_intents)
    p = sub.add_parser("collect-evidence")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--retmax", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(fn=collect_evidence)
    p = sub.add_parser("collect-answers")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(fn=collect_answers)
    p = sub.add_parser("score")
    p.add_argument("--llm", action="store_true", help="also score the rules+LLM intent classifier (uses API calls)")
    p.set_defaults(fn=score)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
