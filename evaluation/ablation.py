"""Offline NLP ablation, knowledge-base gap analysis and latency on the ToxFox gold labels. No API needed.

  python evaluation/ablation.py --data data_tox

Writes to results/: summary_ablation.(md|csv|tex|json), kb_gaps_unmatched.csv, kb_gaps_missed_gold.csv, latency.json,
run_manifest.json.  Each ablation switches ONE component off and re-measures ingredient identification, so the paper
can show what every component contributes. Significance: paired Wilcoxon on per-image F1 vs the full system,
Holm-corrected across the ablations."""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import toxfox_eval as tx  # noqa: E402
from core import nlp  # noqa: E402
from stats import boot_ci, fmt_ci, holm, prf, wilcoxon_p, write_manifest  # noqa: E402


@contextlib.contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


@contextlib.contextmanager
def stack(*cms):
    with contextlib.ExitStack() as st:
        for c in cms:
            st.enter_context(c)
        yield


CONFIGS = {
    "Full system": (lambda: stack(), False),
    "- fuzzy matching": (lambda: patched(nlp.difflib, "get_close_matches", lambda *a, **k: []), False),
    "- prefix guard (di-/tri-/poly-)": (lambda: patched(nlp, "_is_homologue", lambda a, b: False), False),
    "- line-break splitting": (lambda: stack(), True),  # newlines flattened to spaces = behaviour before the fix
    "- synonym merge (aqua=water, parfum=fragrance)": (lambda: patched(tx, "SYNONYMS", {}), False),
}


def per_image(sample: dict, flatten: bool) -> dict:
    text = sample["gold_text"].replace("\n", " ") if flatten else sample["gold_text"]
    ings = nlp.analyze_text(text)
    pred = {tx.canon(i.name) for i in ings if tx.canon(i.name)}
    gold = tx.gold_set(sample["gold_inci"])
    tp, fp, fn = len(pred & gold), len(pred - gold), len(gold - pred)
    return {"id": sample["id"], "ident": (tp, fp, fn), "f1": prf(tp, fp, fn)[2], "exact": int(pred == gold),
            "coverage": (sum(i.matched for i in ings) / len(ings)) if ings else 0.0,
            "pred": pred, "gold": gold, "ings": ings}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    a = ap.parse_args()

    samples = tx.load_dataset(a.data)
    if not samples:
        raise SystemExit("No usable samples found in --data.")
    random.Random(a.seed).shuffle(samples)
    samples = samples[: a.n]
    print(f"{len(samples)} images")

    results, full_rows = [], None
    rows_by_cfg = {}
    for name, (ctx, flatten) in CONFIGS.items():
        with ctx():
            rows = [per_image(s, flatten) for s in samples]
        rows_by_cfg[name] = rows
        if name == "Full system":
            full_rows = rows
        mic = lambda w: (lambda rs: prf(*(sum(r["ident"][i] for r in rs) for i in range(3)))[w])
        results.append({
            "configuration": name, "n_images": len(rows),
            "P_%": fmt_ci(boot_ci(rows, mic(0))), "R_%": fmt_ci(boot_ci(rows, mic(1))),
            "F1_%": fmt_ci(boot_ci(rows, mic(2))),
            "exact_match_%": fmt_ci(boot_ci(rows, lambda rs: statistics.fmean(r["exact"] for r in rs))),
            "_f1": [r["f1"] for r in rows]})
    pv = [None] + [wilcoxon_p(results[i]["_f1"], results[0]["_f1"]) for i in range(1, len(results))]
    adj = holm(pv)
    for r, p, q in zip(results, pv, adj):
        r["p_vs_full"] = "-" if p is None else f"{p:.4f}"
        r["p_holm"] = "-" if q is None else f"{q:.4f}"
        r.pop("_f1")
    tx.write_outputs(a.out, results, {}, "ablation")

    # knowledge-base gaps (full system)
    unmatched, missed = Counter(), Counter()
    for r in full_rows:
        for i in r["ings"]:
            if not i.matched:
                unmatched[tx.canon(i.name)] += 1
        for m in r["gold"] - r["pred"]:
            missed[m] += 1
    for fname, ctr, col in (("kb_gaps_unmatched.csv", unmatched, "ingredient_not_in_knowledge_base"),
                            ("kb_gaps_missed_gold.csv", missed, "gold_ingredient_not_found_by_nlp")):
        with open(a.out / fname, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([col, "n_labels"])
            w.writerows(ctr.most_common())
    n_ing = sum(len(r["ings"]) for r in full_rows)
    n_match = sum(i.matched for r in full_rows for i in r["ings"])
    risk = Counter(i.risk for r in full_rows for i in r["ings"])
    print(f"\nKB coverage: {n_match}/{n_ing} ingredient mentions ({100 * n_match / max(1, n_ing):.1f}%); "
          f"risk distribution {dict(risk)}")
    print(f"Top unmatched: {unmatched.most_common(10)}")

    # latency of the deterministic NLP stage
    times = []
    for s in samples:
        t0 = time.perf_counter()
        nlp.analyze_text(s["gold_text"])
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    lat = {"nlp_ms_median": round(statistics.median(times), 2), "nlp_ms_p95": round(times[int(0.95 * len(times)) - 1], 2),
           "n": len(times), "kb_coverage_mentions": f"{n_match}/{n_ing}", "risk_distribution": dict(risk)}
    (a.out / "latency.json").write_text(json.dumps(lat, indent=2), encoding="utf-8")
    print("Latency:", lat)
    write_manifest(a.out, {"dataset_images": len(samples), "seed": a.seed})


if __name__ == "__main__":
    main()
