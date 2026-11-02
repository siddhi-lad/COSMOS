"""Evaluation framework (stage 9). Run from the project root:

  python evaluation/evaluate.py                      # offline: identification, function, risk, intent
  python evaluation/evaluate.py --ocr-dir data/ocr   # + OCR: needs API keys; folder holds img.jpg + img.txt (ground truth)
  python evaluation/evaluate.py --latency            # + timing of the offline NLP stage

Metrics: precision / recall / F1, exact set match, function & risk accuracy,
intent accuracy & macro-F1 (+ confusions), OCR CER / WER / success rate / latency."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import intent, nlp, ocr  # noqa: E402
from core.config import get_settings  # noqa: E402


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def eval_nlp(samples: list[dict]) -> None:
    tp = fp = fn = exact = 0
    f_ok = f_n = r_ok = r_n = 0
    for s in samples:
        pred = nlp.analyze_text(s["text"])
        P, G = {i.name for i in pred}, set(s["ingredients"])
        tp += len(P & G)
        fp += len(P - G)
        fn += len(G - P)
        exact += P == G
        by = {i.name: i for i in pred}
        for k, v in s.get("functions", {}).items():
            f_n += 1
            f_ok += k in by and v in by[k].functions
        for k, v in s.get("risk", {}).items():
            r_n += 1
            r_ok += k in by and by[k].risk == v
    p, r, f = prf(tp, fp, fn)
    print("== Ingredient identification ==")
    print(f"precision {p:.3f}  recall {r:.3f}  F1 {f:.3f}  exact-set-match {exact}/{len(samples)}")
    print(f"== Function mapping ==  accuracy {f_ok}/{f_n} = {f_ok / max(f_n, 1):.3f}")
    print(f"== Risk assessment ==   accuracy {r_ok}/{r_n} = {r_ok / max(r_n, 1):.3f}")


def eval_intent(pairs: list[list[str]]) -> None:
    conf: dict[str, Counter] = defaultdict(Counter)
    correct = 0
    for q, gold in pairs:
        pred = intent.classify(q).intent
        conf[gold][pred] += 1
        correct += pred == gold
    f1s = []
    for lab in intent.INTENTS:
        tp = conf[lab][lab]
        fp = sum(conf[g][lab] for g in conf if g != lab)
        fn = sum(v for k, v in conf[lab].items() if k != lab)
        if tp + fn:  # only labels present in the gold set
            f1s.append(prf(tp, fp, fn)[2])
    print("== Intent classification (rules only) ==")
    print(f"accuracy {correct}/{len(pairs)} = {correct / len(pairs):.3f}  macro-F1 {statistics.mean(f1s):.3f}")
    for g, row in conf.items():
        wrong = {k: v for k, v in row.items() if k != g}
        if wrong:
            print(f"  gold {g}: misclassified as {dict(wrong)}")


def eval_ocr(folder: Path) -> None:
    settings = get_settings()
    cers, wers, lats, exact_sets = [], [], [], []
    ok = n = 0
    for img in sorted(folder.glob("*")):
        gt = img.with_suffix(".txt")
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp") or not gt.exists():
            continue
        n += 1
        res = ocr.run_ocr(img.read_bytes(), settings)
        lats.append(res.latency)
        if not res.ok:
            print(f"  {img.name}: OCR failed – {res.attempts[-1]['error']}")
            continue
        ok += 1
        truth = gt.read_text(encoding="utf-8").strip()
        cers.append(edit_distance(list(res.text), list(truth)) / max(len(truth), 1))
        wers.append(edit_distance(res.text.split(), truth.split()) / max(len(truth.split()), 1))
        exact_sets.append({i.name for i in nlp.analyze_text(res.text)} == {i.name for i in nlp.analyze_text(truth)})
    print("== OCR ==")
    if n:
        print(f"success rate {ok}/{n}  CER {statistics.mean(cers or [0]):.3f}  WER {statistics.mean(wers or [0]):.3f}  "
              f"exact ingredient-set match {sum(exact_sets)}/{len(exact_sets)}  "
              f"latency median {statistics.median(lats):.1f}s")
    else:
        print("no image + .txt pairs found")


def eval_latency(samples: list[dict]) -> None:
    ts = []
    for s in samples:
        t = time.time()
        nlp.analyze_text(s["text"])
        ts.append(time.time() - t)
    print(f"== Latency (NLP stage, offline) == median {statistics.median(ts) * 1000:.2f} ms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(ROOT / "evaluation" / "gold.json"))
    ap.add_argument("--ocr-dir")
    ap.add_argument("--latency", action="store_true")
    a = ap.parse_args()
    gold = json.loads(Path(a.gold).read_text(encoding="utf-8"))
    eval_nlp(gold["samples"])
    eval_intent(gold["intents"])
    if a.latency:
        eval_latency(gold["samples"])
    if a.ocr_dir:
        eval_ocr(Path(a.ocr_dir))
