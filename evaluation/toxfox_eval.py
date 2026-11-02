"""COSMOS evaluation on the ToxFox OCR dataset (ki-iw/toxfox-ocr, data/images + data/ground_truth).

Three experiments, one command each (run from the project root):

  python evaluation/toxfox_eval.py --data DATA_DIR --mode text            # offline, no API, seconds
  python evaluation/toxfox_eval.py --data DATA_DIR --mode ocr             # needs API keys in .env
  python evaluation/toxfox_eval.py --data DATA_DIR --mode ocr --degrade blur   # robustness (also: lowres, jpeg, dark)

DATA_DIR = any folder that contains the *.yaml files and the images (sub-folders are searched).

What is measured
  text mode  (gold label text -> COSMOS NLP): ingredient-identification P/R/F1 vs INCI_list, KB coverage,
             and "concern-flag" P/R/F1 vs ToxFox's pollutant list (independent of the KB: nothing was added to it).
  ocr mode   (image -> OCR -> NLP): CER / WER on the ingredient section vs raw_ingredients_list, success rate,
             latency, plus the same identification and flag metrics end-to-end - per OCR route and for the full
             fallback chain. Re-runs reuse cached OCR output (results/ocr_cache) so you never pay twice.

All numbers come with 95 % bootstrap confidence intervals (resampling images, seed fixed)."""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats import write_manifest  # noqa: E402

from core import llm, nlp, ocr  # noqa: E402
from core.config import get_settings  # noqa: E402
from core.knowledge import normalize  # noqa: E402

SYNONYMS = {"aqua": "water", "eau": "water", "aqua water": "water", "water aqua": "water",
            "parfum": "fragrance", "fragrance": "fragrance", "parfum fragrance": "fragrance"}


# ───────────────────────────── data ─────────────────────────────
def load_dataset(data_dir: Path) -> list[dict]:
    images = {}
    for p in data_dir.rglob("*"):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            images[p.stem] = p
    samples = []
    for y in sorted(data_dir.rglob("*.yaml")):
        try:
            d = yaml.safe_load(y.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, dict) or not d.get("raw_ingredients_list") or not d.get("INCI_list"):
            continue
        stem = Path(str(d.get("image_name", y.stem))).stem
        samples.append({"id": str(d.get("id", y.stem)), "gold_text": str(d["raw_ingredients_list"]),
                        "gold_inci": [str(x) for x in d["INCI_list"]], "product": d.get("product_name", ""),
                        "image": images.get(stem) or images.get(y.stem)})
    return samples


def load_pollutants(path: Path | None) -> dict[str, str]:
    if not path or not path.is_file():
        return {}
    d = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {normalize(k): str(v).strip() for k, v in d.items()}


def canon(name: str) -> str:
    """Comparison key for an ingredient name: case/punctuation-free, brackets removed, synonyms merged."""
    n = normalize(re.sub(r"\([^)]*\)", " ", name))
    return SYNONYMS.get(n, n)


def gold_set(inci: list[str]) -> set[str]:
    return {canon(x) for x in inci if canon(x)}


# ───────────────────────────── metrics ─────────────────────────────
def edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def norm_text(t: str) -> str:
    t = nlp.clean_text(t).lower()
    t = re.sub(r"\s+", " ", t).strip(" .,;:")
    return t


def cer_wer(pred: str, gold: str) -> tuple[float, float]:
    p, g = norm_text(pred), norm_text(gold)
    if not g:
        return 0.0, 0.0
    return edit_distance(p, g) / len(g), edit_distance(p.split(), g.split()) / max(1, len(g.split()))


def strip_brackets(t: str) -> str:
    return re.sub(r"\s*\([^)]*\)", "", t)


def inci_text(inci: list[str]) -> str:
    """Clean reference text from INCI_list (duplicates such as AQUA / WATER collapsed). The dataset's own
    raw_ingredients_list is sometimes noisy OCR, so the headline CER/WER uses this cleaned reference
    (the raw-text variant is still reported as CER_raw / WER_raw)."""
    seen, out = set(), []
    for x in inci:
        k = canon(x)
        if k and k not in seen:
            seen.add(k)
            out.append(x.lower())
    return ", ".join(out)


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def boot_ci(rows: list[dict], stat, n: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    """(point, lo, hi) of stat(rows) with a percentile bootstrap over rows (images)."""
    if not rows:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    point = stat(rows)
    vals = sorted(stat([rows[rng.randrange(len(rows))] for _ in rows]) for _ in range(n))
    return point, vals[int(0.025 * n)], vals[int(0.975 * n) - 1]


def micro(key: str, which: int):
    def f(rows):
        tp, fp, fn = (sum(r[key][i] for r in rows) for i in range(3))
        return prf(tp, fp, fn)[which]
    return f


def mean_of(key: str):
    return lambda rows: statistics.fmean(r[key] for r in rows)


def fmt(t: tuple[float, float, float], pct: bool = True) -> str:
    k = 100 if pct else 1
    return f"{t[0] * k:.1f} [{t[1] * k:.1f}, {t[2] * k:.1f}]" if pct else f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


# ───────────────────────────── scoring one sample ─────────────────────────────
def score_text(text: str, gold_inci: list[str], pollutants: dict[str, str]) -> dict:
    ings = nlp.analyze_text(text)
    pred = {canon(i.name) for i in ings if canon(i.name)}
    gold = gold_set(gold_inci)
    ident = (len(pred & gold), len(pred - gold), len(gold - pred))
    out = {"n_gold": len(gold), "n_pred": len(pred), "ident": ident,
           "exact": int(pred == gold), "coverage": (sum(i.matched for i in ings) / len(ings)) if ings else 0.0,
           "missed": sorted(gold - pred)[:12], "extra": sorted(pred - gold)[:12]}
    if pollutants:
        gold_flag = {g for g in gold if g in pollutants or any(canon(k) == g for k in pollutants)}
        pred_flag = {canon(i.name) for i in ings if i.risk in ("Medium", "High")}
        out["flag"] = (len(pred_flag & gold_flag), len(pred_flag - gold_flag), len(gold_flag - pred_flag))
        out["n_gold_flag"] = len(gold_flag)
    return out


# ───────────────────────────── OCR runners ─────────────────────────────
def degrade(data: bytes, kind: str) -> bytes:
    from PIL import Image, ImageEnhance, ImageFilter
    im = Image.open(io.BytesIO(data)).convert("RGB")
    if kind == "blur":
        im = im.filter(ImageFilter.GaussianBlur(2.0))
    elif kind == "lowres":
        w, h = im.size
        im = im.resize((max(1, w // 3), max(1, h // 3))).resize((w, h))
    elif kind == "dark":
        im = ImageEnhance.Brightness(im).enhance(0.45)
    elif kind != "jpeg":
        raise SystemExit(f"unknown --degrade {kind}")
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=15 if kind == "jpeg" else 90)
    return buf.getvalue()


def ocr_single(route, image_bytes: bytes, timeout: float) -> tuple[str, float, str]:
    """Run exactly one route (no fallback). Returns (text, seconds, error)."""
    t0 = time.time()
    try:
        data_url = ocr.prepare_image(image_bytes)
        messages = [{"role": "user", "content": [{"type": "text", "text": ocr.OCR_PROMPT},
                                                  {"type": "image_url", "image_url": {"url": data_url}}]}]
        raw = llm.complete(route.base, route.key, route.model, messages, temperature=0.0,
                           max_tokens=1500, timeout=timeout)
        text = ocr.clean_ocr_output(raw)
        return text, time.time() - t0, "" if text else "empty"
    except Exception as e:  # network / auth / rate limit - recorded, never fatal
        return "", time.time() - t0, f"{type(e).__name__}: {e}"[:160]


def get_ocr(label: str, sample: dict, settings, route, cache_dir: Path, deg: str | None, pause: float) -> dict:
    cache = cache_dir / re.sub(r"[^A-Za-z0-9]+", "_", label) / f"{sample['id']}.json"
    if cache.is_file():
        return json.loads(cache.read_text(encoding="utf-8"))
    data = sample["image"].read_bytes()
    if deg:
        data = degrade(data, deg)
    if route is None:  # full fallback chain exactly as the app runs it
        t0 = time.time()
        r = ocr.run_ocr(data, settings)
        res = {"text": r.text, "seconds": time.time() - t0, "error": "" if r.ok else "chain failed",
               "provider": r.provider}
    else:
        text, secs, err = ocr_single(route, data, settings.ocr_timeout)
        res = {"text": text, "seconds": secs, "error": err, "provider": route.label}
    if res["text"]:  # never cache failures, so a re-run retries them (e.g. after fixing a key or model id)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(res), encoding="utf-8")
    time.sleep(pause)
    return res


# ───────────────────────────── reporting ─────────────────────────────
def summarise(name: str, rows: list[dict], with_ocr: bool) -> dict:
    s = {"config": name, "n_images": len(rows)}
    for key, lab in (("ident", "ident"), ("flag", "flag")):
        sub = [r for r in rows if key in r]
        if not sub:
            continue
        for i, m in enumerate(("P", "R", "F1")):
            s[f"{lab}_{m}"] = fmt(boot_ci(sub, micro(key, i)))
    s["exact_match_%"] = fmt(boot_ci(rows, mean_of("exact")))
    s["kb_coverage_%"] = fmt(boot_ci(rows, mean_of("coverage")))
    if with_ocr:
        ok = [r for r in rows if r["ocr_ok"]]
        s["ocr_success_%"] = fmt(boot_ci(rows, mean_of("ocr_ok")))
        if ok:
            s["CER_%"] = fmt(boot_ci(ok, mean_of("cer")))
            s["WER_%"] = fmt(boot_ci(ok, mean_of("wer")))
            s["CER_raw_%"] = fmt(boot_ci(ok, mean_of("cer_raw")))
            s["latency_s_median"] = f"{statistics.median(r['seconds'] for r in ok):.1f}"
    return s


def write_outputs(out: Path, summaries: list[dict], per_sample: dict[str, list[dict]], tag: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"summary_{tag}.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    cols = list(dict.fromkeys(k for s in summaries for k in s))
    with open(out / f"summary_{tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(summaries)
    for name, rows in per_sample.items():
        flat = [{k: (json.dumps(v) if isinstance(v, (list, tuple)) else v) for k, v in r.items()} for r in rows]
        if flat:
            with open(out / f"per_image_{tag}_{re.sub(r'[^A-Za-z0-9]+', '_', name)}.csv", "w", newline="",
                      encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(flat[0]))
                w.writeheader()
                w.writerows(flat)
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(s.get(c, "")) for c in cols) + " |" for s in summaries]
    (out / f"summary_{tag}.md").write_text("\n".join(md), encoding="utf-8")
    tex = ["\\begin{tabular}{l" + "c" * (len(cols) - 1) + "}", "\\hline",
           " & ".join(c.replace("_", "\\_").replace("%", "\\%") for c in cols) + " \\\\ \\hline"]
    tex += [" & ".join(str(s.get(c, "")).replace("%", "\\%") for c in cols) + " \\\\" for s in summaries]
    tex += ["\\hline", "\\end{tabular}"]
    (out / f"summary_{tag}.tex").write_text("\n".join(tex), encoding="utf-8")
    print("\n".join(md))
    print(f"\nSaved to {out}/  (summary_{tag}.md/.csv/.tex/.json + per_image_*.csv)")


def paired_wilcoxon(a: list[dict], b: list[dict], key: str = "cer") -> str:
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return "scipy not installed - skipping significance test"
    bm = {r["id"]: r for r in b if r["ocr_ok"]}
    pairs = [(r[key], bm[r["id"]][key]) for r in a if r["ocr_ok"] and r["id"] in bm]
    if len(pairs) < 10 or all(x == y for x, y in pairs):
        return "too few paired images for a significance test"
    stat, p = wilcoxon([x for x, _ in pairs], [y for _, y in pairs])
    return f"Wilcoxon signed-rank on {len(pairs)} paired images: p = {p:.4g}"


# ───────────────────────────── main ─────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--mode", choices=["text", "ocr"], default="text")
    ap.add_argument("--n", type=int, default=150, help="images to sample (fixed seed)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pollutants", type=Path, default=ROOT / "evaluation" / "data" / "pollutant_mapping.yml",
                    help="ToxFox data/inci/pollutant_mapping.yml (optional; enables the concern-flag test)")
    ap.add_argument("--routes", default="all", help="ocr mode: 'all' or comma list of substrings (dots,paddle,muse)")
    ap.add_argument("--no-chain", action="store_true", help="ocr mode: skip the full fallback-chain run")
    ap.add_argument("--degrade", choices=["blur", "lowres", "jpeg", "dark"], help="ocr mode: robustness test")
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between API calls (free-tier rate limits)")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    a = ap.parse_args()

    samples = load_dataset(a.data)
    if a.mode == "ocr":
        samples = [s for s in samples if s["image"]]
    if not samples:
        raise SystemExit("No usable samples found. DATA_DIR must contain the .yaml files (and images for --mode ocr).")
    random.Random(a.seed).shuffle(samples)
    samples = samples[: a.n]
    pollutants = load_pollutants(a.pollutants)
    print(f"{len(samples)} images · pollutant list: {'loaded' if pollutants else 'not found (flag test skipped)'}")

    if a.mode == "text":
        rows = []
        for s in samples:
            r = score_text(s["gold_text"], s["gold_inci"], pollutants)
            r.update(id=s["id"], product=s["product"])
            rows.append(r)
        write_outputs(a.out, [summarise("gold text -> COSMOS NLP", rows, False)], {"nlp": rows}, "text")
        write_manifest(a.out, {"mode": "text", "n_images": len(samples), "seed": a.seed})
        return

    settings = get_settings()
    if not settings.ocr_routes:
        raise SystemExit("No OCR route available - put OPENROUTER_API_KEY / NVIDIA_API_KEY in .env first.")
    routes = [(r.label, r) for r in settings.ocr_routes]
    if a.routes != "all":
        wanted = [w.strip().lower() for w in a.routes.split(",")]
        routes = [(l, r) for l, r in routes if any(w in l.lower() for w in wanted)]
    configs = routes + ([] if a.no_chain else [("Full fallback chain", None)])
    tag = "ocr" + (f"_{a.degrade}" if a.degrade else "")
    cache_dir = a.out / "ocr_cache" / (a.degrade or "clean")
    summaries, per_sample = [], {}
    for label, route in configs:
        print(f"\n== {label} ==")
        rows = []
        for k, s in enumerate(samples, 1):
            res = get_ocr(label, s, settings, route, cache_dir, a.degrade, a.pause)
            seg = nlp.extract_segment(res["text"]) if res["text"] else ""
            r = score_text(res["text"], s["gold_inci"], pollutants) if res["text"] else score_text("", s["gold_inci"], pollutants)
            cer, wer = cer_wer(seg, s["gold_text"]) if res["text"] else (1.0, 1.0)
            cer_i, wer_i = cer_wer(strip_brackets(seg), inci_text(s["gold_inci"])) if res["text"] else (1.0, 1.0)
            r.update(id=s["id"], product=s["product"], ocr_ok=int(bool(res["text"])), cer=cer_i, wer=wer_i, cer_raw=cer, wer_raw=wer,
                     seconds=round(res["seconds"], 2), error=res["error"], provider=res.get("provider", ""))
            rows.append(r)
            if k % 10 == 0:
                print(f"  {k}/{len(samples)}")
        summaries.append(summarise(label, rows, True))
        per_sample[label] = rows
    labels = list(per_sample)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            print(f"{labels[i]} vs {labels[j]}: {paired_wilcoxon(per_sample[labels[i]], per_sample[labels[j]])}")
    write_outputs(a.out, summaries, per_sample, tag)
    write_manifest(a.out, {"mode": tag, "n_images": len(samples), "seed": a.seed, "degrade": a.degrade})


if __name__ == "__main__":
    main()
