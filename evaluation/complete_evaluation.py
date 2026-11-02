"""Fills the evaluation gaps listed in the COSMOS paper. Put this file in cosmos/evaluation/ and run from the cosmos folder.

  python evaluation/complete_evaluation.py robustness --data data_tox --n 30     # redo blur/lowres/dark/jpeg with breaker reset + 429 retry
  python evaluation/complete_evaluation.py all --data data_tox --n 59            # Dots3 only: robustness + ablation + KB gaps (recommended)
  python evaluation/complete_evaluation.py ling                                  # shows why Ling Sante returns no verdicts
  python evaluation/complete_evaluation.py intents                               # writes 7-intent question file (adds general_cosmetic)
Then: two humans fill annotator1/annotator2, rater1/rater2, and the answer columns; run
  python evaluation/annotation_tools.py score
Outputs go to results/complete_*.json|csv. Nothing here fabricates data: every call is logged with status."""
import argparse, csv, json, random, statistics, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
import toxfox_eval as tx
from core import http, llm, ocr, nlp
from core.config import get_settings
from stats import boot_ci, fmt_ci, prf, write_manifest
OUT = ROOT / "results"


def call(route, img, timeout, retries=3):
    """One OCR call with breaker reset and exponential backoff on 429. Returns (text, secs, status)."""
    for k in range(retries + 1):
        http.breaker.reset()
        t0 = time.time()
        text, secs, err = tx.ocr_single(route, img, timeout)
        if text:
            return text, time.time() - t0, "ok"
        if "429" in err and k < retries:
            time.sleep(20 * 2 ** k); continue
        return "", time.time() - t0, err or "empty"


def load(a):
    s = [x for x in tx.load_dataset(a.data) if x["image"]]
    random.Random(a.seed).shuffle(s)
    return s[: a.n]


def save_partial(summaries, current_condition, current_rows, levels):
    """Write progress after each image so an interrupted run retains completed work."""
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "completed_conditions": summaries,
        "current_condition": current_condition,
        "current_rows": current_rows,
        "planned_conditions": levels,
        "updated_at_epoch": time.time(),
    }
    (OUT / "complete_robustness_progress.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )


def robustness(a):
    st = get_settings(); route = [r for r in st.ocr_routes if "dots" in r.label.lower()][0]
    pol = tx.load_pollutants(ROOT / "evaluation" / "data" / "pollutant_mapping.yml")
    samples = load(a); summ = []
    OUT.mkdir(parents=True, exist_ok=True)
    total = len(samples)
    print(
        f"Starting robustness evaluation: {total} images x {len(a.levels)} conditions; "
        f"pause={a.pause}s; OCR timeout={st.ocr_timeout}s",
        flush=True,
    )

    for deg in a.levels:
        rows = []
        print(f"\\n=== Condition: {deg} ({total} images) ===", flush=True)
        for i, s in enumerate(samples, start=1):
            t_img = time.time()
            try:
                data = s["image"].read_bytes()
                data = tx.degrade(data, deg) if deg != "clean" else data
                text, secs, status = call(route, data, st.ocr_timeout)
                r = tx.score_text(text, s["gold_inci"], pol)
                cer, wer = tx.cer_wer(
                    tx.strip_brackets(nlp.extract_segment(text)),
                    tx.inci_text(s["gold_inci"])
                ) if text else (1.0, 1.0)
                r.update(
                    id=s["id"], ocr_ok=int(bool(text)), cer=cer, wer=wer,
                    seconds=secs, status=status
                )
            except KeyboardInterrupt:
                print("\\nInterrupted. Saving progress before exit.", flush=True)
                save_partial(summ, deg, rows, a.levels)
                raise
            except Exception as exc:
                r = {
                    "id": s.get("id", f"sample_{i}"),
                    "ocr_ok": 0,
                    "cer": 1.0,
                    "wer": 1.0,
                    "seconds": time.time() - t_img,
                    "status": f"exception:{type(exc).__name__}:{exc}",
                    "ident": [0, 0, 0],
                }

            rows.append(r)
            print(
                f"[{deg}] {i}/{total} | OCR={'OK' if r['ocr_ok'] else 'FAIL'} "
                f"| status={r['status']} | {r['seconds']:.1f}s",
                flush=True,
            )
            save_partial(summ, deg, rows, a.levels)
            if a.pause > 0:
                time.sleep(a.pause)

        ok = [r for r in rows if r["ocr_ok"]]
        f1 = lambda rs: prf(*(sum(r["ident"][i] for r in rs) for i in range(3)))[2]
        summ.append({"condition": deg, "n": len(rows), "success": f"{len(ok)}/{len(rows)}",
                     "F1_%": fmt_ci(boot_ci(rows, f1)),
                     "CER_%": fmt_ci(boot_ci(ok, lambda rs: statistics.fmean(r["cer"] for r in rs))) if ok else "n/a",
                     "errors": sorted({r["status"] for r in rows if r["status"] != "ok"})[:3]})
        print("Condition summary:", summ[-1], flush=True)
        (OUT / "complete_robustness.json").write_text(
            json.dumps(summ, indent=2), encoding="utf-8"
        )
        save_partial(summ, deg, rows, a.levels)
    write_manifest(OUT, {"complete_robustness": a.levels})


def run_all(a):
    """Dots3-only pipeline: robustness (clean+4 degradations, breaker reset, 429 backoff) + offline ablation/KB gaps."""
    import subprocess
    robustness(a)
    subprocess.run([sys.executable, str(ROOT / "evaluation" / "ablation.py"), "--data", str(a.data)], check=False)
    print("\nDone. Report whatever the numbers are: results/complete_robustness.json, summary_ablation.md, kb_gaps_*.csv")


def ling(a):
    from core import ling_sante
    st = get_settings()
    print("ling_enabled:", st.ling_enabled, "| has_openrouter:", st.has_openrouter, "| model:", st.ling_model)
    msg = [{"role": "system", "content": ling_sante.SYSTEM},
           {"role": "user", "content": "[1] Ingredient: niacinamide\nTitle: Niacinamide in dermatology\nAbstract: Improves barrier function."}]
    try:
        out = llm.complete(ling_sante.OPENROUTER_BASE, st.openrouter_key, st.ling_model, msg, temperature=0, max_tokens=80, timeout=30)
        print("raw reply:", out, "\nparsed:", ling_sante._parse(out))
    except Exception as e:
        print("Ling Sante call failed ->", type(e).__name__, e, "\nFix the model id in .env (LING_SANTE_MODEL) or the key.")


def intents(a):
    import annotation_tools as t
    t.QUESTIONS += "\n"  # keep generator logic; then guarantee general_cosmetic coverage
    t.make_intents(argparse.Namespace(n=49))  # 7 per group x 7 groups -> general_cosmetic included
    print("Now: two humans label independently WITHOUT seeing the regexes; then run annotation_tools.py score")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="c", required=True)
    for n, f in (("robustness", robustness), ("all", run_all)):
        p = sub.add_parser(n); p.add_argument("--data", required=True, type=Path); p.add_argument("--n", type=int, default=30)
        p.add_argument("--seed", type=int, default=42); p.add_argument("--pause", type=float, default=0.5, help="seconds to wait between images (default: 0.5)")
        p.add_argument("--levels", nargs="+", default=["clean", "blur", "lowres", "dark", "jpeg"]); p.set_defaults(fn=f)
    sub.add_parser("ling").set_defaults(fn=ling); sub.add_parser("intents").set_defaults(fn=intents)
    a = ap.parse_args(); a.fn(a)
