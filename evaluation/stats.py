"""Small, dependency-free statistics helpers shared by all evaluation scripts."""
from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import statistics
import sys
from datetime import datetime
from pathlib import Path


# ── classification / retrieval metrics ──────────────────────────────────────
def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def macro_f1(gold: list[str], pred: list[str]) -> tuple[float, dict]:
    labels = sorted(set(gold) | set(pred))
    per = {}
    for lab in labels:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred))
        fp = sum(g != lab and p == lab for g, p in zip(gold, pred))
        fn = sum(g == lab and p != lab for g, p in zip(gold, pred))
        per[lab] = prf(tp, fp, fn) + (sum(g == lab for g in gold),)
    return statistics.fmean(v[2] for v in per.values()) if per else 0.0, per


def precision_at_k(rels: list[int], k: int) -> float:
    top = rels[:k]
    return sum(top) / k if k else 0.0


def mrr(rels: list[int]) -> float:
    for i, r in enumerate(rels, 1):
        if r:
            return 1 / i
    return 0.0


def ndcg_at_k(rels: list[int], k: int) -> float:
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))
    ideal = sorted(rels, reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


# ── agreement ───────────────────────────────────────────────────────────────
def cohen_kappa(a: list, b: list, weights: str | None = None) -> float:
    """Cohen's kappa (unweighted, or 'linear' for ordinal scores)."""
    pairs = [(x, y) for x, y in zip(a, b) if x not in ("", None) and y not in ("", None)]
    if not pairs:
        return float("nan")
    cats = sorted({x for p in pairs for x in p}, key=lambda v: (str(type(v)), v))
    idx = {c: i for i, c in enumerate(cats)}
    k, n = len(cats), len(pairs)
    obs = [[0] * k for _ in range(k)]
    for x, y in pairs:
        obs[idx[x]][idx[y]] += 1
    row = [sum(r) for r in obs]
    col = [sum(obs[i][j] for i in range(k)) for j in range(k)]

    def w(i, j):
        if weights == "linear" and k > 1:
            return abs(i - j) / (k - 1)
        return 0.0 if i == j else 1.0

    do = sum(w(i, j) * obs[i][j] for i in range(k) for j in range(k)) / n
    de = sum(w(i, j) * row[i] * col[j] for i in range(k) for j in range(k)) / (n * n)
    return 1 - do / de if de else 1.0


# ── inference ───────────────────────────────────────────────────────────────
def boot_ci(rows: list, stat, n: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    if not rows:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    point = stat(rows)
    vals = sorted(stat([rows[rng.randrange(len(rows))] for _ in rows]) for _ in range(n))
    return point, vals[int(0.025 * n)], vals[max(0, int(0.975 * n) - 1)]


def mcnemar_exact(correct_a: list[int], correct_b: list[int]) -> float:
    """Exact two-sided McNemar p-value on paired correct/incorrect outcomes."""
    b = sum(x and not y for x, y in zip(correct_a, correct_b))
    c = sum(y and not x for x, y in zip(correct_a, correct_b))
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def wilcoxon_p(a: list[float], b: list[float]) -> float | None:
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None
    pairs = [(x, y) for x, y in zip(a, b) if x != y]
    if len(pairs) < 6:
        return None
    return float(wilcoxon([x for x, _ in pairs], [y for _, y in pairs]).pvalue)


def holm(pvals: list[float | None]) -> list[float | None]:
    """Holm-Bonferroni adjusted p-values (None entries are skipped)."""
    idx = [i for i, p in enumerate(pvals) if p is not None]
    order = sorted(idx, key=lambda i: pvals[i])
    adj: list[float | None] = [None] * len(pvals)
    m, running = len(order), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def fmt_ci(t: tuple[float, float, float], pct: bool = True, nd: int = 1) -> str:
    k = 100 if pct else 1
    return f"{t[0] * k:.{nd}f} [{t[1] * k:.{nd}f}, {t[2] * k:.{nd}f}]"


# ── reproducibility ─────────────────────────────────────────────────────────
def write_manifest(out: Path, extra: dict | None = None) -> None:
    """Freeze everything a reviewer needs to reproduce a run (never stores API keys)."""
    root = Path(__file__).resolve().parent.parent
    info = {"created": datetime.now().isoformat(timespec="seconds"), "python": sys.version.split()[0],
            "platform": platform.platform(), "argv": sys.argv[1:]}
    try:
        from core.config import get_settings
        s = get_settings()
        info["models"] = {"dots": s.dots_model, "paddle": s.paddle_model if s.has_paddle else None,
                          "nemotron": s.nemotron_model, "muse": s.muse_model, "ling_sante": s.ling_model,
                          "ling_enabled": s.ling_enabled}
        info["timeouts_s"] = {"ocr": s.ocr_timeout, "llm": s.llm_timeout, "evidence": s.evidence_timeout}
    except Exception as e:  # never block a run on the manifest
        info["models_error"] = type(e).__name__
    kb = root / "core" / "knowledge.py"
    if kb.is_file():
        info["knowledge_base_sha256"] = hashlib.sha256(kb.read_bytes()).hexdigest()[:16]
        try:
            from core.knowledge import KB
            info["knowledge_base_entries"] = len(KB)
        except Exception:
            pass
    for mod in ("streamlit", "requests", "pandas", "PIL", "yaml", "scipy"):
        try:
            info.setdefault("packages", {})[mod] = __import__(mod).__version__
        except Exception:
            pass
    info.update(extra or {})
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_manifest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
