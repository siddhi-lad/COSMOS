"""Central configuration. Reads environment variables / .env at call time so keys
pasted in the UI take effect immediately.

Variable names deliberately match the project's earlier .env
(OPENROUTER_API_KEY, NVIDIA_API_KEY, DOTS_MODEL, NEMOTRON_MODEL, MUSE_MODEL,
LING_SANTE_MODEL, REQUEST_TIMEOUT, MAX_UPLOAD_SIZE_MB) so that file can be reused as-is.
Variables from the old backend that are not used here (CORS_ORIGINS, PADDLEOCR_DEVICE …)
are simply ignored."""
from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent


def load_env() -> dict:
    """Load .env robustly and report what happened.
    - looks next to app.py first, then in the current working directory
    - copes with the encodings Windows editors produce (UTF-8, UTF-8 with BOM, UTF-16, cp1252)
    - OVERRIDES variables that already exist but are empty (a common silent cause of 'key ignored')"""
    tried = []
    for path in (ROOT / ".env", Path.cwd() / ".env"):
        tried.append(str(path))
        if not path.is_file():
            continue
        raw = path.read_bytes()
        for enc in ("utf-8-sig", "utf-16", "cp1252"):
            try:
                text = raw.decode(enc)
            except UnicodeDecodeError:
                continue
            values = {k: v for k, v in dotenv_values(stream=io.StringIO(text)).items() if k and v}
            if not values and enc != "cp1252":
                continue  # wrong encoding guess produces nothing useful – try the next one
            for k, v in values.items():
                os.environ[k] = v.strip()
            return {"found": True, "path": str(path), "encoding": enc, "count": len(values),
                    "vars": sorted(values)}
    return {"found": False, "tried": tried, "count": 0, "vars": []}


ENV_STATUS = load_env()


def env_status_text() -> str:
    if ENV_STATUS["found"]:
        keys = [
            k for k in ("OPENROUTER_API_KEY", "NVIDIA_API_KEY")
            if k in ENV_STATUS["vars"]
        ]
        return (
            f"Environment configuration loaded "
            f"({ENV_STATUS['count']} values; "
            f"providers configured: {', '.join(keys) or 'none'})"
        )

    return "No environment configuration found. Configure API keys below."

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
NVIDIA_BASE = "https://integrate.api.nvidia.com/v1"


def _env(name: str, default: str = "") -> str:
    # tolerate values written with surrounding quotes in .env
    return (os.getenv(name) or default).strip().strip('"').strip("'")


def _float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _flag(name: str, default: bool) -> bool:
    v = _env(name, "1" if default else "0").lower()
    return v not in ("0", "false", "no", "off", "")


@dataclass(frozen=True)
class Route:
    """One concrete way to call a model: provider endpoint + key + model id."""
    label: str
    base: str
    key: str
    model: str


@dataclass(frozen=True)
class Settings:
    openrouter_key: str
    nvidia_key: str
    paddle_key: str
    paddle_base: str
    ncbi_key: str
    ncbi_email: str
    dots_model: str
    paddle_model: str
    nemotron_model: str
    muse_model: str
    ling_model: str
    ocr_timeout: float
    llm_timeout: float
    evidence_timeout: float
    max_upload_mb: int
    llm_intent_fallback: bool
    ling_enabled: bool

    # ── availability ──
    @property
    def has_openrouter(self) -> bool:
        return bool(self.openrouter_key)

    @property
    def has_nvidia(self) -> bool:
        return bool(self.nvidia_key)

    @property
    def has_paddle(self) -> bool:
        """PaddleOCR-VL endpoint is used whenever a URL is configured; the key is optional."""
        return bool(self.paddle_base)

    @property
    def has_llm(self) -> bool:
        return self.has_openrouter or self.has_nvidia

    # ── ordered fallback chains (abstract: Nemotron primary → Muse Glimmer fallback) ──
    def _routes(self, model: str, name: str) -> list[Route]:
        out = []
        if self.has_openrouter:
            out.append(Route(f"{name} via OpenRouter", OPENROUTER_BASE, self.openrouter_key, model))
        if self.has_nvidia:
            out.append(Route(f"{name} via NVIDIA", NVIDIA_BASE, self.nvidia_key, model))
        return out

    @property
    def llm_routes(self) -> list[Route]:
        return self._routes(self.nemotron_model, "Nemotron") + self._routes(self.muse_model, "Muse Glimmer")

    @property
    def ocr_routes(self) -> list[Route]:
        routes = [r for r in self._routes(self.dots_model, "Dots3") if r.base == OPENROUTER_BASE]
        if self.has_paddle:
            routes.append(Route("PaddleOCR-VL (cloud)", self.paddle_base, self.paddle_key, self.paddle_model))
        routes += self._routes(self.muse_model, "Muse Glimmer")
        return routes


def get_settings() -> Settings:
    req = _float("REQUEST_TIMEOUT", 90)
    return Settings(
        openrouter_key=_env("OPENROUTER_API_KEY"),
        nvidia_key=_env("NVIDIA_API_KEY"),
        paddle_key=_env("PADDLE_OCR_API_KEY"),
        paddle_base=_env("PADDLE_OCR_BASE_URL", "").rstrip("/"),
        ncbi_key=_env("NCBI_API_KEY"),
        ncbi_email=_env("NCBI_EMAIL"),
        dots_model=_env("DOTS_MODEL", "dots-studio/dots-3-note-preview:free"),
        paddle_model=_env("PADDLE_OCR_MODEL", "PaddlePaddle/PaddleOCR-VL"),
        nemotron_model=_env("NEMOTRON_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b"),
        muse_model=_env("MUSE_MODEL", "nvidia/muse-glimmer-30b-a3b"),
        ling_model=_env("LING_SANTE_MODEL", "inclusionai/ling-3.0-flash-sante:free"),
        # free models can be slow: allow up to REQUEST_TIMEOUT but cap OCR/LLM individually
        ocr_timeout=_float("OCR_TIMEOUT", min(req, 90)),
        llm_timeout=_float("LLM_TIMEOUT", min(req, 90)),
        evidence_timeout=_float("EVIDENCE_TIMEOUT", 12),
        max_upload_mb=int(_float("MAX_UPLOAD_SIZE_MB", 10)),
        llm_intent_fallback=_flag("LLM_INTENT_FALLBACK", True),
        ling_enabled=_flag("LING_SANTE_ENABLED", True),
    )


logging.basicConfig(level=getattr(logging, _env("LOG_LEVEL", "INFO").upper(), logging.INFO))
