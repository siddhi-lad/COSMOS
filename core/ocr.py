"""Stage 2 – cloud, multi-stage OCR with fallback:
Dots3 (OpenRouter) → PaddleOCR-VL (cloud) → Muse Glimmer (OpenRouter).
Nothing runs locally; the image is downscaled first to keep upload time low."""
from __future__ import annotations

import base64
import io
import json
import re
import time
from dataclasses import dataclass, field

from PIL import Image, ImageOps

from . import llm
from .config import Settings

OCR_PROMPT = (
    "Transcribe ALL text printed on this cosmetic product label exactly as written. "
    "Pay special attention to the INGREDIENTS list and keep ingredients comma separated. "
    "Return plain text only – no commentary, no markdown."
)
MAX_SIDE = 1600


@dataclass
class OCRResult:
    text: str = ""
    provider: str = ""
    model: str = ""
    latency: float = 0.0
    attempts: list[dict] = field(default_factory=list)  # [{provider, ok, seconds, error}]

    @property
    def ok(self) -> bool:
        return bool(self.text)


def prepare_image(data: bytes) -> str:
    """Return a base64 data-URL of a downscaled JPEG (faster upload, fewer timeouts)."""
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB", "L"):
        bg = Image.new("RGB", img.size, "white")
        bg.paste(img.convert("RGBA"), mask=img.convert("RGBA").split()[-1])
        img = bg
    else:
        img = img.convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def clean_ocr_output(text: str) -> str:
    """Models sometimes wrap output in code fences or JSON layout blocks."""
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    if text[:1] in "[{":
        try:
            obj = json.loads(text)
            pieces: list[str] = []

            def walk(o):
                if isinstance(o, dict):
                    if isinstance(o.get("text"), str):
                        pieces.append(o["text"])
                    for v in o.values():
                        if isinstance(v, (list, dict)):
                            walk(v)
                elif isinstance(o, list):
                    for v in o:
                        walk(v)

            walk(obj)
            if pieces:
                text = "\n".join(pieces)
        except ValueError:
            pass
    text = re.sub(r"<[^>]+>", " ", text)  # stray html tags
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _good_enough(text: str) -> bool:
    return len(re.findall(r"[A-Za-z]{3,}", text)) >= 3


def _providers(s: Settings) -> list[tuple[str, str, str, str]]:
    return [(r.label, r.base, r.key, r.model) for r in s.ocr_routes]


def run_ocr(image_bytes: bytes, settings: Settings) -> OCRResult:
    providers = _providers(settings)
    result = OCRResult()
    if not providers:
        result.attempts.append({"provider": "-", "ok": False, "seconds": 0,
                                "error": "No OCR provider available – add OPENROUTER_API_KEY (or NVIDIA_API_KEY) to .env"})
        return result
    try:
        data_url = prepare_image(image_bytes)
    except Exception as e:
        result.attempts.append({"provider": "image", "ok": False, "seconds": 0, "error": f"Unreadable image: {e}"})
        return result
    messages = [{"role": "user", "content": [
        {"type": "text", "text": OCR_PROMPT},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]}]
    t_all = time.time()
    for label, base, key, model in providers:
        t0 = time.time()
        try:
            raw = llm.complete(base, key, model, messages, temperature=0.0, max_tokens=1500,
                               timeout=settings.ocr_timeout)
            text = clean_ocr_output(raw)
            if not _good_enough(text):
                raise llm.LLMError("no readable text returned")
            result.attempts.append({"provider": label, "ok": True, "seconds": round(time.time() - t0, 1), "error": ""})
            result.text, result.provider, result.model = text, label, model
            result.latency = round(time.time() - t_all, 1)
            return result
        except llm.AuthError as e:
            result.attempts.append({"provider": label, "ok": False, "seconds": round(time.time() - t0, 1), "error": str(e)})
        except llm.LLMError as e:
            result.attempts.append({"provider": label, "ok": False, "seconds": round(time.time() - t0, 1), "error": str(e)})
    result.latency = round(time.time() - t_all, 1)
    return result
