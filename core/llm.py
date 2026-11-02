"""OpenAI-compatible chat client used for every LLM / vision-OCR call.
Works with OpenRouter and any OpenAI-compatible OCR endpoint. Adds ordered model fallback, a circuit
breaker (failed models are skipped for a while) and streaming."""
from __future__ import annotations

import json
from typing import Iterator

from . import http
from .config import NVIDIA_BASE, OPENROUTER_BASE, Settings

_models_cache = http.TTLCache(ttl=600, max_items=4)


class LLMError(Exception):
    pass


class AuthError(LLMError):
    pass


def _headers(key: str) -> dict:
    h = {"Content-Type": "application/json", "HTTP-Referer": "http://localhost:8501", "X-Title": "COSMOS"}
    if key:  # some endpoints (e.g. a free PaddleOCR-VL one) need no key
        h["Authorization"] = f"Bearer {key}"
    return h


def _no_think(base: str) -> dict:
    """Ask reasoning models to answer directly (no chain of thought). Provider-specific parameters."""
    if base == NVIDIA_BASE:
        return {"chat_template_kwargs": {"enable_thinking": False}}
    if base == OPENROUTER_BASE:
        return {"reasoning": {"enabled": False}}
    return {}


def _post(base: str, key: str, payload: dict, timeout, stream: bool = False):
    """POST a chat completion with reasoning switched off; if the provider rejects the extra
    parameter (HTTP 400) transparently retry once with the plain payload."""
    url = f"{base}/chat/completions"
    extras = _no_think(base)
    if extras:
        r = http.session().post(url, headers=_headers(key), json={**payload, **extras}, timeout=timeout, stream=stream)
        if r.status_code != 400:
            return r
        r.close()
    return http.session().post(url, headers=_headers(key), json=payload, timeout=timeout, stream=stream)


def _classify_failure(status: int) -> float:
    """Cool-down seconds before we try a failing model again."""
    if status in (400, 404):  # unknown model / unsupported input
        return 600
    if status == 429:
        return 45
    return 60


def _extract_text(data: dict) -> str:
    try:
        msg = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return ""
    content = msg.get("content")
    if isinstance(content, list):  # some providers return parts
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return (content or "").strip()


def complete(
    base_url: str,
    key: str,
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.2,
    max_tokens: int = 900,
    timeout: float = 45,
) -> str:
    """One chat completion against one model. Raises LLMError on failure."""
    brk = f"{base_url}|{model}"
    if http.breaker.is_open(brk):
        raise LLMError(f"{model} temporarily skipped (recent failure)")
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    try:
        r = _post(base_url, key, payload, timeout)
    except Exception as e:  # timeout / connection
        http.breaker.trip(brk, 30)
        raise LLMError(f"{model}: {type(e).__name__}") from e
    if r.status_code in (401, 403):
        http.breaker.trip(brk, 300)  # don't keep hitting an endpoint that rejects us
        raise AuthError(f"{model}: authentication failed ({r.status_code}) – check your API key")
    if r.status_code != 200:
        http.breaker.trip(brk, _classify_failure(r.status_code))
        detail = ""
        try:
            detail = r.json().get("error", {}).get("message", "")[:160]
        except Exception:
            pass
        raise LLMError(f"{model}: HTTP {r.status_code} {detail}".strip())
    try:
        text = _extract_text(r.json())
    except ValueError as e:
        raise LLMError(f"{model}: invalid JSON response") from e
    if not text:
        http.breaker.trip(brk, 30)
        raise LLMError(f"{model}: empty response")
    return text


def chat(settings: Settings, messages: list[dict], *, temperature: float = 0.2, max_tokens: int = 900) -> tuple[str, str]:
    """Try each route in order (Nemotron → Muse Glimmer, each via OpenRouter then NVIDIA).
    Returns (text, route_label)."""
    routes = settings.llm_routes
    if not routes:
        raise AuthError("No LLM API key set (OPENROUTER_API_KEY or NVIDIA_API_KEY)")
    errors = []
    for r in routes:
        try:
            return complete(r.base, r.key, r.model, messages, temperature=temperature, max_tokens=max_tokens,
                            timeout=settings.llm_timeout), r.label
        except LLMError as e:  # includes AuthError: a bad key on one route must not block the others
            errors.append(f"{r.label}: {str(e)[:120]}")
    raise LLMError("All LLMs failed: " + " | ".join(errors))


def chat_stream(settings: Settings, messages: list[dict], meta: dict, *, temperature: float = 0.2,
                max_tokens: int = 900) -> Iterator[str]:
    """Stream tokens from the first route that responds. `meta['model']` is set to the
    route that answered. If streaming fails before the first token the next route is
    tried; if everything fails an LLMError is raised."""
    routes = settings.llm_routes
    if not routes:
        raise AuthError("No LLM API key set (OPENROUTER_API_KEY or NVIDIA_API_KEY)")
    errors = []
    for route in routes:
        brk = f"{route.base}|{route.model}"
        if http.breaker.is_open(brk):
            errors.append(f"{route.label}: skipped (recent failure)")
            continue
        payload = {"model": route.model, "messages": messages, "temperature": temperature,
                   "max_tokens": max_tokens, "stream": True}
        started = False
        try:
            with _post(route.base, route.key, payload, (8, settings.llm_timeout), stream=True) as r:
                if r.status_code in (401, 403):
                    http.breaker.trip(brk, 300)
                    errors.append(f"{route.label}: authentication failed")
                    continue
                if r.status_code != 200:
                    http.breaker.trip(brk, _classify_failure(r.status_code))
                    errors.append(f"{route.label}: HTTP {r.status_code}")
                    continue
                for raw in r.iter_lines(decode_unicode=True):
                    if not raw or raw.startswith(":") or not raw.startswith("data:"):
                        continue
                    chunk = raw[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        delta = json.loads(chunk)["choices"][0]["delta"].get("content") or ""
                    except (ValueError, KeyError, IndexError, TypeError):
                        continue
                    if delta:
                        if not started:
                            meta["model"] = route.label
                            started = True
                        yield delta
            if started:
                return
            http.breaker.trip(brk, 30)
            errors.append(f"{route.label}: empty stream")
        except Exception as e:
            if started:  # partial answer already shown – stop cleanly
                return
            http.breaker.trip(brk, 30)
            errors.append(f"{route.label}: {type(e).__name__}")
    raise LLMError("All LLMs failed: " + " | ".join(errors))


def list_models(base: str, key: str = "") -> set[str]:
    """Live catalogue of model ids for an OpenAI-compatible provider (cached 10 min)."""
    cached = _models_cache.get(base)
    if cached is not None:
        return cached
    r = http.session().get(f"{base}/models", headers={"Authorization": f"Bearer {key}"} if key else {}, timeout=15)
    r.raise_for_status()
    ids = {m.get("id", "") for m in r.json().get("data", [])}
    _models_cache.set(base, ids)
    return ids


def check_models(settings: Settings) -> list[dict]:
    """Report which configured model ids exist right now on each provider."""
    catalogues: dict[str, set[str] | None] = {}
    for base, key, on in ((OPENROUTER_BASE, settings.openrouter_key, settings.has_openrouter),
                          (NVIDIA_BASE, settings.nvidia_key, settings.has_nvidia)):
        if on:
            try:
                catalogues[base] = list_models(base, key)
            except Exception:
                catalogues[base] = None
    rows = []
    for role, base_label, base, model in (
        [("OCR 1 – Dots3", "OpenRouter", OPENROUTER_BASE, settings.dots_model),
         ("LLM 1 – Nemotron", "OpenRouter", OPENROUTER_BASE, settings.nemotron_model),
         ("LLM 1 – Nemotron", "NVIDIA", NVIDIA_BASE, settings.nemotron_model),
         ("LLM 2 / OCR 3 – Muse Glimmer", "OpenRouter", OPENROUTER_BASE, settings.muse_model),
         ("LLM 2 / OCR 3 – Muse Glimmer", "NVIDIA", NVIDIA_BASE, settings.muse_model),
         ("Evidence validation – Ling Sante", "OpenRouter", OPENROUTER_BASE, settings.ling_model)]):
        if base not in catalogues:
            continue
        cat = catalogues[base]
        status = "could not load catalogue" if cat is None else ("found" if model in cat else "NOT FOUND")
        rows.append({"Role": role, "Provider": base_label, "Model id": model, "Status": status})
    rows.insert(1, {"Role": "OCR 2 – PaddleOCR-VL", "Provider": "endpoint in .env", "Model id": settings.paddle_model,
                    "Status": "checked on first use"})
    return rows