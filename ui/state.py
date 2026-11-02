"""Session state helpers. Everything lives in st.session_state (current session only)."""
from __future__ import annotations

from datetime import datetime

import streamlit as st

from core.pipeline import ScanResult


def init() -> None:
    ss = st.session_state
    ss.setdefault("scans", [])          # list[ScanResult]
    ss.setdefault("current", None)      # scan id
    ss.setdefault("events", [])         # [{time, kind, text, scan_id}]
    ss.setdefault("chat", [])           # [{role, content, meta}]
    ss.setdefault("profile", [])
    ss.setdefault("depth", 4)
    ss.setdefault("extra_evidence", {})  # ingredient name -> processed items (Evidence Explorer lookups)
    ss.setdefault("stream_for", None)   # scan id awaiting summary streaming
    ss.setdefault("queued_q", None)


def current_scan() -> ScanResult | None:
    for s in st.session_state.scans:
        if s.id == st.session_state.current:
            return s
    return None


def add_scan(scan: ScanResult) -> None:
    st.session_state.scans.append(scan)
    st.session_state.current = scan.id
    log("Scan", f"{scan.label} – {len(scan.ingredients)} ingredients", scan.id)


def log(kind: str, text: str, scan_id: str | None = None) -> None:
    st.session_state.events.append({"time": datetime.now(), "kind": kind, "text": text, "scan_id": scan_id})


def clear() -> None:
    for k in ("scans", "events", "chat", "extra_evidence"):
        st.session_state[k] = [] if k in ("scans", "events", "chat") else {}
    st.session_state.current = None
    st.session_state.stream_for = None
    st.session_state.queued_q = None
