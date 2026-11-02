"""COSMOS – Cosmetic Ingredient Intelligence.   Run:  streamlit run app.py"""
import os
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="COSMOS · Cosmetic Ingredient Intelligence", page_icon=":material/spa:", layout="wide")

from core import http, llm  # noqa: E402
from core.config import env_status_text, get_settings  # noqa: E402
from core.nlp import PROFILE_RULES  # noqa: E402
from ui import pages, state, theme  # noqa: E402

theme.inject()
state.init()
st.logo(str(Path(__file__).parent / "ui" / "assets" / "logo.svg"))

nav = {
    "scanner": st.Page(pages.scanner, title="Ingredient Scanner", icon=":material/document_scanner:", default=True, url_path="scanner"),
    "intelligence": st.Page(pages.intelligence, title="Ingredient Intelligence", icon=":material/science:", url_path="intelligence"),
    "explorer": st.Page(pages.explorer, title="Evidence Explorer", icon=":material/menu_book:", url_path="evidence"),
    "assistant": st.Page(pages.assistant, title="AI Assistant", icon=":material/chat_bubble:", url_path="assistant"),
    "history": st.Page(pages.history, title="History (Current Session)", icon=":material/history:", url_path="history"),
}
st.session_state["_pages"] = nav
current = st.navigation(list(nav.values()))


def _key_input(label: str, env: str) -> None:
    val = st.text_input(label, value=os.getenv(env, ""), type="password", key=f"k_{env}")
    if val != os.getenv(env, ""):
        os.environ[env] = val.strip()
        http.breaker.reset()  # a new key deserves a fresh try


with st.sidebar:
    st.divider()
    st.markdown("**Your skin profile**")
    st.session_state.profile = st.multiselect(
        "Flag ingredients relevant to:", list(PROFILE_RULES), default=st.session_state.profile,
        label_visibility="collapsed", placeholder="Choose any that apply")
    st.session_state.depth = st.slider("Evidence depth (ingredients researched per scan)", 0, 8,
                                       st.session_state.depth,
                                       help="Higher = more complete but slower. 0 skips live evidence at scan time.")
    with st.expander("Connections"):
        _key_input("OpenRouter API key", "OPENROUTER_API_KEY")
        _key_input("NVIDIA API key", "NVIDIA_API_KEY")
        _key_input("PaddleOCR-VL key (optional)", "PADDLE_OCR_API_KEY")
        _key_input("NCBI API key (optional)", "NCBI_API_KEY")
        s = get_settings()
        st.caption("Configuration loaded from environment variables.")
        ling_on = st.toggle("AI evidence validation (Ling Sante)", value=s.ling_enabled,
                            help="One extra batched call that flags irrelevant studies. Turn off for the fastest scans.")
        os.environ["LING_SANTE_ENABLED"] = "1" if ling_on else "0"
        st.caption(f"OpenRouter {'✓' if s.has_openrouter else '✗'} · NVIDIA {'✓' if s.has_nvidia else '✗'}"
                   f" · PaddleOCR-VL {'✓' if s.has_paddle else '✗'} · PubMed ✓ · PubChem ✓")
        if st.button("Check model ids", use_container_width=True):
            try:
                st.dataframe(llm.check_models(s), hide_index=True, use_container_width=True)
                http.breaker.reset()
            except Exception as e:
                st.error(f"Could not load model catalogues: {type(e).__name__}")
    st.caption("API-only · no local models")

current.run()