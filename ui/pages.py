"""The five application pages (stage 8). Every control here performs a real action."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core import evidence, intent, ling_sante, nlp, pipeline, processing, synthesis
from core.config import get_settings
from core.knowledge import KB
from core.nlp import RISK_ORDER, Ingredient
from . import state, theme

SOURCES = ["PubMed", "PubChem", "FDA", "CosIng"]


# ───────────────────────────── shared pieces ─────────────────────────────────
def _need_scan() -> pipeline.ScanResult | None:
    scan = state.current_scan()
    if not scan or not scan.ingredients:
        theme.empty("No ingredients analysed yet.<br>Go to <b>Ingredient Scanner</b> and upload a label or enter a list.")
        return None
    return scan


def _flags_text(ing: Ingredient) -> str:
    return "; ".join(ing.flags(st.session_state.profile))


def _scan_evidence(scan, name: str) -> list[dict]:
    return scan.evidence.get(name) or st.session_state.extra_evidence.get(name, [])


def _render_item(it: dict, key: str) -> None:
    cur = theme.pill("curated pointer", "cur") if it.get("curated") else ""
    ok = "" if it.get("validated", True) else theme.pill("needs review", "Medium")
    yr = f" · {it['year']}" if it.get("year") else ""
    venue = f" · {it['venue']}" if it.get("venue") else ""
    snippet = it["snippet"][:360] + ("…" if len(it["snippet"]) > 360 else "")
    st.markdown(
        f'<div class="card"><div>{theme.pill(it["source"], "src")}{cur}{ok}'
        f'<span class="meta">{it["id"]}{yr}{venue}</span></div><h4 style="margin-top:.45rem">{it["title"]}</h4>'
        f'<p>{snippet}</p></div>', unsafe_allow_html=True)
    st.link_button("Open source ↗", it["url"], key=key)


def _fetch_one(name: str) -> list[dict]:
    """On-demand evidence for any ingredient name (used by Explorer and Assistant)."""
    settings = get_settings()
    ing = nlp.identify(name, 0)
    res = evidence.fetch_ingredient(ing, settings)
    e = KB.get(ing.name)
    items = processing.process(ing.name, res["items"], list(e.aliases) if e else None)
    ling_sante.validate({ing.name: items}, settings)
    st.session_state.extra_evidence[ing.name] = items
    return items


# ───────────────────────────── 1. Scanner ────────────────────────────────────
def scanner() -> None:
    theme.hero("Ingredient Scanner", "From product label to evidence-backed answers",
               "Upload a photo of the ingredient list, or type it in. COSMOS reads it, identifies each ingredient, "
               "and checks it against scientific and regulatory sources.")
    settings = get_settings()
    scan_no = len(st.session_state.scans) + 1
    tab_img, tab_txt = st.tabs(["Upload label image", "Enter ingredients manually"])
    run_image = run_text = False
    with tab_img:
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            up = st.file_uploader("Product label (JPEG, PNG or WebP)", type=["jpg", "jpeg", "png", "webp"])
            too_big = up is not None and up.size > settings.max_upload_mb * 1024 * 1024
            if too_big:
                st.error(f"Image is larger than {settings.max_upload_mb} MB – please upload a smaller photo.")
            run_image = st.button("Extract ingredients", type="primary", disabled=up is None or too_big, key="go_img")
            if not settings.has_openrouter:
                st.info("Add OPENROUTER_API_KEY (.env or sidebar → Connections) for the Dots3 and Muse Glimmer OCR "
                        "stages. PaddleOCR-VL is tried without a key. Manual entry always works.")
        with c2:
            if up is not None:
                st.image(up, caption="Preview", use_container_width=True)
    with tab_txt:
        txt = st.text_area("Ingredient list", height=150, key="manual_txt",
                           placeholder="Aqua, Glycerin, Niacinamide, Phenoxyethanol, Carbomer, Parfum …")
        run_text = st.button("Analyze ingredients", type="primary", disabled=not txt.strip(), key="go_txt")

    if run_image or run_text:
        label = f"Scan {scan_no} – " + (up.name.rsplit(".", 1)[0][:24] if run_image and up else "Manual list")
        with st.status("Analysing…", expanded=True) as status:
            scan = pipeline.run_scan(
                label, settings, st.session_state.depth,
                image_bytes=up.getvalue() if run_image else None,
                manual_text=txt if run_text else "",
                progress=lambda stage, msg: st.write(msg))
            if scan.ingredients:
                status.update(label=f"Found {len(scan.ingredients)} ingredients", state="complete", expanded=False)
            else:
                status.update(label="No ingredients found", state="error", expanded=True)
        state.add_scan(scan) if scan.ingredients else None
        if scan.ingredients:
            st.session_state.stream_for = scan.id
            st.rerun()
        _explain_failure(scan)
        return

    scan = state.current_scan()
    if scan:
        _render_scan(scan, settings)
    else:
        theme.empty("Your results will appear here.")


def _explain_failure(scan: pipeline.ScanResult) -> None:
    if scan.ocr and not scan.ocr.ok:
        st.error("The label could not be read by any OCR provider.")
        st.dataframe(pd.DataFrame(scan.ocr.attempts).rename(columns=str.title), hide_index=True, use_container_width=True)
        st.info("Tips: use a sharper, well-lit photo of the ingredient list, check your API keys and model ids "
                "(sidebar → Connections → Check model ids), or type the list in the manual tab.")
    else:
        st.warning("Text was read, but no ingredient list could be separated from it. "
                   "Try editing the text or use the manual tab.")


def _render_scan(scan: pipeline.ScanResult, settings) -> None:
    summ = nlp.summarize(scan.ingredients)
    st.markdown(f"#### {scan.label}")
    m = st.columns(4)
    m[0].metric("Ingredients", summ["total"])
    m[1].metric("Recognised", summ["recognised"])
    m[2].metric("Medium concern", summ["medium"])
    m[3].metric("High concern", summ["high"])

    with st.expander("Pipeline trace – what ran and how long it took"):
        if scan.ocr:
            used = scan.ocr.provider or "none succeeded"
            st.markdown(f"**OCR provider:** {used}")
            st.dataframe(pd.DataFrame(scan.ocr.attempts).rename(columns=str.title), hide_index=True,
                         use_container_width=True)
        t = scan.timings
        st.caption("Timings: " + " · ".join(f"{k} {v}s" for k, v in t.items()) if t else "")
        if scan.evidence_errors:
            st.caption("Source notes: " + "; ".join(sorted(set(scan.evidence_errors))))
    with st.expander("Extracted text (edit to fix OCR mistakes, then re-analyse)"):
        edited = st.text_area("Text", scan.raw_text, height=130, key=f"edit_{scan.id}", label_visibility="collapsed")
        if st.button("Re-analyse edited text", key=f"re_{scan.id}"):
            with st.spinner("Re-analysing…"):
                new = pipeline.run_scan(scan.label + " (edited)", settings, st.session_state.depth, manual_text=edited)
            if new.ingredients:
                state.add_scan(new)
                st.session_state.stream_for = new.id
                st.rerun()
            st.warning("No ingredients found in the edited text.")

    st.markdown("### Summary")
    if st.session_state.stream_for == scan.id and not scan.summary:
        _stream_summary(scan, settings)
    elif scan.summary:
        st.markdown(scan.summary)
        st.caption(f"Generated by: {scan.summary_model}")
    else:
        if st.button("Generate summary", type="primary", key=f"sum_{scan.id}"):
            st.session_state.stream_for = scan.id
            st.rerun()


def _stream_summary(scan: pipeline.ScanResult, settings) -> None:
    profile = st.session_state.profile
    messages = pipeline.prepare_summary(scan, profile)
    meta: dict = {}
    box = st.empty()
    text = ""
    if settings.has_llm:
        with box.container():
            with st.spinner("Writing summary…"):
                text = st.write_stream(synthesis.stream_answer(settings, messages, meta)) or ""
    else:
        meta["error"] = "no LLM API key"
    box.empty()
    final = pipeline.finalize_summary(scan, text, meta.get("model", ""), profile, meta.get("error", ""))
    st.session_state.stream_for = None
    st.markdown(final)
    st.caption(f"Generated by: {scan.summary_model}")


# ───────────────────────────── 2. Intelligence ───────────────────────────────
def intelligence() -> None:
    theme.hero("Ingredient Intelligence", "Function, risk and evidence – ingredient by ingredient",
               "Functions and risk levels come from a curated knowledge base; evidence counts come from live sources.")
    scan = _need_scan()
    if not scan:
        return
    rows = []
    for i in scan.ingredients:
        ev = _scan_evidence(scan, i.name)
        counts = processing.count_sources(ev)
        rows.append({"#": i.position, "Ingredient": i.name, "Function": ", ".join(i.functions) or "—", "Risk": i.risk,
                     "Profile flags": _flags_text(i) or "—",
                     "Studies": counts.get("PubMed", 0) if ev else None})
    df = pd.DataFrame(rows)
    f1, f2 = st.columns([2, 1])
    risks = f1.multiselect("Filter by risk", ["High", "Medium", "Low", "Unknown"], default=["High", "Medium", "Low", "Unknown"])
    query = f2.text_input("Search ingredient", placeholder="e.g. glycerin")
    view = df[df["Risk"].isin(risks)]
    if query.strip():
        view = view[view["Ingredient"].str.contains(query.strip(), case=False)]
    colour = {"Low": "color:#2E7D5B;font-weight:600", "Medium": "color:#B7791F;font-weight:600",
              "High": "color:#B83A4B;font-weight:600", "Unknown": "color:#7A857F"}
    styler = view.style
    styler = (styler.map if hasattr(styler, "map") else styler.applymap)(lambda v: colour.get(v, ""), subset=["Risk"])
    st.dataframe(styler, hide_index=True, use_container_width=True,
                 column_config={"Studies": st.column_config.NumberColumn(
                     "PubMed studies", help="Blank = evidence not fetched yet for this ingredient")})
    b1, b2, _ = st.columns([1.3, 1, 2])
    with b1:
        missing = [i for i in scan.ingredients if i.name not in scan.evidence and i.name not in st.session_state.extra_evidence]
        if st.button(f"Fetch evidence for {len(missing)} remaining", disabled=not missing):
            bar = st.progress(0.0, text="Retrieving evidence…")
            pipeline.add_evidence(scan, get_settings(), len(missing), only=missing,
                                  on_done=lambda d, t, n: bar.progress(d / t, text=f"{n} ({d}/{t})"))
            state.log("Evidence", f"Fetched evidence for {len(missing)} ingredients", scan.id)
            st.rerun()
    with b2:
        st.download_button("Download CSV", df.to_csv(index=False).encode(), file_name=f"{scan.label}.csv", mime="text/csv")

    st.markdown("### Ingredient detail")
    pick = st.selectbox("Choose an ingredient", [i.name for i in scan.ingredients], label_visibility="collapsed")
    ing = next(i for i in scan.ingredients if i.name == pick)
    flags = _flags_text(ing)
    match = {"exact": "exact match", "fuzzy": f"approximate match ({ing.confidence:.0%})", "unmatched": "not recognised"}[ing.match_type]
    st.markdown(
        f'<div class="card"><div>{theme.pill(ing.risk + " concern", ing.risk)}'
        f'<span class="meta">{", ".join(ing.functions) or "function unknown"} · {match}</span></div>'
        f'<p style="margin-top:.5rem">{ing.risk_note}</p>'
        + (f'<p><b>Regulatory note:</b> {ing.eu_note}</p>' if ing.eu_note and ing.matched else "")
        + (f'<p><b>For your profile:</b> {flags}</p>' if flags else "") + "</div>", unsafe_allow_html=True)
    if ing.raw.strip().lower() != ing.name.lower():
        st.caption(f"Printed on label as: “{ing.raw}”")


# ───────────────────────────── 3. Evidence Explorer ──────────────────────────
def explorer() -> None:
    theme.hero("Evidence Explorer", "Look at the sources behind every claim",
               "Peer-reviewed literature (PubMed), chemistry (PubChem), and regulatory context (FDA, EU CosIng).")
    scan = state.current_scan()
    names = [i.name for i in scan.ingredients] if scan else []
    c1, c2 = st.columns([1, 1])
    pick = c1.selectbox("Ingredient from current scan", names or ["(no scan yet)"], disabled=not names)
    custom = c2.text_input("…or search any ingredient", placeholder="e.g. bakuchiol")
    target = custom.strip() or (pick if names else "")
    if not target:
        theme.empty("Scan a product, or type an ingredient name above.")
        return
    canon = nlp.identify(target, 0).name
    items = _scan_evidence(scan, canon) if scan else st.session_state.extra_evidence.get(canon, [])
    if not items:
        if st.button(f"Retrieve evidence for {canon}", type="primary"):
            with st.spinner("Querying PubMed, PubChem…"):
                items = _fetch_one(canon)
            state.log("Evidence", f"Looked up {canon}", scan.id if scan else None)
            st.rerun()
        else:
            st.caption("Evidence for this ingredient has not been retrieved yet.")
            return
    by = {s: [i for i in items if i["source"] == s] for s in SOURCES}
    tabs = st.tabs(["Top evidence"] + [f"{s} ({len(by[s])})" for s in SOURCES] + ["Validation"])
    with tabs[0]:
        studies = [i for i in items if i["source"] == "PubMed"][:3] or items[:3]
        for n, it in enumerate(studies):
            _render_item(it, f"top{n}")
    for t, s in zip(tabs[1:5], SOURCES):
        with t:
            if not by[s]:
                st.caption(f"No {s} records for this ingredient.")
            for n, it in enumerate(by[s]):
                _render_item(it, f"{s}{n}")
    with tabs[5]:
        st.caption("Evidence processing (Ling Sante): records are normalised, de-duplicated and rule-validated "
                   "(title, secure link, abstract), ranked by source type, review status, recency and relevance, "
                   "then checked by the Ling Sante model for relevance to topical cosmetic use.")
        st.dataframe(pd.DataFrame([{"Source": i["source"], "Title": i["title"][:90], "Year": i.get("year"),
                                    "Score": i["score"], "Validated": "✓" if i["validated"] else "✗",
                                    "AI check": i.get("ai_verdict", "—"),
                                    "Notes": ", ".join(i["flags"]) or "—"} for i in items]),
                     hide_index=True, use_container_width=True)


# ───────────────────────────── 4. AI Assistant ───────────────────────────────
def assistant() -> None:
    theme.hero("AI Assistant", "Ask about your product or any cosmetic ingredient",
               "Answers are grounded in retrieved evidence, cite their sources and state their limits.")
    settings = get_settings()
    scan = state.current_scan()
    if not settings.has_llm:
        st.warning("Add OPENROUTER_API_KEY or NVIDIA_API_KEY (.env or sidebar → Connections) to enable the assistant.")
    chat = st.session_state.chat
    for m in chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("meta"):
                st.caption(m["meta"])
    if not chat:
        starters = ["Is this product suitable for sensitive skin?"] if scan else []
        starters += ["What does niacinamide do?", "Is phenoxyethanol safe?"]
        cols = st.columns(len(starters))
        for c, q in zip(cols, starters):
            if c.button(q, key=f"starter_{q}", use_container_width=True):
                st.session_state.queued_q = q
                st.rerun()
    q = st.chat_input("Ask a question about your product or an ingredient…", disabled=not settings.has_llm)
    q = q or st.session_state.queued_q
    st.session_state.queued_q = None
    if not q:
        return
    with st.chat_message("user"):
        st.markdown(q)
    chat.append({"role": "user", "content": q})
    with st.chat_message("assistant"):
        with st.spinner("Classifying question and retrieving evidence…"):
            names = [i.name for i in scan.ingredients] if scan else []
            res = intent.classify(q, settings, names)
            ings, ev = _assistant_context(scan, res)
            profile = st.session_state.profile
            ctx, numbered = synthesis.build_context(ings, ev, profile)
            msgs = synthesis.question_messages(q, ctx, res.intent, [m for m in chat[:-1]])
        meta: dict = {}
        with st.spinner("Writing answer…"):
            raw = st.write_stream(synthesis.stream_answer(settings, msgs, meta)) or ""
        if raw:
            text, used = synthesis.validate_citations(raw, len(numbered))
        else:
            text = ("I couldn't reach the language model just now. "
                    + ("Here is what the knowledge base says:\n\n" + "\n".join(
                        f"- **{i.name}** – {i.risk} concern. {i.risk_note}" for i in ings[:5]) if ings else
                       "Please try again in a moment."))
            used = []
            st.markdown(text)
        final = text + synthesis.sources_markdown(numbered, used)
        label = f"Intent: {res.intent.replace('_', ' ')} ({res.method}, {res.confidence:.0%})" + (
            f" · model: {meta['model']}" if meta.get("model") else "")
        st.caption(label)
    chat.append({"role": "assistant", "content": final, "meta": label})
    state.log("Question", q, scan.id if scan else None)
    if final != raw:  # sources / cleaned citations were added – redraw from history
        st.rerun()


def _assistant_context(scan, res):
    """Choose which ingredients to discuss and make sure evidence exists for them."""
    settings = get_settings()
    if res.ingredients:
        ings = [nlp.identify(n, k) for k, n in enumerate(res.ingredients[:3], 1)]
    elif scan and res.intent in ("product_suitability", "ingredient_safety", "evidence_lookup", "regulatory_status",
                                 "ingredient_identification", "ingredient_function"):
        ings = evidence.prioritise(scan.ingredients)[:4]
    else:
        return [], {}
    ev: dict[str, list[dict]] = {}
    missing = []
    for i in ings:
        have = (scan.evidence.get(i.name) if scan else None) or st.session_state.extra_evidence.get(i.name)
        if have:
            ev[i.name] = have
        else:
            missing.append(i)
    if missing:
        raw = evidence.gather(missing, settings, depth=len(missing))
        for n, r in raw.items():
            e = KB.get(n)
            items = processing.process(n, r["items"], list(e.aliases) if e else None)
            st.session_state.extra_evidence[n] = items
            ev[n] = items
        ling_sante.validate({n: ev[n] for n in raw}, settings)
    return ings, ev


# ───────────────────────────── 5. History ────────────────────────────────────
def history() -> None:
    theme.hero("Current-session history", "Everything you did in this session",
               "History lives only in this browser session and disappears when you clear it or close the tab.")
    events = st.session_state.events
    if not events:
        theme.empty("Nothing yet – your scans, lookups and questions will be listed here.")
        return
    for n, ev in enumerate(reversed(events)):
        c1, c2, c3 = st.columns([1, 6, 1.4])
        c1.markdown(f'<span class="meta">{ev["time"].strftime("%H:%M")}</span>', unsafe_allow_html=True)
        c2.markdown(f'{theme.pill(ev["kind"], "src")} {ev["text"]}', unsafe_allow_html=True)
        if ev["kind"] == "Scan" and ev["scan_id"]:
            if c3.button("Open", key=f"open_{n}"):
                st.session_state.current = ev["scan_id"]
                st.switch_page(st.session_state["_pages"]["scanner"])
    st.divider()
    if st.button("Clear session", type="primary"):
        state.clear()
        st.rerun()
