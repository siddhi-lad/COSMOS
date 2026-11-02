# COSMOS – Cosmetic Ingredient Intelligence

From product label to evidence-backed answers. Streamlit app. **API-only: no model is ever downloaded or run locally.**

## Project structure

```
cosmos/
├── app.py                  # Streamlit entry point: navigation, sidebar (skin profile, depth, API keys)
├── requirements.txt
├── .env.example            # copy to .env and add keys
├── .streamlit/config.toml  # theme
├── core/                   # the pipeline (no Streamlit imports – reusable & testable)
│   ├── config.py           # settings from .env / environment
│   ├── http.py             # pooled session, retries, TTL cache, rate limiter, circuit breaker
│   ├── llm.py              # OpenAI-compatible client: model fallback + streaming (OpenRouter + an optional PaddleOCR-VL endpoint)
│   ├── ocr.py              # [2] OCR chain: Dots3 → PaddleOCR-VL → Muse Glimmer
│   ├── knowledge.py        # curated ingredient KB (functions, risk, regulatory notes)
│   ├── nlp.py              # [3] cleaning → identification → function mapping → risk
│   ├── intent.py           # [4] rule-based intent classifier (+ optional LLM for ambiguous queries)
│   ├── evidence.py         # [6] PubMed · PubChem · FDA · CosIng retrieval (parallel)
│   ├── processing.py       # [6] evidence normalise / de-duplicate / rank (rules)
│   ├── ling_sante.py       # [6] Ling Sante model: one batched relevance check of retrieved studies
│   ├── synthesis.py        # [5][7] prompts, citation validation, sources, offline fallback
│   └── pipeline.py         # orchestrates 1 → 7
├── ui/                     # [8] theme, session state, the five pages
├── evaluation/             # [9] evaluate.py + gold.json
└── tests/test_offline.py   # 11 offline tests (no keys, no network)
```

## Setup (VS Code)

1. Install **Python 3.10+** and open the `cosmos` folder in VS Code (`File → Open Folder`).
2. Open a terminal (`Ctrl+` ` ) and create an environment:
   - Windows: `python -m venv .venv` then `.venv\Scripts\activate`
   - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
3. In VS Code press `Ctrl+Shift+P` → **Python: Select Interpreter** → choose `.venv`.
4. `pip install -r requirements.txt`
5. Create your `.env`: copy `.env.example` to `.env` (Windows: `copy .env.example .env`) and paste your keys.
   Your old `.env` works as-is: the variable names are the same (`OPENROUTER_API_KEY`, `NVIDIA_API_KEY`, `DOTS_MODEL`,
   `NEMOTRON_MODEL`, `MUSE_MODEL`, `LING_SANTE_MODEL`, `REQUEST_TIMEOUT`, `MAX_UPLOAD_SIZE_MB`).
   - **OpenRouter key** covers Dots3, Muse Glimmer, Nemotron and Ling Sante – there are no separate Dots3/Muse keys.
   - **NVIDIA key** is an optional second route for Nemotron and Muse.
   - **PaddleOCR-VL** is tried without a key (see `PADDLE_OCR_*`); it is a cloud call, nothing is installed locally.
   - **NCBI key** is optional (faster PubMed).
   You can also paste keys in the app instead: sidebar → Connections.
6. Run: `streamlit run app.py` → opens http://localhost:8501
7. First run: sidebar → Connections → **Check model ids**. It queries OpenRouter/NVIDIA live and shows which of your model ids exist. Fix any `NOT FOUND` in `.env` and restart.

Without any keys the app still works for **manual ingredient entry**, ingredient intelligence, PubMed/PubChem evidence and a rule-based summary; keys add OCR, the AI narrative and the assistant.

## How the graphical abstract maps to the code

| Abstract box | Implementation |
|---|---|
| 1 Input (image / manual) | Scanner page – two tabs |
| 2 OCR engine, 3-stage fallback | `core/ocr.py` – tries each provider in order; skips ones without a key; failed models are skipped for a cool-down period |
| 3 NLP: cleaning → identification → function → risk | `core/nlp.py` + `core/knowledge.py` |
| 4 Intent classifier (7 categories, optional LLM) | `core/intent.py` – rules first, LLM only if confidence < 0.55 |
| 5 LLM layer (Nemotron primary, Muse fallback) | `core/llm.py` + `core/config.py` routes: Nemotron → Muse, each via OpenRouter then NVIDIA; streaming |
| 6 Evidence layer (PubMed, PubChem, FDA, CosIng, Ling Sante) | `core/evidence.py`, `core/processing.py`, `core/ling_sante.py` |
| 7 Response synthesis (takeaways, per-ingredient, citations, limits, plain words) | `core/synthesis.py` – citations validated in code; source links built in code |
| 8 Web app (5 pages) | `ui/pages.py` |
| 9 Evaluation | `evaluation/evaluate.py` |

## Latency and error handling

Images are downscaled to 1600 px before upload · evidence for the highest-risk ingredients is fetched in parallel (depth slider) · results cached for 1 h · PubMed/PubChem rate-limited to stay under their limits · summary/answers are streamed · failed models are skipped for a cool-down · every network failure degrades to a still-useful result (rule-based summary, curated regulatory notes).

## Be aware (honest limits)

- **FDA and CosIng have no free public ingredient API.** Those two tabs show *curated* notes from `core/knowledge.py` plus links to the official databases, labelled "curated pointer". PubMed and PubChem are live API calls.
- **Ling Sante** is the `LING_SANTE_MODEL` from your `.env`: it checks whether each retrieved study is relevant to cosmetic use. It adds one batched call per retrieval and can be switched off in the sidebar; on any failure the rule-based ranking is kept.
- **Model ids** come from your earlier `.env`. I could not verify them from my sandbox – use *Check model ids*. A missing model just moves the chain to the next route.
- The knowledge base holds ~70 common ingredients. Unlisted ingredients show **Unknown** instead of a guess. Add rows to `_ROWS` in `knowledge.py` to extend it.
- Evaluation scores on `gold.json` are trivially high because that starter set was written against the KB. Use your expert-labelled dataset for real numbers. Metrics not covered by the script (retrieval nDCG/MRR, human rating of answers) need labelled data you supply.
- Information only – not medical advice.

## Tests and evaluation

```
python -m unittest discover -s tests -v
python evaluation/evaluate.py --latency
python evaluation/evaluate.py --ocr-dir data/ocr     # folder of label.jpg + label.txt (ground truth)
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| "All LLMs failed … HTTP 404" | model id no longer exists → *Check model ids*, edit `.env` |
| OCR "HTTP 400" | that model doesn't accept images → pick a vision model |
| "authentication failed" | wrong/missing key |
| PubMed results slow | add `NCBI_API_KEY`; lower *Evidence depth* |
| Changes to `.env` ignored | restart Streamlit (or paste keys in the sidebar) |

## Evaluation on the ToxFox dataset

```
pip install -r evaluation/requirements-eval.txt
# 1. download ~150 image + .yaml pairs from github.com/ki-iw/toxfox-ocr (data/images, data/ground_truth) into one folder, e.g. data_tox/
# 2. copy data/inci/pollutant_mapping.yml into evaluation/data/   (enables the concern-flag test)
python evaluation/toxfox_eval.py --data data_tox --mode text                  # offline, seconds
python evaluation/toxfox_eval.py --data data_tox --mode ocr                   # per OCR route + full chain (needs keys)
python evaluation/toxfox_eval.py --data data_tox --mode ocr --degrade blur    # also: lowres, jpeg, dark
```
Results (tables as .md / .csv / .tex, per-image CSVs) go to `results/`. OCR output is cached, so re-runs are free.
Check the dataset licence before redistributing images.