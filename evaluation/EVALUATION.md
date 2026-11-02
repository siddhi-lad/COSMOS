# COSMOS evaluation – one list, in run order

Run everything from the `cosmos` folder with the virtual environment active.
`pip install -r evaluation/requirements-eval.txt` once.

| # | What | Command | Needs | Time | Paper table |
|---|------|---------|-------|------|-------------|
| 1 | **Freeze the setup** (models, versions, KB hash, seed – written automatically into `results/run_manifest.json` by every command below) | – | – | – | "Experimental setup" |
| 2 | **Identification on ToxFox gold text** (P/R/F1, exact match, KB coverage, concern-flag test vs ToxFox pollutant list) | `python evaluation/toxfox_eval.py --data data_tox --mode text` | data only | seconds | Table 1 |
| 3 | **Ablation** – each NLP component switched off, paired Wilcoxon + Holm | `python evaluation/ablation.py --data data_tox` | data only | seconds | Table 2 |
| 4 | **Knowledge-base gap analysis** (most frequent unrecognised ingredients) + **NLP latency** | produced by step 3 → `results/kb_gaps_*.csv`, `latency.json` | – | – | Limitations / Table 1 note |
| 5 | **OCR, Dots3 only** (CER, WER, success rate, latency, end-to-end identification) | `python evaluation/toxfox_eval.py --data data_tox --mode ocr --routes dots --no-chain --pause 3` | OpenRouter key | ~5 min | Table 3 |
| 6 | **OCR robustness** (blur / low-res / dark / heavy JPEG) | repeat step 5 with `--degrade blur`, `lowres`, `dark`, `jpeg` (and `--n 30`) | OpenRouter key | ~3 min each | Table 4 |
| 7 | **Intent classifier** – make the question set | `python evaluation/annotation_tools.py make-intents` | – | seconds | – |
| 8 | **Two people label the 70 questions independently** (`evaluation/annotation/intent_questions.csv`, columns annotator1 / annotator2; agree disputes in `adjudicated`) | – | 2 humans | ~15 min each | – |
| 9 | **Evidence layer** – retrieve papers for 30 ingredients | `python evaluation/annotation_tools.py collect-evidence --n 30` | internet (+ OpenRouter key for Ling Sante) | ~5 min | – |
| 10 | **Two people rate paper relevance 1/0** (`annotation/evidence_relevance.csv`, columns rater1 / rater2) | – | 2 humans | ~30 min | – |
| 11 | **Answer quality** – generate 20 answers | `python evaluation/annotation_tools.py collect-answers --n 20` | internet + LLM keys | ~10 min | – |
| 12 | **Two people rate the answers** (`annotation/answer_quality.csv`: faithful 1-5, unsupported claims, helpful 1-5) | – | 2 humans | ~30 min each | – |
| 13 | **Score all human-labelled parts** (accuracy, macro-F1, kappa, P@3, MRR, nDCG@5, citation precision, Ling Sante precision/recall, answer scores) | `python evaluation/annotation_tools.py score` (add `--llm` to also score the LLM intent fallback) | – | seconds | Tables 5–7 |
| 14 | **Collect tables** – everything is already saved as `.md`, `.csv`, `.tex`, `.json` in `results/` | – | – | – | copy into the paper |

If time is very short, do steps 2–6 (fully automatic) and 7–8 (15 minutes). Steps 9–12 add the evidence and answer-quality results.

## What each result means (and its limits – say these in the paper)

* **Identification (2):** measured on the ToxFox INCI lists, which were produced for German-market products; the knowledge base
  is small (curated), so risk/function can only be given for recognised ingredients – report the coverage figure.
* **Concern flags (2):** an independent check against ToxFox's pollutant list. Low recall is expected – the knowledge base
  contains only a few of those substances. Report it as a coverage limitation, not as an accuracy score.
* **OCR (5, 6):** the dataset's own `raw_ingredients_list` is sometimes noisy, so the headline CER/WER uses a cleaned reference
  built from `INCI_list` (`CER_raw` is also reported). Free-tier API limits can cause failed calls; they are counted in the success
  rate and never hidden.
* **Annotation parts (7–13):** results are only as good as the raters. Report how many raters, that they worked independently,
  and the kappa. The intent file contains no suggested labels on purpose.
* **Ablation (3):** a component with no measurable effect on this dataset (e.g. fuzzy matching) should be reported as such.
* **Not covered:** clinical validity of risk labels (they are curated, not clinically validated), multilingual labels, and
  user studies. State these as future work.
