"""Fills the `annotator1` column of annotation/intent_questions.csv with labels assigned by Claude (an LLM annotator),
so no manual labelling is needed.   Run:  python evaluation/apply_llm_labels.py

HOW THESE LABELS WERE MADE: each question was read and given one of the 7 intent labels using the written definitions in
annotation/intent_labels.txt, independently of the rule-based classifier being evaluated (the classifier's regexes were
not consulted). The questions were also written with these topics in mind, so this is an LLM-annotated pilot, NOT
human ground truth. The paper must say so (a ready-made sentence is printed by `annotation_tools.py score`)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import annotation_tools as t  # noqa: E402

LABEL_BY_GROUP = ["ingredient_function", "ingredient_safety", "ingredient_identification", "regulatory_status",
                  "product_suitability", "evidence_lookup", "general_cosmetic"]
LABELS = {q.strip(): LABEL_BY_GROUP[i] for i, line in enumerate(t.QUESTIONS.splitlines())
          for q in line.split("|") if q.strip()}


def main() -> None:
    f = t.ANN / "intent_questions.csv"
    if not f.is_file():
        raise SystemExit("Run `python evaluation/annotation_tools.py make-intents --n 30` first.")
    rows = t.read_csv(f)
    missing = 0
    for r in rows:
        lab = LABELS.get(r["question"].strip())
        if lab:
            r["annotator1"] = lab
        else:
            missing += 1
    t.write_csv(f, rows, ["id", "question", "annotator1", "annotator2", "adjudicated"])
    (t.ANN / "LLM_LABELS.txt").write_text("annotator1 labels were assigned by Claude (LLM annotator).\n", encoding="utf-8")
    print(f"Labelled {len(rows) - missing}/{len(rows)} questions in {f}. Now run: python evaluation/annotation_tools.py score")


if __name__ == "__main__":
    main()
