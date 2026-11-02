"""Offline tests (no network, no API keys). Run:  python -m unittest discover -s tests -v"""
import sys, unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import evidence, intent, nlp, ocr, processing, synthesis  # noqa: E402

PUBMED_XML = """<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>123</PMID><Article>
<Journal><JournalIssue><PubDate><Year>2023</Year></PubDate></JournalIssue><Title>J Test Derm</Title></Journal>
<ArticleTitle>Niacinamide in <i>dermatology</i>: an overview</ArticleTitle>
<Abstract><AbstractText Label="BACKGROUND">Niacinamide improves skin barrier function.</AbstractText></Abstract>
<PublicationTypeList><PublicationType>Review</PublicationType></PublicationTypeList></Article></MedlineCitation>
<PubmedData/></PubmedArticle></PubmedArticleSet>"""


class NLP(unittest.TestCase):
    def test_pipeline(self):
        ings = nlp.analyze_text("INGREDIENTS: Aqua/Water/Eau, Glycerin, Fragrance (Linalool, Limonene), "
                                "Phenoxyethanoll, Xyzzy Compound. May contain CI 77891")
        names = [i.name for i in ings]
        self.assertEqual(names, ["Aqua", "Glycerin", "Parfum", "Phenoxyethanol", "Xyzzy Compound"])
        self.assertEqual(ings[3].match_type, "fuzzy")
        self.assertEqual(ings[4].risk, "Unknown")

    def test_profile_flags(self):
        retinol = nlp.identify("Retinol")
        self.assertIn("Ask a clinician before use during pregnancy", retinol.flags(["Pregnant or nursing"]))
        self.assertEqual(retinol.flags([]), [])


class Intent(unittest.TestCase):
    def test_rules(self):
        self.assertEqual(intent.classify("Is phenoxyethanol safe for sensitive skin?").intent, "ingredient_safety")
        self.assertEqual(intent.classify("Is retinol banned in the EU?").intent, "regulatory_status")
        self.assertIn("Niacinamide", intent.classify("what does niacinamide do").ingredients)


class Evidence(unittest.TestCase):
    def test_parse_and_process(self):
        items = evidence._parse_pubmed_xml(PUBMED_XML)
        self.assertEqual(items[0]["id"], "PMID 123")
        self.assertEqual(items[0]["kind"], "review")
        self.assertIn("dermatology", items[0]["title"])
        ranked = processing.process("Niacinamide", items + evidence.regulatory_items("Niacinamide"))
        self.assertTrue(all(r["validated"] for r in ranked))

    def test_network_failure_degrades_gracefully(self):
        from core.config import get_settings
        with mock.patch("core.http.get_json", side_effect=RuntimeError("boom")):
            res = evidence.fetch_ingredient(nlp.identify("Glycerin"), get_settings())
        self.assertTrue(res["errors"])
        self.assertEqual({i["source"] for i in res["items"]}, {"FDA", "CosIng"})


class LingSante(unittest.TestCase):
    def test_parse_and_apply(self):
        from core import ling_sante
        from core.config import get_settings
        items = [{"source": "PubMed", "title": "A", "snippet": "x", "score": 1.0, "flags": [], "validated": True},
                 {"source": "PubMed", "title": "B", "snippet": "y", "score": 0.9, "flags": [], "validated": True}]
        s = get_settings().__class__(**{**get_settings().__dict__, "openrouter_key": "x", "ling_enabled": True})
        reply = 'Sure: [{"id":1,"verdict":"irrelevant"},{"id":2,"verdict":"relevant"}]'
        with mock.patch("core.llm.complete", return_value=reply):
            self.assertTrue(ling_sante.validate({"Niacinamide": items}, s))
        self.assertEqual(items[0]["title"], "B")  # irrelevant paper pushed down
        self.assertFalse(items[1]["validated"])

    def test_failure_is_silent(self):
        from core import ling_sante, llm
        from core.config import get_settings
        s = get_settings().__class__(**{**get_settings().__dict__, "openrouter_key": "x", "ling_enabled": True})
        items = [{"source": "PubMed", "title": "A", "snippet": "x", "score": 1.0, "flags": [], "validated": True}]
        with mock.patch("core.llm.complete", side_effect=llm.LLMError("down")):
            self.assertFalse(ling_sante.validate({"X": items}, s))
        self.assertTrue(items[0]["validated"])


class Reasoning(unittest.TestCase):
    def _run(self, tokens):
        from core.config import get_settings
        meta = {}
        with mock.patch("core.llm.chat_stream", lambda *a, **k: iter(tokens)):
            return "".join(synthesis.stream_answer(get_settings(), [], meta)), meta

    def test_marker_strips_reasoning(self):
        out, _ = self._run(["Here's a thinking process: 1. Analyze", " User Input...\n[[ANS", "WER]]\nNot ideal [1]."])
        self.assertEqual(out, "Not ideal [1].")

    def test_leak_without_marker_is_dropped(self):
        out, meta = self._run(["Here's a thinking process:\n1. Analyze User Input: ..."])
        self.assertEqual(out, "")
        self.assertIn("reasoning", meta["error"])

    def test_clean_model_without_marker_passes(self):
        out, _ = self._run(["Niacinamide is a vitamin B3 form [1]."])
        self.assertIn("vitamin B3", out)

    def test_title_hit_filters_context(self):
        good = {"source": "PubMed", "title": "Phenoxyethanol safety", "snippet": "x", "year": 2020, "kind": "article",
                "url": "https://a", "id": "PMID 1", "venue": "", "curated": False}
        bad = {**good, "title": "Hair growth in rats", "snippet": "uses phenoxyethanol as vehicle", "url": "https://b", "id": "PMID 2"}
        ev = processing.process("Phenoxyethanol", [bad, good])
        self.assertEqual(ev[0]["title"], "Phenoxyethanol safety")
        _, numbered = synthesis.build_context(nlp.analyze_text("Phenoxyethanol"), {"Phenoxyethanol": ev}, [])
        self.assertEqual([n["title"] for n in numbered], ["Phenoxyethanol safety"])


class Synthesis(unittest.TestCase):
    def test_citation_validation(self):
        text, used = synthesis.validate_citations("Good [1]. Fake [7]. Both [2, 9].", 2)
        self.assertEqual(used, [1, 2])
        self.assertNotIn("7", text)
        self.assertNotIn("9", text)

    def test_fallback(self):
        out = synthesis.fallback_summary(nlp.analyze_text("Aqua, Phenoxyethanol, Parfum"), ["Sensitive skin"], "offline")
        self.assertIn("Phenoxyethanol", out)
        self.assertIn("offline", out)


class OCR(unittest.TestCase):
    def test_clean_json_and_fences(self):
        raw = '```json\n[{"bbox":[0,0,1,1],"category":"Text","text":"Aqua, Glycerin, Niacinamide"}]\n```'
        self.assertEqual(ocr.clean_ocr_output(raw), "Aqua, Glycerin, Niacinamide")

    def test_fallback_chain(self):
        from core.config import get_settings
        from core import llm
        import io
        from PIL import Image
        buf = io.BytesIO(); Image.new("RGB", (50, 50), "white").save(buf, "PNG")
        calls = []

        def fake(base, key, model, *a, **k):
            calls.append(model)
            if len(calls) < 3:
                raise llm.LLMError("down")
            return "Ingredients: Aqua, Glycerin, Niacinamide"
        s = get_settings().__class__(**{**get_settings().__dict__, "openrouter_key": "x", "nvidia_key": "",
                                         "paddle_base": "https://example.com/v1"})
        with mock.patch("core.llm.complete", side_effect=fake):
            res = ocr.run_ocr(buf.getvalue(), s)
        self.assertTrue(res.ok)
        self.assertEqual(len(res.attempts), 3)
        self.assertIn("Muse", res.provider)
        self.assertEqual([a["provider"] for a in res.attempts][:2], ["Dots3 via OpenRouter", "PaddleOCR-VL (cloud)"])


if __name__ == "__main__":
    unittest.main()
