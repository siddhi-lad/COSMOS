"""Curated cosmetic-ingredient knowledge base used by stages 3 (function mapping
and risk assessment) and 6 (regulatory notes for the FDA / CosIng cards).

It is intentionally small and hand-checked rather than large and noisy. Anything
not found here is reported as "unrecognised" – never guessed. Extend it by adding
rows to `_ROWS`. Risk levels describe *concern for general consumers*:
Low = widely used, well tolerated · Medium = known irritant/allergen or regulatory
limits · High = banned/strongly restricted in at least one major market.

Regulatory text is a pointer, not legal advice – always confirm on the linked
CosIng / FDA pages (limits change)."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

FDA_DEFAULT = ("The US FDA does not pre-approve cosmetic ingredients (except colour additives); "
               "manufacturers are responsible for safety (FD&C Act, MoCRA 2022).")
COSING_DEFAULT = "No specific entry flagged in the curated list – check the CosIng database for current Annex status."

# name, aliases, functions, risk, note, tags, eu_note, chem (PubChem name; None = skip), pubmed terms (None = skip)
_ROWS: list[tuple] = [
    ("Aqua", ["water", "aqua water", "aqua water eau", "eau", "purified water"], ["Solvent"], "Low",
     "Primary solvent; no safety concern at cosmetic use.", (), None, None, None),
    ("Glycerin", ["glycerol", "glycerine"], ["Humectant", "Skin conditioning"], "Low",
     "Draws water into the skin; very well tolerated.", (), None, "glycerin", ["glycerin", "glycerol"]),
    ("Niacinamide", ["nicotinamide", "vitamin b3"], ["Skin conditioning", "Barrier support"], "Low",
     "Generally well tolerated; occasional flushing or irritation at higher concentrations.", (), None,
     "niacinamide", ["niacinamide", "nicotinamide"]),
    ("Phenoxyethanol", ["2 phenoxyethanol"], ["Preservative"], "Medium",
     "Common preservative; may irritate some sensitive skin. Limited to 1% in the EU.", ("irritant",),
     "Restricted preservative: EU Annex V, maximum 1.0%.", "phenoxyethanol", ["phenoxyethanol"]),
    ("Carbomer", ["carbopol"], ["Thickener"], "Low", "Gelling polymer; not absorbed through skin.", (), None, None,
     ["carbomer"]),
    ("Sodium Hyaluronate", ["hyaluronic acid", "sodium hyaluronate crosspolymer"], ["Humectant"], "Low",
     "Hydrating; very well tolerated.", (), None, "sodium hyaluronate", ["hyaluronic acid", "sodium hyaluronate"]),
    ("Parfum", ["fragrance", "perfume", "aroma", "parfum fragrance"], ["Fragrance"], "Medium",
     "A leading cause of cosmetic contact allergy; exact components are undisclosed.", ("allergen", "fragrance", "irritant"),
     "EU requires individual labelling of listed fragrance allergens.", None, ["fragrance", "perfume"]),
    ("Retinol", ["vitamin a"], ["Skin conditioning", "Anti-ageing"], "Medium",
     "Effective but irritating (dryness, peeling) and increases sun sensitivity; use caution in pregnancy.",
     ("irritant", "pregnancy_caution"), "Restricted in the EU (Annex III) – concentration limits apply.", "retinol",
     ["retinol"]),
    ("Retinyl Palmitate", [], ["Skin conditioning", "Antioxidant"], "Medium",
     "Milder vitamin A ester; same cautions in pregnancy and sun exposure.", ("pregnancy_caution",),
     "Falls under EU vitamin A restrictions (Annex III).", "retinyl palmitate", ["retinyl palmitate"]),
    ("Methylisothiazolinone", ["mit", "methylisothiazolone"], ["Preservative"], "High",
     "Potent contact allergen; banned in leave-on cosmetics in the EU.", ("allergen", "irritant"),
     "EU: prohibited in leave-on products; rinse-off limited to 15 ppm (Annex V).", "methylisothiazolinone",
     ["methylisothiazolinone"]),
    ("Methylchloroisothiazolinone", ["mci"], ["Preservative"], "High",
     "Contact allergen; only permitted in rinse-off products in the EU.", ("allergen", "irritant"),
     "EU: rinse-off only, strict limits (Annex V).", "methylchloroisothiazolinone", ["methylchloroisothiazolinone"]),
    ("Triclosan", [], ["Antimicrobial", "Preservative"], "High",
     "Restricted antimicrobial with environmental and resistance concerns.", (),
     "EU: restricted to defined product types and limits (Annex V).", "triclosan", ["triclosan"]),
    ("Hydroquinone", [], ["Skin lightening"], "High", "Prohibited in cosmetics in the EU; can cause irritation and ochronosis.", (),
     "EU: prohibited in cosmetic products (Annex II).", "hydroquinone", ["hydroquinone"]),
    ("Methylparaben", [], ["Preservative"], "Low", "Well-studied preservative; considered safe at permitted levels.", (),
     "EU Annex V: max 0.4% (single ester), 0.8% (mixtures).", "methylparaben", ["methylparaben", "methyl paraben"]),
    ("Ethylparaben", [], ["Preservative"], "Low", "Well-studied preservative; considered safe at permitted levels.", (),
     "EU Annex V: max 0.4% (single ester), 0.8% (mixtures).", "ethylparaben", ["ethylparaben"]),
    ("Propylparaben", [], ["Preservative"], "Medium", "Longer-chain paraben; stricter EU limits because of endocrine-activity debate.", (),
     "EU Annex V: propyl + butyl parabens total max 0.14%.", "propylparaben", ["propylparaben"]),
    ("Butylparaben", [], ["Preservative"], "Medium", "Longer-chain paraben; stricter EU limits because of endocrine-activity debate.", (),
     "EU Annex V: propyl + butyl parabens total max 0.14%.", "butylparaben", ["butylparaben"]),
    ("Sodium Lauryl Sulfate", ["sls", "sodium dodecyl sulfate"], ["Surfactant"], "Medium",
     "Effective cleanser but a known skin irritant, especially for dry or sensitive skin.", ("irritant",), None,
     "sodium lauryl sulfate", ["sodium lauryl sulfate", "sodium dodecyl sulfate"]),
    ("Sodium Laureth Sulfate", ["sles"], ["Surfactant"], "Low", "Milder than SLS; may still irritate sensitive skin.", (), None, None,
     ["sodium laureth sulfate"]),
    ("Cocamidopropyl Betaine", ["capb"], ["Surfactant"], "Medium", "Mild surfactant; allergy is usually from manufacturing impurities.", ("allergen",), None,
     None, ["cocamidopropyl betaine"]),
    ("Alcohol Denat.", ["alcohol denat", "denatured alcohol", "sd alcohol 40"], ["Solvent", "Astringent"], "Medium",
     "Drying and potentially irritating in high amounts, especially on sensitive skin.", ("irritant",), None, "ethanol",
     ["denatured alcohol", "ethanol"]),
    ("Salicylic Acid", ["bha", "beta hydroxy acid"], ["Exfoliant", "Anti-acne"], "Medium",
     "Effective exfoliant; can dry or irritate. Seek advice before use in pregnancy.", ("irritant", "pregnancy_caution"),
     "EU Annex III/V: concentration limits and age restrictions apply.", "salicylic acid", ["salicylic acid"]),
    ("Glycolic Acid", ["aha"], ["Exfoliant"], "Medium", "Can irritate and increases sun sensitivity; daily sunscreen advised.", ("irritant",), None,
     "glycolic acid", ["glycolic acid"]),
    ("Lactic Acid", [], ["Exfoliant", "pH adjuster"], "Medium", "Gentler AHA; may irritate and increases sun sensitivity.", ("irritant",), None,
     "lactic acid", ["lactic acid"]),
    ("Citric Acid", [], ["pH adjuster", "Chelating agent"], "Low", "Used in small amounts to balance pH.", (), None, "citric acid", ["citric acid"]),
    ("Sodium Hydroxide", ["lye", "caustic soda"], ["pH adjuster"], "Low", "Used in tiny amounts and neutralised in the formula.", (), None,
     "sodium hydroxide", None),
    ("Ascorbic Acid", ["vitamin c", "l ascorbic acid"], ["Antioxidant", "Skin conditioning"], "Low",
     "Antioxidant; can sting at low pH or high concentration.", ("irritant",), None, "ascorbic acid", ["ascorbic acid", "vitamin C"]),
    ("Tocopherol", ["vitamin e"], ["Antioxidant"], "Low", "Antioxidant that also protects the formula; well tolerated.", (), None,
     "tocopherol", ["tocopherol", "vitamin E"]),
    ("Panthenol", ["provitamin b5", "dexpanthenol", "vitamin b5"], ["Humectant", "Skin conditioning"], "Low", "Soothing and hydrating; very well tolerated.", (), None,
     "panthenol", ["panthenol", "dexpanthenol"]),
    ("Allantoin", [], ["Skin conditioning", "Soothing"], "Low", "Soothing; very well tolerated.", (), None, "allantoin", ["allantoin"]),
    ("Dimethicone", [], ["Emollient", "Occlusive"], "Low", "Silicone that smooths and protects; not absorbed.", (), None, None, ["dimethicone"]),
    ("Cyclopentasiloxane", ["d5"], ["Emollient", "Solvent"], "Medium", "Volatile silicone; EU limits in rinse-off products over environmental concerns.", (),
     "EU: restricted (Annex XVII, REACH) in wash-off products.", None, ["cyclopentasiloxane"]),
    ("Cetearyl Alcohol", [], ["Emollient", "Emulsifier", "Thickener"], "Low", "Fatty alcohol (not drying); well tolerated.", (), None, None, None),
    ("Cetyl Alcohol", [], ["Emollient", "Thickener"], "Low", "Fatty alcohol; well tolerated.", (), None, "cetyl alcohol", ["cetyl alcohol"]),
    ("Stearic Acid", [], ["Emulsifier", "Thickener"], "Low", "Fatty acid; well tolerated.", (), None, "stearic acid", None),
    ("Caprylic/Capric Triglyceride", ["caprylic capric triglyceride"], ["Emollient"], "Low", "Light coconut-derived emollient; well tolerated.", (), None, None,
     ["caprylic capric triglyceride"]),
    ("Butylene Glycol", [], ["Solvent", "Humectant"], "Low", "Solvent/humectant; well tolerated.", (), None, "butylene glycol", ["butylene glycol"]),
    ("Propylene Glycol", [], ["Solvent", "Humectant"], "Low", "Occasional sensitisation in sensitive individuals.", ("irritant",), None,
     "propylene glycol", ["propylene glycol"]),
    ("Xanthan Gum", [], ["Thickener"], "Low", "Natural-origin thickener; well tolerated.", (), None, None, None),
    ("Ethylhexylglycerin", [], ["Preservative booster", "Skin conditioning"], "Low", "Boosts preservatives; rare sensitisation.", (), None,
     "ethylhexylglycerin", ["ethylhexylglycerin"]),
    ("Sodium Benzoate", [], ["Preservative"], "Low", "Mild preservative effective at low pH.", (), None, "sodium benzoate", ["sodium benzoate"]),
    ("Potassium Sorbate", [], ["Preservative"], "Low", "Mild preservative; well tolerated.", (), None, "potassium sorbate", ["potassium sorbate"]),
    ("Benzyl Alcohol", [], ["Preservative", "Fragrance"], "Medium", "Preservative/solvent; a listed fragrance allergen in the EU.", ("allergen",),
     "EU: labelled fragrance allergen; preservative limit 1% (Annex V).", "benzyl alcohol", ["benzyl alcohol"]),
    ("Disodium EDTA", ["edta", "tetrasodium edta"], ["Chelating agent"], "Low", "Stabilises formulas; well tolerated.", (), None, "edetate disodium",
     ["disodium EDTA", "EDTA"]),
    ("Squalane", [], ["Emollient"], "Low", "Lightweight, skin-identical emollient; well tolerated.", (), None, "squalane", ["squalane"]),
    ("Butyrospermum Parkii Butter", ["shea butter", "butyrospermum parkii shea butter", "butyrospermum parkii shea butter"],
     ["Emollient", "Occlusive"], "Low", "Rich plant butter; well tolerated apart from rare nut-allergy concerns.", (), None, None, ["shea butter"]),
    ("Cocos Nucifera Oil", ["coconut oil"], ["Emollient"], "Low", "Moisturising but can clog pores on acne-prone skin.", ("comedogenic",), None, None,
     ["coconut oil"]),
    ("Isopropyl Myristate", [], ["Emollient"], "Low", "Light emollient; can be pore-clogging on acne-prone skin.", ("comedogenic",), None,
     "isopropyl myristate", ["isopropyl myristate"]),
    ("Petrolatum", ["petroleum jelly"], ["Occlusive"], "Low", "Highly effective occlusive; refined grades are safe.", (), None, None, ["petrolatum"]),
    ("Paraffinum Liquidum", ["mineral oil"], ["Emollient", "Occlusive"], "Low", "Cosmetic-grade mineral oil is non-irritating.", (), None, None,
     ["mineral oil"]),
    ("Lanolin", [], ["Emollient", "Occlusive"], "Medium", "Effective moisturiser but a recognised contact allergen for some people.", ("allergen",), None, None,
     ["lanolin"]),
    ("Urea", [], ["Humectant", "Exfoliant"], "Low", "Hydrates at low %, gently exfoliates at higher %.", (), None, "urea", ["urea"]),
    ("Sodium PCA", [], ["Humectant"], "Low", "Natural moisturising factor component; well tolerated.", (), None, None, ["sodium PCA"]),
    ("Triethanolamine", [], ["pH adjuster", "Emulsifier"], "Medium", "Can form nitrosamines with certain ingredients; mild irritant.", ("irritant",),
     "EU: restricted to avoid nitrosamine formation (Annex III).", "triethanolamine", ["triethanolamine"]),
    ("Titanium Dioxide", ["ci 77891"], ["UV filter", "Colorant", "Opacifier"], "Low", "Mineral sun filter; avoid inhaling powder or spray forms.", (),
     "EU: UV filter (Annex VI) and colorant; nano form specially regulated.", "titanium dioxide", ["titanium dioxide"]),
    ("Zinc Oxide", ["ci 77947"], ["UV filter", "Skin protectant"], "Low", "Mineral sun filter; well tolerated.", (), "EU: UV filter (Annex VI).",
     "zinc oxide", ["zinc oxide"]),
    ("Benzophenone-3", ["oxybenzone"], ["UV filter"], "Medium", "Chemical sun filter; photo-allergy and environmental concerns.", ("allergen",),
     "EU: UV filter (Annex VI) with maximum concentrations and labelling.", "oxybenzone", ["oxybenzone", "benzophenone-3"]),
    ("Linalool", [], ["Fragrance"], "Medium", "Fragrance component that can sensitise once oxidised.", ("allergen", "fragrance"),
     "EU: labelled fragrance allergen.", "linalool", ["linalool"]),
    ("Limonene", [], ["Fragrance"], "Medium", "Citrus fragrance component that can sensitise once oxidised.", ("allergen", "fragrance"),
     "EU: labelled fragrance allergen.", "limonene", ["limonene"]),
    ("Citronellol", [], ["Fragrance"], "Medium", "Fragrance component; known contact allergen.", ("allergen", "fragrance"),
     "EU: labelled fragrance allergen.", "citronellol", ["citronellol"]),
    ("Geraniol", [], ["Fragrance"], "Medium", "Fragrance component; known contact allergen.", ("allergen", "fragrance"),
     "EU: labelled fragrance allergen.", "geraniol", ["geraniol"]),
    ("Coumarin", [], ["Fragrance"], "Medium", "Fragrance component; known contact allergen.", ("allergen", "fragrance"),
     "EU: labelled fragrance allergen.", "coumarin", ["coumarin"]),
    ("Butylphenyl Methylpropional", ["lilial"], ["Fragrance"], "High", "Prohibited in EU cosmetics (reproductive toxicity classification).", ("allergen", "fragrance"),
     "EU: prohibited since 2022.", "lilial", ["butylphenyl methylpropional", "lilial"]),
    ("Ceramide NP", ["ceramide 3", "ceramide np"], ["Skin conditioning", "Barrier support"], "Low", "Skin-identical lipid; strengthens the barrier.", (), None, None,
     ["ceramide"]),
    ("Cholesterol", [], ["Skin conditioning", "Barrier support"], "Low", "Skin-identical lipid; well tolerated.", (), None, "cholesterol", ["cholesterol"]),
    ("Bakuchiol", [], ["Skin conditioning", "Anti-ageing"], "Low", "Plant-derived retinol alternative; well tolerated in studies.", (), None,
     "bakuchiol", ["bakuchiol"]),
    ("Centella Asiatica Extract", ["centella asiatica leaf extract", "cica"], ["Skin conditioning", "Soothing"], "Low", "Soothing botanical; well tolerated.", (), None,
     None, ["centella asiatica"]),
    ("Aloe Barbadensis Leaf Juice", ["aloe vera", "aloe barbadensis leaf extract", "aloe vera gel"], ["Skin conditioning", "Soothing"], "Low",
     "Soothing botanical; well tolerated.", (), None, None, ["aloe vera"]),
    ("Camellia Sinensis Leaf Extract", ["green tea extract"], ["Antioxidant", "Skin conditioning"], "Low", "Antioxidant botanical; well tolerated.", (), None, None,
     ["green tea", "camellia sinensis"]),
]


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = s.replace("&", " and ")
    s = re.sub(r"\d+(\.\d+)?\s*%", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


@dataclass(frozen=True)
class KBEntry:
    name: str
    aliases: tuple[str, ...]
    functions: tuple[str, ...]
    risk: str
    note: str
    tags: tuple[str, ...]
    eu_note: str | None
    chem: str | None
    pm_terms: tuple[str, ...] | None

    @property
    def cosing_note(self) -> str:
        return self.eu_note or COSING_DEFAULT


KB: dict[str, KBEntry] = {}
INDEX: dict[str, str] = {}  # normalised alias → canonical name
for _n, _al, _fn, _rk, _nt, _tg, _eu, _ch, _pm in _ROWS:
    KB[_n] = KBEntry(_n, tuple(_al), tuple(_fn), _rk, _nt, tuple(_tg), _eu, _ch, tuple(_pm) if _pm else None)
    for _a in [_n, *_al]:
        INDEX[normalize(_a)] = _n
