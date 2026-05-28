#!/usr/bin/env python3
"""
scripts/5_tag_hpo_subtrees.py
────────────────────────────────────────────────────────────────────────────────
Tag every HPO term with a category (symptom / lab_finding / sign)
by walking subtrees of the Human Phenotype Ontology.

Inputs
──────
    data/processed/symptom_index.json
    data/raw/hp.obo

Output
──────
    data/processed/symptom_categories.json   { hp_id → "symptom" | "lab_finding" | "sign" }
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

import pronto
from dotenv import load_dotenv

load_dotenv()

SCRIPT_DIR     = Path(__file__).resolve().parent
DATA_DIR       = Path(os.getenv("DATA_DIR",       str(SCRIPT_DIR.parent / "data")))
ONTOLOGIES_DIR = Path(os.getenv("ONTOLOGIES_DIR", str(SCRIPT_DIR.parent / "data" / "raw")))
PROCESSED_DIR  = DATA_DIR / "processed"

SYMPTOM_INDEX = PROCESSED_DIR / "symptom_index.json"
HP_OBO        = ONTOLOGIES_DIR / "hp.obo"
OUTPUT        = PROCESSED_DIR / "symptom_categories.json"

SYMPTOM_ROOTS = {
    "HP:0025142",  # Constitutional symptom
    "HP:0000707",  # Nervous system abnormality
    "HP:0000478",  # Eye abnormality
    "HP:0000598",  # Ear abnormality
    "HP:0000119",  # Genitourinary abnormality
    "HP:0003011",  # Musculoskeletal abnormality (musculature)
    "HP:0001574",  # Skin/integument abnormality
    "HP:0025031",  # Digestive system abnormality
    "HP:0002086",  # Respiratory system abnormality
}

LAB_ROOTS = {
    "HP:0001939",  # Abnormality of metabolism/homeostasis
    "HP:0001871",  # Abnormality of blood and blood-forming tissues
    "HP:0003256",  # Abnormality of the coagulation cascade
    "HP:0012415",  # Abnormal blood gas level
}

ALL_ROOTS = SYMPTOM_ROOTS | LAB_ROOTS


def main():
    assert SYMPTOM_INDEX.exists(), f"FATAL: {SYMPTOM_INDEX} not found"
    assert HP_OBO.exists(), f"FATAL: {HP_OBO} not found"

    symptom_index = json.loads(SYMPTOM_INDEX.read_text(encoding="utf-8"))
    print(f"Loaded symptom_index.json: {len(symptom_index)} entries")

    print(f"Loading hp.obo ...")
    ont = pronto.Ontology(str(HP_OBO))
    print(f"Loaded hp.obo: {len(ont.terms())} terms")

    for root_id in sorted(ALL_ROOTS):
        term = ont.get(root_id)
        if term is None or term.obsolete:
            print(f"WARNING: subtree root {root_id} not found or obsolete — its descendants will be tagged 'sign'")

    print("Building descendant sets for subtree roots ...")
    root_descendants: dict[str, set[str]] = {}
    for root_id in ALL_ROOTS:
        term = ont.get(root_id)
        if term is None or term.obsolete:
            root_descendants[root_id] = set()
            continue
        root_descendants[root_id] = {t.id for t in term.subclasses(distance=None, with_self=True)}

    def tag_term(hp_id: str) -> str:
        term = ont.get(hp_id)
        if term is None:
            return "sign"
        ancestor_ids = {t.id for t in term.superclasses(distance=None, with_self=True)}
        for root_id in SYMPTOM_ROOTS:
            if root_id in ancestor_ids:
                return "symptom"
        for root_id in LAB_ROOTS:
            if root_id in ancestor_ids:
                return "lab_finding"
        return "sign"

    print("Tagging HP terms ...")
    categories: dict[str, str] = {}
    for hp_id in symptom_index:
        categories[hp_id] = tag_term(hp_id)

    print("\nRunning verification checks ...")

    expected_count = len(symptom_index)
    actual_count   = len(categories)
    if actual_count != expected_count:
        print(f"FAIL check 1: count mismatch — output {actual_count} vs expected {expected_count}")
        sys.exit(1)
    print(f"  PASS check 1: output count == {actual_count}")

    present_cats = set(categories.values())
    if not {"symptom", "lab_finding", "sign"}.issubset(present_cats):
        print(f"FAIL check 2: not all three categories present — found {present_cats}")
        sys.exit(1)
    print(f"  PASS check 2: all three categories present")

    exertional = categories.get("HP:0002875")
    if exertional != "symptom":
        print(f"FAIL check 3: HP:0002875 tagged '{exertional}', expected 'symptom'")
        sys.exit(1)
    print(f"  PASS check 3: HP:0002875 -> 'symptom'")

    meta_term = ont.get("HP:0001939")
    if meta_term and not meta_term.obsolete:
        meta_descs = {t.id for t in meta_term.subclasses(distance=None, with_self=False)}
        lab_descs  = [hp for hp in meta_descs if categories.get(hp) == "lab_finding"]
        if not lab_descs:
            print("FAIL check 4: HP:0001939 descendants include no lab_finding tags")
            sys.exit(1)
        print(f"  PASS check 4: HP:0001939 has {len(lab_descs)} lab_finding descendants")
    else:
        print("  WARN check 4: HP:0001939 not in ontology — skipping")

    bad = [hp for hp, cat in categories.items() if not cat]
    if bad:
        print(f"FAIL check 5: {len(bad)} entries with null/empty category")
        sys.exit(1)
    print(f"  PASS check 5: no null or empty values")

    dist  = Counter(categories.values())
    total = len(categories)
    print(f"\nCategory distribution:")
    for cat in ["symptom", "lab_finding", "sign"]:
        count = dist[cat]
        pct   = 100 * count / total
        expected = {"symptom": (35, 55), "lab_finding": (25, 40), "sign": (15, 30)}[cat]
        in_range = expected[0] <= pct <= expected[1]
        flag = "" if in_range else "  WARNING: outside expected range"
        print(f"  {cat:<12}: {count:>6}  ({pct:.1f}%){flag}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(categories, indent=None), encoding="utf-8")
    print(f"\nWrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} KB)")
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
