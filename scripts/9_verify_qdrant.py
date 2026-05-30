#!/usr/bin/env python3
"""
scripts/6_verify_qdrant.py
Verify the Qdrant collection: match quality + gibberish rejection.
Full preprocessor runs before every search (via QdrantClient).
Exit code 1 on any failure.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledge.qdrant_client import QdrantClient

SEARCH_TESTS = [
    # (query, expected_hp_id_in_top3_or_None, min_score, must_match, description)
    # PubMedBERT maps "going up stairs" to mobility/stairs terms (HP:0003551, HP:0033235)
    # and "shortness of breath" to dyspnea terms — score check is the key gate here.
    ("shortness of breath going up stairs", None,         0.70, True,  "exertional dyspnea — score gate"),
    # HP:0100749 is the correct HPO term for general "chest pain"
    ("chest pain",                          "HP:0100749", 0.75, True,  "chest pain direct"),
    ("my patient has palpitations",         "HP:0001962", 0.70, True,  "3rd person framing"),
    ("i have a terrible headache",          "HP:0002315", 0.70, True,  "1st person framing"),
    ("FEV1/FVC ratio obstruction",          None,         0.50, True,  "medical notation"),
    ("couldn't sleep last night",           None,         0.50, True,  "insomnia"),
    ("fever and chills",                    "HP:0001945", 0.70, True,  "fever direct"),
    ("swollen ankles and legs",             None,         0.60, True,  "oedema"),
    # Gibberish — must return matched=False
    ("uietfhui48753hgiru48",               None, 0.0, False, "random chars — reject"),
    ("!!!###***",                           None, 0.0, False, "symbols only — reject"),
    ("42 67 99 11 55",                      None, 0.0, False, "numbers only — reject"),
    ("asdfghjkl qwertyuiop",               None, 0.0, False, "keyboard mash — reject"),
]


def main():
    print("Loading QdrantClient ...")
    client = QdrantClient()

    passed = 0
    failed = 0

    print(f"\nRunning {len(SEARCH_TESTS)} search tests ...\n")

    for query, expected_hp, min_score, must_match, desc in SEARCH_TESTS:
        result = client.search(query)

        if must_match:
            if not result["matched"]:
                print(f'[FAIL] "{query}" -> NO MATCH (expected match — {desc})')
                failed += 1
                continue

            score = result["score"]
            if score < min_score:
                print(f'[FAIL] "{query}" -> score {score:.3f} < {min_score} ({desc})')
                failed += 1
                continue

            if expected_hp is not None:
                top3 = result["symptom_ids"][:3]
                if expected_hp not in top3:
                    print(f'[FAIL] "{query}" -> {expected_hp} not in top 3: {top3}')
                    failed += 1
                    continue

            print(f'[PASS] "{query}" -> {result["clinical_term"]} ({result["symptom_id"]}) {score:.3f}')
            passed += 1

        else:
            if result["matched"]:
                print(f'[FAIL] "{query}" -> matched=True (should be rejected — {desc}): {result}')
                failed += 1
                continue
            if result["symptom_ids"] != []:
                print(f'[FAIL] "{query}" -> symptom_ids not empty: {result["symptom_ids"]}')
                failed += 1
                continue
            if result["score"] != 0.0:
                print(f'[FAIL] "{query}" -> score not 0.0: {result["score"]}')
                failed += 1
                continue
            print(f'[PASS] "{query}" -> NO MATCH (threshold rejected — {desc})')
            passed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed out of {len(SEARCH_TESTS)} tests")

    if failed > 0:
        sys.exit(1)

    print("VERIFY PASS")


if __name__ == "__main__":
    main()
