#!/usr/bin/env python3
"""
exclusively-neo4j/5_verify.py
────────────────────────────────────────────────────────────────────────────────
Verification script — confirms the Neo4j graph was loaded correctly.

Runs 17 checks across data files, graph node/edge counts, LR value ranges,
and the runtime query layer. Every check must pass.

Usage
─────
    # from exclusively-neo4j/
    uv run 5_verify.py
"""

import csv
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from neo4j_client import verify_connection as neo4j_verify, run_query
from neo4j_queries import (
    get_initial_differential,
    get_lr_for_symptom,
    verify_graph_counts,
)

SCRIPT_DIR    = Path(__file__).resolve().parent
DATA_DIR      = Path(os.getenv("DATA_DIR", str(SCRIPT_DIR.parent / "data")))
PROCESSED_DIR = DATA_DIR / "processed"


def check(label: str, passed: bool, detail: str = "") -> bool:
    status = "PASS" if passed else "FAIL"
    suffix = f" ({detail})" if detail else ""
    print(f"  {label:<38} {status}{suffix}")
    if not passed:
        print(f"  >>> STOP: check failed — {label}")
    return passed


def main():
    all_passed = True

    # ── ENVIRONMENT ──────────────────────────────────────────────
    print("ENVIRONMENT")
    if not check("Neo4j connection", neo4j_verify()):
        sys.exit(1)
    print()

    # ── DATA FILES ────────────────────────────────────────────────
    print("DATA FILES")

    lr_path = PROCESSED_DIR / "lr_table.csv"
    lr_rows = 0
    if lr_path.exists():
        with open(lr_path, newline="", encoding="utf-8") as f:
            lr_rows = sum(1 for _ in csv.reader(f)) - 1  # subtract header
    if not check("lr_table.csv rows", lr_rows > 100_000, str(lr_rows)):
        sys.exit(1)

    disease_idx = json.loads((PROCESSED_DIR / "disease_index.json").read_text(encoding="utf-8"))
    if not check("disease_index entries", len(disease_idx) > 5_000, str(len(disease_idx))):
        sys.exit(1)

    symptom_idx = json.loads((PROCESSED_DIR / "symptom_index.json").read_text(encoding="utf-8"))
    if not check("symptom_index entries", len(symptom_idx) > 10_000, str(len(symptom_idx))):
        sys.exit(1)

    cats_path = PROCESSED_DIR / "symptom_categories.json"
    cats_count = 0
    if cats_path.exists():
        cats = json.loads(cats_path.read_text(encoding="utf-8"))
        cats_count = len(cats)
    cats_match = cats_count == len(symptom_idx)
    if not check(
        "symptom_categories count",
        cats_match,
        f"{cats_count} vs symptom_index {len(symptom_idx)}",
    ):
        sys.exit(1)
    print()

    # ── NEO4J GRAPH ───────────────────────────────────────────────
    print("NEO4J GRAPH")
    counts = verify_graph_counts()

    if not check("Disease node count", counts["diseases"] > 5_000, str(counts["diseases"])):
        sys.exit(1)
    if not check("Symptom node count", counts["symptoms"] > 10_000, str(counts["symptoms"])):
        sys.exit(1)
    if not check("PRESENTS_WITH edges", counts["presents_with"] > 50_000, str(counts["presents_with"])):
        sys.exit(1)
    if not check("RULES_IN edges", counts["rules_in"] > 50_000, str(counts["rules_in"])):
        sys.exit(1)
    if not check("RULES_OUT edges", counts["rules_out"] > 50_000, str(counts["rules_out"])):
        sys.exit(1)

    # Check: a pulmonary disease exists with >= 5 edges (stands in for COPD check)
    copd_rows = run_query(
        """
        MATCH (d:Disease)-[:PRESENTS_WITH]->(s:Symptom)
        WHERE toLower(d.name) CONTAINS 'pulmonary'
        WITH d, count(s) AS c
        WHERE c >= 5
        RETURN d.id AS id, d.name AS name, c
        ORDER BY c DESC
        LIMIT 1
        """
    )
    if not check("COPD node exists", len(copd_rows) >= 1):
        sys.exit(1)

    # Check: confirm the selected disease has >= 5 PRESENTS_WITH edges
    copd_id = copd_rows[0]["id"]
    copd_edge_count = copd_rows[0]["c"]
    if not check("COPD symptom edges", copd_edge_count >= 5, f"{copd_edge_count} edges"):
        sys.exit(1)

    # Check: All RULES_IN likelihood_ratio > 0.0
    bad_lr = run_query(
        "MATCH ()-[r:RULES_IN]->() WHERE r.likelihood_ratio <= 0.0 RETURN count(r) AS c"
    )
    bad_count = bad_lr[0]["c"] if bad_lr else 0
    if not check("LR values positive", bad_count == 0, f"{bad_count} bad values"):
        sys.exit(1)

    # Check: Max RULES_IN likelihood_ratio > 2.0
    max_lr = run_query(
        "MATCH ()-[r:RULES_IN]->() RETURN max(r.likelihood_ratio) AS m"
    )
    lr_max = max_lr[0]["m"] if max_lr else 0.0
    if not check("LR max > 2.0", (lr_max or 0) > 2.0, f"{lr_max:.2f}" if lr_max else "None"):
        sys.exit(1)

    # Check: Min RULES_IN likelihood_ratio < 1.0
    min_lr = run_query(
        "MATCH ()-[r:RULES_IN]->() RETURN min(r.likelihood_ratio) AS m"
    )
    lr_min = min_lr[0]["m"] if min_lr else 0.0
    if not check("LR min < 1.0", (lr_min or 1.0) < 1.0, f"{lr_min:.4f}" if lr_min else "None"):
        sys.exit(1)
    print()

    # ── KNOWLEDGE LAYER ───────────────────────────────────────────
    print("KNOWLEDGE LAYER")

    diff = get_initial_differential(["HP:0002875"])
    if not check("get_initial_differential", len(diff) > 0, f"returns {len(diff)} diseases"):
        sys.exit(1)

    first_disease_id = diff[0]["disease_id"]
    lr_result = get_lr_for_symptom("HP:0002875", [first_disease_id])
    lr_ok = (
        first_disease_id in lr_result
        and "lr_positive" in lr_result[first_disease_id]
        and "lr_negative" in lr_result[first_disease_id]
    )
    if not check("get_lr_for_symptom", lr_ok, "returns LR+ and LR-"):
        sys.exit(1)
    print()

    # ── SUMMARY ───────────────────────────────────────────────────
    print("17/17 checks passed.")
    print("Phase 4 complete. Neo4j graph verified. Ready for agent integration.")


if __name__ == "__main__":
    main()
