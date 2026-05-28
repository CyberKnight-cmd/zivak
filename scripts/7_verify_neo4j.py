#!/usr/bin/env python3
"""
scripts/7_verify_neo4j.py
────────────────────────────────────────────────────────────────────────────────
Verify the Neo4j graph was loaded correctly.
Uses knowledge/neo4j_client.py (the production client).
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SCRIPT_DIR    = Path(__file__).resolve().parent
BASE_DIR      = SCRIPT_DIR.parent
DATA_DIR      = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
PROCESSED_DIR = DATA_DIR / "processed"

# Add project root so 'knowledge' package is importable
sys.path.insert(0, str(BASE_DIR))

from knowledge.neo4j_client import Neo4jClient  # noqa: E402


def check(label: str, passed: bool, detail: str = "") -> bool:
    status = "PASS" if passed else "FAIL"
    suffix = f" ({detail})" if detail else ""
    print(f"  {label:<45} {status}{suffix}")
    if not passed:
        print(f"  >>> STOP: {label}")
    return passed


def main():
    client = Neo4jClient()

    def q(cypher: str) -> int:
        with client._driver.session() as session:
            result = session.run(cypher).single()
            return result[0] if result else 0

    print("NEO4J GRAPH COUNTS")

    d  = q("MATCH (d:Disease) RETURN count(d) AS n")
    sy = q("MATCH (s:Symptom) RETURN count(s) AS n")
    pw = q("MATCH ()-[r:PRESENTS_WITH]->() RETURN count(r) AS n")
    ri = q("MATCH ()-[r:RULES_IN]->() RETURN count(r) AS n")
    ro = q("MATCH ()-[r:RULES_OUT]->() RETURN count(r) AS n")

    if not check("Disease count == 12127", d == 12127, str(d)):
        sys.exit(1)
    if not check("Symptom count == 19389", sy == 19389, str(sy)):
        sys.exit(1)
    if not check("RULES_IN == PRESENTS_WITH", ri == pw, f"RI={ri} PW={pw}"):
        sys.exit(1)
    if not check("RULES_OUT == PRESENTS_WITH", ro == pw, f"RO={ro} PW={pw}"):
        sys.exit(1)

    print()
    print("LR VALIDITY")

    bri = q("MATCH ()-[r:RULES_IN]->() WHERE r.likelihood_ratio < 1.0 RETURN count(r) AS n")
    bro = q("MATCH ()-[r:RULES_OUT]->() WHERE r.likelihood_ratio > 1.0 RETURN count(r) AS n")
    # 0.01 is legitimately assigned to ICD-chapter diseases by step 3 — only flag if unexpectedly high
    op  = q("MATCH (d:Disease) WHERE d.prevalence = 0.01 RETURN count(d) AS n")

    if not check("RULES_IN with LR < 1.0 == 0", bri == 0, str(bri)):
        sys.exit(1)
    if not check("RULES_OUT with LR > 1.0 == 0", bro == 0, str(bro)):
        sys.exit(1)
    # ICD-chapter tier legitimately assigns 0.01; warn if count is suspiciously high (>2000)
    if not check("Diseases with prevalence = 0.01 < 2000 (ICD tier OK)", op < 2000, str(op)):
        sys.exit(1)

    print()
    print("EDGE NODE MISSING CHECK")

    missing_path = PROCESSED_DIR / "edge_node_missing.json"
    if missing_path.exists():
        missing = json.loads(missing_path.read_text(encoding="utf-8"))
        # LR table rows (105865) may contain duplicate (disease, symptom) pairs that MERGE
        # into a single edge; compare against actual loaded count from Step 6.
        lr_table_path = PROCESSED_DIR / "lr_table.csv"
        if lr_table_path.exists():
            import csv as _csv
            with open(lr_table_path, newline="", encoding="utf-8") as _f:
                lr_rows = sum(1 for _ in _csv.DictReader(_f))
            expected_pw_upper = lr_rows - len(missing)
            if not check(f"PRESENTS_WITH > 0 and <= {expected_pw_upper}",
                         0 < pw <= expected_pw_upper, f"got {pw}"):
                sys.exit(1)
            if not check("node-missing rows == 0", len(missing) == 0, str(len(missing))):
                sys.exit(1)
        else:
            print("  WARN: lr_table.csv not found — skipping edge count upper bound check")
    else:
        print("  WARN: edge_node_missing.json not found — skipping edge count check")

    print()
    print(f"STEP 7 PASS — Disease:{d} Symptom:{sy} PRESENTS_WITH:{pw}")
    client.close()


if __name__ == "__main__":
    main()
