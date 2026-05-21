#!/usr/bin/env python3
"""
exclusively-neo4j/4_load_neo4j.py
────────────────────────────────────────────────────────────────────────────────
Script 4 — Load the Neo4j disease-symptom knowledge graph.

Reads processed data files produced by scripts 2, 3, and 4a, then
bulk-loads Disease nodes, Symptom nodes, and three edge types into Neo4j.

Inputs  (DATA_DIR/processed/)
──────────────────────────────
    disease_index.json
    symptom_index.json
    symptom_categories.json
    lr_table.csv

Usage
─────
    # from exclusively-neo4j/
    uv run 4_load_neo4j.py
"""

import csv
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from neo4j_client import run_write_query, run_write_batch, verify_connection
from neo4j_queries import verify_graph_counts

SCRIPT_DIR    = Path(__file__).resolve().parent
DATA_DIR      = Path(os.getenv("DATA_DIR", str(SCRIPT_DIR.parent / "data")))
PROCESSED_DIR = DATA_DIR / "processed"

DISEASE_INDEX = PROCESSED_DIR / "disease_index.json"
SYMPTOM_INDEX = PROCESSED_DIR / "symptom_index.json"
SYMPTOM_CATS  = PROCESSED_DIR / "symptom_categories.json"
LR_TABLE      = PROCESSED_DIR / "lr_table.csv"

BATCH_SIZE = 500


def part_a_constraints():
    print("Part A: Creating constraints and indexes ...")
    run_write_query(
        "CREATE CONSTRAINT disease_id IF NOT EXISTS "
        "FOR (d:Disease) REQUIRE d.id IS UNIQUE"
    )
    run_write_query(
        "CREATE CONSTRAINT symptom_id IF NOT EXISTS "
        "FOR (s:Symptom) REQUIRE s.id IS UNIQUE"
    )
    run_write_query(
        "CREATE INDEX disease_name IF NOT EXISTS "
        "FOR (d:Disease) ON (d.name)"
    )
    run_write_query(
        "CREATE INDEX symptom_name IF NOT EXISTS "
        "FOR (s:Symptom) ON (s.name)"
    )
    print("  Constraints and indexes created.")


def part_b_disease_nodes():
    print("Part B: Loading Disease nodes ...")
    disease_index = json.loads(DISEASE_INDEX.read_text(encoding="utf-8"))
    total = len(disease_index)

    query = """
    UNWIND $batch AS row
    MERGE (d:Disease {id: row.id})
    SET d.name = row.name,
        d.icd = row.icd,
        d.omim_id = row.omim_id,
        d.orpha_id = row.orpha_id,
        d.prevalence = row.prevalence
    """

    batch = []
    loaded = 0
    for doid, info in disease_index.items():
        name = info.get("name", "") if isinstance(info, dict) else str(info)
        icd = info.get("icd", []) if isinstance(info, dict) else []
        omim_ids = info.get("omim", []) if isinstance(info, dict) else []
        orpha_ids = info.get("orpha", []) if isinstance(info, dict) else []

        batch.append({
            "id": doid,
            "name": name,
            "icd": icd if isinstance(icd, list) else [],
            "omim_id": omim_ids[0] if omim_ids else None,
            "orpha_id": orpha_ids[0] if orpha_ids else None,
            "prevalence": 0.01,
        })

        if len(batch) >= BATCH_SIZE:
            run_write_batch(query, batch)
            loaded += len(batch)
            batch = []
            if loaded % 2000 == 0:
                print(f"  Loaded {loaded} / {total} disease nodes")

    if batch:
        run_write_batch(query, batch)
        loaded += len(batch)

    print(f"  Loaded {loaded} / {total} disease nodes — DONE")
    return loaded


def part_c_symptom_nodes():
    print("Part C: Loading Symptom nodes ...")
    symptom_index = json.loads(SYMPTOM_INDEX.read_text(encoding="utf-8"))
    categories = json.loads(SYMPTOM_CATS.read_text(encoding="utf-8"))
    total = len(symptom_index)

    query = """
    UNWIND $batch AS row
    MERGE (s:Symptom {id: row.id})
    SET s.name = row.name,
        s.synonyms = row.synonyms,
        s.category = row.category
    """

    batch = []
    loaded = 0
    for hp_id, info in symptom_index.items():
        name = info.get("name", "") if isinstance(info, dict) else str(info)
        synonyms = info.get("synonyms", []) if isinstance(info, dict) else []

        batch.append({
            "id": hp_id,
            "name": name,
            "synonyms": synonyms if isinstance(synonyms, list) else [],
            "category": categories.get(hp_id, "sign"),
        })

        if len(batch) >= BATCH_SIZE:
            run_write_batch(query, batch)
            loaded += len(batch)
            batch = []
            if loaded % 2000 == 0:
                print(f"  Loaded {loaded} / {total} symptom nodes")

    if batch:
        run_write_batch(query, batch)
        loaded += len(batch)

    print(f"  Loaded {loaded} / {total} symptom nodes — DONE")
    return loaded


def part_d_edges():
    print("Part D: Loading edges from lr_table.csv ...")

    pw_query = """
    UNWIND $batch AS row
    MATCH (d:Disease {id: row.doid})
    MATCH (s:Symptom {id: row.hp_id})
    MERGE (d)-[r:PRESENTS_WITH]->(s)
    SET r.sensitivity = row.sensitivity
    """
    ri_query = """
    UNWIND $batch AS row
    MATCH (d:Disease {id: row.doid})
    MATCH (s:Symptom {id: row.hp_id})
    MERGE (s)-[r:RULES_IN]->(d)
    SET r.likelihood_ratio = row.lr_positive
    """
    ro_query = """
    UNWIND $batch AS row
    MATCH (d:Disease {id: row.doid})
    MATCH (s:Symptom {id: row.hp_id})
    MERGE (s)-[r:RULES_OUT]->(d)
    SET r.likelihood_ratio = row.lr_negative
    """

    pw_batch, ri_batch, ro_batch = [], [], []
    edges_processed = 0
    skipped_empty = 0

    with open(LR_TABLE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doid = row.get("disease_doid", "").strip()
            if not doid:
                skipped_empty += 1
                continue

            hp_id = row.get("hp_id", "").strip()
            if not hp_id:
                skipped_empty += 1
                continue

            try:
                sensitivity = float(row.get("sensitivity", 0))
                lr_positive = float(row.get("lr_positive", 1.0))
                lr_negative = float(row.get("lr_negative", 1.0))
            except (ValueError, TypeError):
                skipped_empty += 1
                continue

            record = {"doid": doid, "hp_id": hp_id, "sensitivity": sensitivity,
                      "lr_positive": lr_positive, "lr_negative": lr_negative}
            pw_batch.append(record)
            ri_batch.append(record)
            ro_batch.append(record)

            if len(pw_batch) >= BATCH_SIZE:
                run_write_batch(pw_query, pw_batch)
                run_write_batch(ri_query, ri_batch)
                run_write_batch(ro_query, ro_batch)
                edges_processed += len(pw_batch)
                pw_batch, ri_batch, ro_batch = [], [], []
                if edges_processed % 5000 == 0:
                    print(f"  Processed {edges_processed} edge rows ...")

    if pw_batch:
        run_write_batch(pw_query, pw_batch)
        run_write_batch(ri_query, ri_batch)
        run_write_batch(ro_query, ro_batch)
        edges_processed += len(pw_batch)

    print(f"  Edge rows processed: {edges_processed}, skipped (empty DOID/HP): {skipped_empty}")
    return edges_processed, skipped_empty


def main():
    print("Verifying Neo4j connection ...")
    if not verify_connection():
        print("FAIL: Cannot connect to Neo4j. Run docker compose up -d first.")
        sys.exit(1)
    print("  Neo4j connection OK")
    print()

    part_a_constraints()
    print()

    disease_count = part_b_disease_nodes()
    print()

    symptom_count = part_c_symptom_nodes()
    print()

    edges_processed, skipped_empty = part_d_edges()
    print()

    counts = verify_graph_counts()

    print("=" * 39)
    print("  NEO4J LOAD COMPLETE")
    print("=" * 39)
    print(f"  Disease nodes created:       {disease_count}")
    print(f"  Symptom nodes created:       {symptom_count}")
    print(f"  PRESENTS_WITH edges:         {counts['presents_with']}")
    print(f"  RULES_IN edges:              {counts['rules_in']}")
    print(f"  RULES_OUT edges:             {counts['rules_out']}")
    print(f"  Rows skipped (empty DOID):   {skipped_empty}")
    print("=" * 39)


if __name__ == "__main__":
    main()
