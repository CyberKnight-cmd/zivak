#!/usr/bin/env python3
"""
scripts/6_load_neo4j.py
────────────────────────────────────────────────────────────────────────────────
Load the Neo4j disease-symptom knowledge graph.

Reads processed data files produced by scripts 1–5, then bulk-loads
Disease nodes, Symptom nodes, and three edge types into Neo4j.

Outputs (data/processed/):
    edge_direction_errors.json   — always []  (no drops under current logic)
    edge_reclassifications.json  — edges where lr_positive < 1.0 (reclassified)
    edge_inversion_errors.json   — corrupt LR pairs (lr+ < 1 AND lr- <= 1), always []
    edge_node_missing.json       — rows where disease DOID has no node in Neo4j
"""

import csv
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SCRIPT_DIR    = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils.neo4j_client  import run_write_query, run_write_batch, verify_connection  # noqa: E402
from utils.neo4j_queries import verify_graph_counts  # noqa: E402

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

    batch  = []
    loaded = 0
    for doid, info in disease_index.items():
        name      = info.get("name", "") if isinstance(info, dict) else str(info)
        icd       = info.get("icd", [])   if isinstance(info, dict) else []
        omim_ids  = info.get("omim", [])  if isinstance(info, dict) else []
        orpha_ids = info.get("orpha", []) if isinstance(info, dict) else []

        batch.append({
            "id":        doid,
            "name":      name,
            "icd":       icd if isinstance(icd, list) else [],
            "omim_id":   omim_ids[0]  if omim_ids  else None,
            "orpha_id":  orpha_ids[0] if orpha_ids else None,
            "prevalence": float(info.get("prevalence", 0.000001)) if isinstance(info, dict) else 0.000001,
        })

        if len(batch) >= BATCH_SIZE:
            run_write_batch(query, batch)
            loaded += len(batch)
            batch   = []
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
    categories    = json.loads(SYMPTOM_CATS.read_text(encoding="utf-8"))
    total         = len(symptom_index)

    query = """
    UNWIND $batch AS row
    MERGE (s:Symptom {id: row.id})
    SET s.name = row.name,
        s.synonyms = row.synonyms,
        s.category = row.category
    """

    batch  = []
    loaded = 0
    for hp_id, info in symptom_index.items():
        name     = info.get("name", "")      if isinstance(info, dict) else str(info)
        synonyms = info.get("synonyms", []) if isinstance(info, dict) else []

        batch.append({
            "id":       hp_id,
            "name":     name,
            "synonyms": synonyms if isinstance(synonyms, list) else [],
            "category": categories.get(hp_id, "sign"),
        })

        if len(batch) >= BATCH_SIZE:
            run_write_batch(query, batch)
            loaded += len(batch)
            batch   = []
            if loaded % 2000 == 0:
                print(f"  Loaded {loaded} / {total} symptom nodes")

    if batch:
        run_write_batch(query, batch)
        loaded += len(batch)

    print(f"  Loaded {loaded} / {total} symptom nodes — DONE")
    return loaded


def _get_loaded_disease_doids() -> set[str]:
    """Query Neo4j for all Disease node IDs loaded in Part B."""
    from utils.neo4j_client import run_query
    rows = run_query("MATCH (d:Disease) RETURN d.id AS id")
    return {r["id"] for r in rows}


def part_d_edges():
    print("Part D: Loading edges from lr_table.csv ...")

    # Fetch the set of Disease DOIDs actually in Neo4j (after Part B)
    print("  Fetching loaded Disease DOIDs from Neo4j ...")
    loaded_disease_doids = _get_loaded_disease_doids()
    print(f"  Found {len(loaded_disease_doids)} Disease nodes in Neo4j")

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
    SET r.likelihood_ratio = row.rules_in_lr
    """
    ro_query = """
    UNWIND $batch AS row
    MATCH (d:Disease {id: row.doid})
    MATCH (s:Symptom {id: row.hp_id})
    MERGE (s)-[r:RULES_OUT]->(d)
    SET r.likelihood_ratio = row.rules_out_lr
    """

    pw_batch, ri_batch, ro_batch = [], [], []
    edges_processed  = 0
    skipped_empty    = 0
    reclassified:    list[dict] = []
    inversion_errors: list[dict] = []
    node_missing:    list[dict] = []

    with open(LR_TABLE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doid  = row.get("disease_doid", "").strip()
            hp_id = row.get("hp_id", "").strip()

            if not doid or not hp_id:
                skipped_empty += 1
                continue

            # Step 1 — check Disease node exists
            if doid not in loaded_disease_doids:
                node_missing.append({
                    "disease_doid":  doid,
                    "hp_id":         hp_id,
                    "disease_omim":  row.get("disease_omim", ""),
                    "reason":        "no_disease_node",
                })
                continue

            try:
                sensitivity  = float(row.get("sensitivity", 0))
                lr_positive  = float(row.get("lr_positive", 1.0))
                lr_negative  = float(row.get("lr_negative", 1.0))
            except (ValueError, TypeError):
                skipped_empty += 1
                continue

            # Step 2 — determine edge direction
            if lr_positive >= 1.0:
                # Normal: presence of symptom argues for disease
                rules_in_lr  = lr_positive
                rules_out_lr = lr_negative

            elif lr_negative > 1.0:
                # Reclassify: absence of symptom argues for disease
                rules_in_lr  = lr_negative
                rules_out_lr = lr_positive
                reclassified.append({
                    "disease_doid":            doid,
                    "hp_id":                   hp_id,
                    "original_lr_positive":    lr_positive,
                    "original_lr_negative":    lr_negative,
                    "reclassified_rules_in_lr":  lr_negative,
                    "reclassified_rules_out_lr": lr_positive,
                    "reason": "lr_positive < 1.0 — symptom more common in population than disease",
                })

            else:
                # Both LRs invalid — corrupt row
                inversion_errors.append({
                    "disease_doid": doid,
                    "hp_id":        hp_id,
                    "lr_positive":  lr_positive,
                    "lr_negative":  lr_negative,
                    "reason":       "both LR values invalid — row skipped",
                })
                continue

            record = {
                "doid": doid, "hp_id": hp_id, "sensitivity": sensitivity,
                "rules_in_lr": rules_in_lr, "rules_out_lr": rules_out_lr,
            }
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

    # Write all four output files
    def _write(path, obj):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2)

    _write(PROCESSED_DIR / "edge_direction_errors.json",  [])
    _write(PROCESSED_DIR / "edge_reclassifications.json", reclassified)
    _write(PROCESSED_DIR / "edge_inversion_errors.json",  inversion_errors)
    _write(PROCESSED_DIR / "edge_node_missing.json",      node_missing)

    print(f"  Edge rows processed:  {edges_processed}")
    print(f"  Skipped (empty):      {skipped_empty}")
    print(f"  Reclassified edges:   {len(reclassified)}")
    print(f"  Inversion errors:     {len(inversion_errors)} (should be 0)")
    print(f"  Node-missing rows:    {len(node_missing)}")
    return edges_processed, skipped_empty, reclassified, inversion_errors, node_missing


def main():
    print("Verifying Neo4j connection ...")
    if not verify_connection():
        print("FAIL: Cannot connect to Neo4j.")
        sys.exit(1)
    print("  Neo4j connection OK\n")

    part_a_constraints()
    print()

    disease_count = part_b_disease_nodes()
    print()

    symptom_count = part_c_symptom_nodes()
    print()

    edges_processed, skipped_empty, reclassified, inversion_errors, node_missing = part_d_edges()
    print()

    counts = verify_graph_counts()

    print("=" * 45)
    print("  NEO4J LOAD COMPLETE")
    print("=" * 45)
    print(f"  Disease nodes created:       {disease_count}")
    print(f"  Symptom nodes created:       {symptom_count}")
    print(f"  PRESENTS_WITH edges:         {counts['presents_with']}")
    print(f"  RULES_IN edges:              {counts['rules_in']}")
    print(f"  RULES_OUT edges:             {counts['rules_out']}")
    print(f"  Rows skipped (empty):        {skipped_empty}")
    print(f"  Reclassified edges:          {len(reclassified)}")
    print(f"  Inversion errors:            {len(inversion_errors)}")
    print(f"  Node-missing rows:           {len(node_missing)}")
    print("=" * 45)


if __name__ == "__main__":
    main()
