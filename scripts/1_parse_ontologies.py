#!/usr/bin/env python3
"""
scripts/1_parse_ontologies.py
────────────────────────────────────────────────────────────────────────────────
Script 1 — Parse all ontology source files and write clean intermediate data.

Inputs  (data/raw/)
───────────────────
    doid.obo          Disease Ontology
    hp.obo            Human Phenotype Ontology
    phenotype.hpoa    HPO disease–phenotype annotations

Outputs  (data/processed/)
──────────────────────────
    disease_index.json      { doid → {name, icd:[…], omim:[…]} }
    symptom_index.json      { hp_id → {name, synonyms:[…]} }
    omim_to_doid.json       { "OMIM:606391" → "DOID:3083" }
    orpha_to_doid.json      { "ORPHA:586" → "DOID:0050425" }
    hpoa_annotations.json   [ {disease_id, hpo_id, frequency_raw}, … ]
    obsolete_doid_map.json  { old_doid → active_doid }  (audit output)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SCRIPT_DIR     = Path(__file__).resolve().parent        # scripts/
DATA_DIR       = Path(os.getenv("DATA_DIR",       str(SCRIPT_DIR.parent / "data")))
ONTOLOGIES_DIR = Path(os.getenv("ONTOLOGIES_DIR", str(SCRIPT_DIR.parent / "data" / "raw")))
PROCESSED_DIR  = DATA_DIR / "processed"

DOID_OBO  = ONTOLOGIES_DIR / "doid.obo"
HP_OBO    = ONTOLOGIES_DIR / "hp.obo"
HPOA_FILE = ONTOLOGIES_DIR / "phenotype.hpoa"

sys.path.insert(0, str(SCRIPT_DIR))
from utils.obo_parser  import parse_all
from utils.hpoa_parser import parse_hpoa

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _check_inputs() -> None:
    missing = [p for p in (DOID_OBO, HP_OBO, HPOA_FILE) if not p.exists()]
    if missing:
        for p in missing:
            logger.error("Required input file not found: %s", p)
        sys.exit(1)


def _write_json(obj: object, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    size_kb = path.stat().st_size / 1024
    logger.info("  ✓ Wrote %-40s  (%6.0f KB)", label, size_kb)


def main() -> None:
    _check_inputs()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("━━━ Parsing Disease Ontology + HPO in parallel ━━━━━━━━━━━━━━━")
    (disease_index, omim_to_doid, orpha_to_doid, obsolete_doid_map), symptom_index = \
        parse_all(DOID_OBO, HP_OBO)

    _write_json(disease_index,    PROCESSED_DIR / "disease_index.json",    "disease_index.json")
    _write_json(omim_to_doid,     PROCESSED_DIR / "omim_to_doid.json",     "omim_to_doid.json")
    _write_json(orpha_to_doid,    PROCESSED_DIR / "orpha_to_doid.json",    "orpha_to_doid.json")
    _write_json(symptom_index,    PROCESSED_DIR / "symptom_index.json",    "symptom_index.json")
    _write_json(obsolete_doid_map, PROCESSED_DIR / "obsolete_doid_map.json", "obsolete_doid_map.json")

    alt_id_count    = sum(1 for k, v in obsolete_doid_map.items() if not k.startswith("DOID:0") or True)
    replaced_count  = len([k for k in obsolete_doid_map if k.startswith("DOID:")])
    logger.info("  obsolete_doid_map: %d entries total", len(obsolete_doid_map))

    logger.info("━━━ HPOA annotations (phenotype.hpoa) ━━━━━━━━━━━━━━━━━━━━━━━━")
    annotations = parse_hpoa(HPOA_FILE)
    _write_json(annotations, PROCESSED_DIR / "hpoa_annotations.json", "hpoa_annotations.json")

    logger.info("━━━ Bridge quality check ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    omim_in_hpoa  = {r["disease_id"] for r in annotations if r["disease_id"].startswith("OMIM:")}
    orpha_in_hpoa = {r["disease_id"] for r in annotations if r["disease_id"].startswith("ORPHA:")}
    mapped_omim   = sum(1 for oid in omim_in_hpoa  if oid in omim_to_doid)
    mapped_orpha  = sum(1 for oid in orpha_in_hpoa if oid in orpha_to_doid)

    logger.info(
        "  OMIM disease IDs in HPOA   : %d  →  %d resolve to a DOID  (%.1f%%)",
        len(omim_in_hpoa), mapped_omim,
        100.0 * mapped_omim / len(omim_in_hpoa) if omim_in_hpoa else 0,
    )
    logger.info(
        "  ORPHA disease IDs in HPOA  : %d  →  %d resolve to a DOID  (%.1f%%)",
        len(orpha_in_hpoa), mapped_orpha,
        100.0 * mapped_orpha / len(orpha_in_hpoa) if orpha_in_hpoa else 0,
    )

    combined_bridge = {**omim_to_doid, **orpha_to_doid}
    resolvable = sum(
        1 for r in annotations
        if r["disease_id"] in combined_bridge and r["hpo_id"] in symptom_index
    )
    logger.info(
        "  Fully resolvable annotation pairs: %d / %d  (%.1f%%)",
        resolvable, len(annotations),
        100.0 * resolvable / len(annotations) if annotations else 0,
    )

    logger.info("━━━ All done ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    main()
