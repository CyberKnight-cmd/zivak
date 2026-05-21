#!/usr/bin/env python3
"""
exclusively-neo4j/3_compute_lr_table.py
────────────────────────────────────────────────────────────────────────────────
Script 3 — Derive the likelihood-ratio table from parsed HPOA annotations.

Inputs  (DATA_DIR/processed/ — written by script 2)
────────────────────────────────────────────────────
    hpoa_annotations.json   [{disease_id, hpo_id, frequency_raw}, …]
    disease_index.json      {doid  → {name, icd, omim}}
    omim_to_doid.json       {"OMIM:606391" → "DOID:3083"}
    orpha_to_doid.json      {"ORPHA:586" → "DOID:0050425"}
    symptom_index.json      {hp_id → {name, synonyms}}

Output  (DATA_DIR/processed/)
──────────────────────────────
    lr_table.csv            One row per resolvable disease–symptom pair

CSV columns
───────────
    disease_omim     OMIM CURIE from HPOA          e.g.  "OMIM:606391"
    disease_doid     Resolved DOID                  e.g.  "DOID:3083"
    disease_name     Human-readable disease name    e.g.  "chronic obstructive pulmonary disease"
    hp_id            HPO CURIE                      e.g.  "HP:0002094"
    symptom_name     Human-readable symptom name    e.g.  "Dyspnea"
    sensitivity      P(symptom | disease)            e.g.  0.545
    lr_positive      LR+  = Se / Bg                 e.g.  5.45
    lr_negative      LR−  = (1−Se) / (1−Bg)         e.g.  0.50
    frequency_raw    Original annotation string      e.g.  "HP:0040282"

Algorithm
─────────
    1.  Load the four JSON files produced by script 2.
    2.  Compute the background rate for every HPO term:
            Bg[hp_id] = distinct_diseases_with_hp_id / total_distinct_diseases
    3.  For each annotation:
            a.  Skip if disease_id is not resolvable to a DOID  (ORPHA, DECIPHER …)
            b.  Skip if frequency_raw cannot be parsed           (empty / unknown)
            c.  Skip if hp_id is not in symptom_index            (obsolete term)
            d.  Compute LR+ and LR− via lr_calculator.compute_lr()
    4.  Write lr_table.csv.

Usage
─────
    # from exclusively-neo4j/
    uv run 3_compute_lr_table.py
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from utils.lr_calculator import compute_lr, parse_frequency

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).resolve().parent
DATA_DIR      = Path(os.getenv("DATA_DIR", str(SCRIPT_DIR.parent / "data")))
PROCESSED_DIR = DATA_DIR / "processed"

HPOA_JSON      = PROCESSED_DIR / "hpoa_annotations.json"
DISEASE_JSON   = PROCESSED_DIR / "disease_index.json"
OMIM_MAP_JSON  = PROCESSED_DIR / "omim_to_doid.json"
ORPHA_MAP_JSON = PROCESSED_DIR / "orpha_to_doid.json"
SYMPTOM_JSON   = PROCESSED_DIR / "symptom_index.json"
LR_TABLE_CSV   = PROCESSED_DIR / "lr_table.csv"

# CSV output columns (in order)
CSV_COLUMNS = [
    "disease_omim",
    "disease_doid",
    "disease_name",
    "hp_id",
    "symptom_name",
    "sensitivity",
    "lr_positive",
    "lr_negative",
    "frequency_raw",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_json(path: Path, label: str) -> Any:
    if not path.exists():
        logger.error("Required input not found: %s — run script 2 first.", path)
        sys.exit(1)
    logger.info("Loading %s …", label)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _compute_background_rates(
    annotations: list[dict],
) -> dict[str, float]:
    """
    Compute P(symptom | ¬disease) for every HP term in the annotation set.

    Background rate = fraction of ALL diseases in HPOA that carry this HP term.
    A symptom annotated to 800 out of 8 000 diseases has Bg = 0.10.

    Parameters
    ----------
    annotations : list[dict]
        Full list from hpoa_annotations.json (already filtered to phenotype
        aspect and non-negated rows by script 2).

    Returns
    -------
    dict[str, float]
        {hp_id → background_rate}
    """
    # Count distinct diseases per HP term
    hp_to_diseases: dict[str, set[str]] = defaultdict(set)
    all_disease_ids: set[str] = set()

    for ann in annotations:
        hp_to_diseases[ann["hpo_id"]].add(ann["disease_id"])
        all_disease_ids.add(ann["disease_id"])

    total = len(all_disease_ids)
    if total == 0:
        raise ValueError("No disease IDs found in annotations — cannot compute background rates.")

    rates = {
        hp_id: len(diseases) / total
        for hp_id, diseases in hp_to_diseases.items()
    }

    logger.info(
        "  Background rates computed for %d HP terms over %d diseases.",
        len(rates),
        total,
    )
    return rates


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:

    # ── load inputs ───────────────────────────────────────────────────────────
    annotations:   list[dict]     = _load_json(HPOA_JSON,      "hpoa_annotations.json")
    disease_index: dict[str, Any] = _load_json(DISEASE_JSON,   "disease_index.json")
    omim_to_doid:  dict[str, str] = _load_json(OMIM_MAP_JSON,  "omim_to_doid.json")
    orpha_to_doid: dict[str, str] = _load_json(ORPHA_MAP_JSON, "orpha_to_doid.json")
    symptom_index: dict[str, Any] = _load_json(SYMPTOM_JSON,   "symptom_index.json")

    # ── build reverse lookup: doid → disease_name ─────────────────────────────
    doid_to_name: dict[str, str] = {
        doid: info["name"]
        for doid, info in disease_index.items()
    }

    # ── background rates ──────────────────────────────────────────────────────
    logger.info("━━━ Computing background rates ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    background_rates = _compute_background_rates(annotations)

    # ── derive LR table ───────────────────────────────────────────────────────
    logger.info("━━━ Deriving LR table ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    rows: list[dict] = []

    # Skip counters — for the quality report at the end
    n_no_doid     = 0   # OMIM not in omim_to_doid (ORPHA, DECIPHER, unmapped OMIM)
    n_no_hpo      = 0   # HP term not in symptom_index (obsolete)
    n_no_freq     = 0   # frequency_raw is empty or unrecognisable
    n_ok          = 0

    for ann in tqdm(annotations, desc="Computing LRs", unit=" rows", ncols=90):
        disease_id:    str = ann["disease_id"]     # e.g. "OMIM:606391"
        hp_id:         str = ann["hpo_id"]         # e.g. "HP:0002094"
        frequency_raw: str = ann["frequency_raw"]  # e.g. "HP:0040282" | "12/34" | "45%"

        # ── resolve disease ───────────────────────────────────────────────────
        # Try OMIM bridge first, then ORPHA bridge.  DECIPHER and other
        # prefixes have no DO mapping and are counted as unresolvable.
        doid = omim_to_doid.get(disease_id) or orpha_to_doid.get(disease_id)
        if doid is None:
            n_no_doid += 1
            continue

        disease_name = doid_to_name.get(doid, "")

        # ── resolve HPO term ──────────────────────────────────────────────────
        symptom = symptom_index.get(hp_id)
        if symptom is None:
            n_no_hpo += 1
            continue
        symptom_name: str = symptom["name"]

        # ── parse frequency → sensitivity ─────────────────────────────────────
        sensitivity = parse_frequency(frequency_raw)
        if sensitivity is None:
            n_no_freq += 1
            continue

        # ── look up background rate ───────────────────────────────────────────
        # Every hp_id in annotations is guaranteed to be in background_rates
        # because we built the dict from the same annotation list.
        bg = background_rates[hp_id]

        # ── compute LRs ───────────────────────────────────────────────────────
        lr = compute_lr(sensitivity, bg)

        rows.append({
            "disease_omim":  disease_id,
            "disease_doid":  doid,
            "disease_name":  disease_name,
            "hp_id":         hp_id,
            "symptom_name":  symptom_name,
            "sensitivity":   lr.sensitivity,
            "lr_positive":   lr.lr_positive,
            "lr_negative":   lr.lr_negative,
            "frequency_raw": frequency_raw,
        })
        n_ok += 1

    # ── write CSV ─────────────────────────────────────────────────────────────
    logger.info("━━━ Writing lr_table.csv ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)

    if df.empty:
        logger.error(
            "lr_table has 0 rows — the OMIM bridge produced no matches.\n"
            "  Most likely cause: omim_to_doid.json was built from an old\n"
            "  doid.obo run before the MIM:→OMIM: normalisation fix.\n"
            "  Fix: re-run  uv run 2_parse_ontologies.py  first,\n"
            "  then re-run this script."
        )
        sys.exit(1)

    # Deduplicate: if the same (disease_omim, hp_id) pair appears more than
    # once (HPOA has occasional duplicates across evidence sources), keep the
    # row with the highest sensitivity — it is the most specific annotation.
    before_dedup = len(df)
    df = (
        df.sort_values("sensitivity", ascending=False)
          .drop_duplicates(subset=["disease_omim", "hp_id"], keep="first")
          .sort_values(["disease_omim", "hp_id"])
          .reset_index(drop=True)
    )
    n_deduped = before_dedup - len(df)

    df.to_csv(LR_TABLE_CSV, index=False, quoting=csv.QUOTE_NONNUMERIC)

    size_kb = LR_TABLE_CSV.stat().st_size / 1024
    logger.info(
        "  ✓ Wrote lr_table.csv  (%d rows, %.0f KB)",
        len(df),
        size_kb,
    )

    # ── quality report ────────────────────────────────────────────────────────
    logger.info("━━━ Quality report ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    total_in = len(annotations)
    logger.info("  Input annotations        : %8d", total_in)
    logger.info("  ✓ Written to lr_table    : %8d  (%.1f %%)", n_ok,       100 * n_ok      / total_in)
    logger.info("  ✗ No DOID bridge         : %8d  (%.1f %%)", n_no_doid,  100 * n_no_doid / total_in)
    logger.info("  ✗ HP term obsolete       : %8d  (%.1f %%)", n_no_hpo,   100 * n_no_hpo  / total_in)
    logger.info("  ✗ Frequency unparseable  : %8d  (%.1f %%)", n_no_freq,  100 * n_no_freq / total_in)
    logger.info("  Duplicates merged        : %8d", n_deduped)

    # Sensitivity distribution
    logger.info("━━━ Sensitivity distribution ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    bins   = [0.0, 0.05, 0.10, 0.30, 0.50, 0.80, 1.01]
    labels = ["0–5%", "5–10%", "10–30%", "30–50%", "50–80%", "80–100%"]
    df["se_bin"] = pd.cut(df["sensitivity"], bins=bins, labels=labels, right=False)
    dist = df["se_bin"].value_counts().sort_index()
    for label, count in dist.items():
        bar = "█" * int(30 * count / len(df)) if len(df) > 0 else ""
        logger.info("  %-10s  %7d  %s", label, count, bar)

    # LR+ distribution
    logger.info("━━━ LR+ distribution ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lr_stats = df["lr_positive"].describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.99])
    for stat, val in lr_stats.items():
        logger.info("  %-6s  %.4f", stat, val)

    logger.info("━━━ All done ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    main()
