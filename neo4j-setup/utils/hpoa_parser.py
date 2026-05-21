"""
scripts/utils/hpoa_parser.py
────────────────────────────────────────────────────────────────────────────────
Parser for the HPO annotation file ``phenotype.hpoa``.

The file is a tab-separated text file with a variable number of comment lines
(starting with ``#``) at the top, followed by a single header row, then data
rows.  Two header formats are handled transparently:

    New (2022+)   database_id  disease_name  qualifier  hpo_id  reference …
    Legacy        #DatabaseID  DiseaseName   Qualifier  HPO_ID  DB_Reference …

Exported function
─────────────────
    parse_hpoa(hpoa_path)  →  list[dict]
"""

from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Columns that must be present after normalisation
_REQUIRED = {"database_id", "qualifier", "hpo_id", "frequency", "aspect"}

# Map legacy / variant column names → canonical names
_LEGACY_MAP: dict[str, str] = {
    "databaseid":   "database_id",
    "#databaseid":  "database_id",
    "hpo_id":       "hpo_id",       # already fine, but explicit
    "diseasename":  "disease_name",
    "db_reference": "reference",
    "curation_date":"biocuration",
    "biocuration":  "biocuration",
}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case, strip leading ``#``, and apply legacy renames."""
    new_names: dict[str, str] = {}
    for col in df.columns:
        key = col.strip().lstrip("#").lower().replace(" ", "_").replace("-", "_")
        new_names[col] = _LEGACY_MAP.get(key, key)
    return df.rename(columns=new_names)


def parse_hpoa(hpoa_path: Path) -> list[dict[str, Any]]:
    """
    Parse ``phenotype.hpoa`` into a list of annotation records.

    Filtering applied
    -----------------
    * Only rows where ``aspect == "P"`` (phenotype) are kept.
      Rows with aspect ``"I"`` (inheritance) and ``"C"`` (clinical modifier)
      are discarded — they don't represent observable symptoms.
    * Rows where ``qualifier == "NOT"`` are discarded (negated associations).
    * Rows with an empty ``hpo_id`` are discarded.

    Parameters
    ----------
    hpoa_path : Path
        Absolute or relative path to ``phenotype.hpoa``.

    Returns
    -------
    list[dict]
        Each dict has three keys::

            {
                "disease_id":    "OMIM:606391",   # source ID from the file
                "hpo_id":        "HP:0002094",
                "frequency_raw": "HP:0040281",    # raw string — may be "", HP
                                                  # term, ratio, or percentage
            }

        ``frequency_raw`` is intentionally left as-is; script 3
        (``3_compute_lr_table.py``) will interpret it.

    Raises
    ------
    ValueError
        If required columns are not found after normalisation.
    FileNotFoundError
        If the file does not exist.
    """
    if not hpoa_path.exists():
        raise FileNotFoundError(f"HPOA file not found: {hpoa_path}")

    logger.info("Parsing HPOA from %s …", hpoa_path)

    # ── Step 1: strip comment lines, keep header + data ──────────────────────
    data_lines: list[str] = []
    with open(hpoa_path, "r", encoding="utf-8") as fh:
        for line in fh:
            # Comment lines begin with '#' *except* for the legacy header line
            # which also starts with '#' but contains tab-separated column names.
            # Heuristic: a comment line contains no tab → skip it.
            # A header/data line always contains at least one tab.
            if line.startswith("#") and "\t" not in line:
                continue
            data_lines.append(line)

    if not data_lines:
        raise ValueError(f"No data rows found in {hpoa_path}")

    # ── Step 2: parse as TSV ──────────────────────────────────────────────────
    df = pd.read_csv(
        StringIO("".join(data_lines)),
        sep="\t",
        dtype=str,          # keep everything as strings; no unexpected coercions
        low_memory=False,
    )

    # ── Step 3: normalise column names ────────────────────────────────────────
    df = _normalise_columns(df)

    missing = _REQUIRED - set(df.columns)
    if missing:
        raise ValueError(
            f"HPOA file is missing expected columns after normalisation: {missing}\n"
            f"Columns found: {list(df.columns)}"
        )

    # ── Step 4: filter ────────────────────────────────────────────────────────

    # Keep only phenotype aspect
    df = df[df["aspect"].fillna("").str.strip() == "P"].copy()

    # Drop negated associations  (qualifier == "NOT")
    df = df[df["qualifier"].fillna("").str.strip().str.upper() != "NOT"].copy()

    # Drop rows without an HP ID
    df = df[df["hpo_id"].notna() & (df["hpo_id"].str.strip() != "")].copy()

    # Drop rows without a disease ID
    df = df[df["database_id"].notna() & (df["database_id"].str.strip() != "")].copy()

    # ── Step 5: normalise frequency ───────────────────────────────────────────
    df["frequency"] = df["frequency"].fillna("").str.strip()

    # ── Step 6: build output records ─────────────────────────────────────────
    records: list[dict[str, Any]] = [
        {
            "disease_id":    row["database_id"].strip(),
            "hpo_id":        row["hpo_id"].strip(),
            "frequency_raw": row["frequency"],
        }
        for _, row in df.iterrows()
    ]

    n_diseases  = len({r["disease_id"] for r in records})
    n_omim      = len({r["disease_id"] for r in records if r["disease_id"].startswith("OMIM:")})
    n_orpha     = len({r["disease_id"] for r in records if r["disease_id"].startswith("ORPHA:")})

    logger.info(
        "  Parsed %d phenotype annotations | %d unique diseases "
        "(%d OMIM, %d ORPHA, rest other).",
        len(records),
        n_diseases,
        n_omim,
        n_orpha,
    )
    return records
