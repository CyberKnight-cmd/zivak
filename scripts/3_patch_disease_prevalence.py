#!/usr/bin/env python3
"""
scripts/3_patch_disease_prevalence.py
--------------------------------------------------------------------------------
Patch 'prevalence' and 'prevalence_source' onto every entry in
disease_index.json using ORPHA prevalence data, ICD chapter defaults,
OMIM prefix estimates, and the orpha_to_doid bridge.

Priority order:
  1. orpha              — real (non-fallback) ORPHA prevalence
  2. icd_chapter        — ICD10 chapter-based population estimate
  3. orpha_unmapped     — has ORPHA xref but no usable prevalence
  4. omim_prefix        — has OMIM xref; estimate from OMIM number prefix
  5. ultra_rare_default — no ICD, ORPHA, or OMIM xref at all
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
PROCESSED = DATA_DIR / "processed"

DISEASE_INDEX    = PROCESSED / "disease_index.json"
ORPHA_PREVALENCE = PROCESSED / "orpha_prevalence.json"
ORPHA_TO_DOID    = PROCESSED / "orpha_to_doid.json"

ORPHA_FALLBACK = 0.00001

ICD_CHAPTER_PREVALENCE: dict[str, float] = {
    "J": 0.05,   # Respiratory
    "I": 0.05,   # Cardiovascular
    "E": 0.03,   # Endocrine
    "K": 0.03,   # Digestive
    "M": 0.03,   # Musculoskeletal
    "F": 0.02,   # Mental
    "G": 0.02,   # Neurological
    "N": 0.02,   # Genitourinary
    "L": 0.02,   # Skin
    "D": 0.01,   # Blood/Immune
    "C": 0.01,   # Neoplasms
    "H": 0.01,   # Eye/Ear
    "A": 0.01,   # Infectious
    "B": 0.01,   # Infectious (parasitic)
    "Q": 0.005,  # Congenital
    "R": 0.005,  # Symptoms/Signs
}

OMIM_PREFIX_PREVALENCE: dict[str, float] = {
    "1": 0.0001,    # 1xxxxx — phenotypic series, historically more common
    "2": 0.000001,
    "3": 0.000001,
    "4": 0.000001,
    "5": 0.000001,
    "6": 0.000001,
}


def main() -> None:
    for path in (DISEASE_INDEX, ORPHA_PREVALENCE, ORPHA_TO_DOID):
        if not path.exists():
            print(f"ERROR: required file not found: {path}", file=sys.stderr)
            sys.exit(1)

    disease_index    = json.loads(DISEASE_INDEX.read_text(encoding="utf-8"))
    orpha_prevalence = json.loads(ORPHA_PREVALENCE.read_text(encoding="utf-8"))
    orpha_to_doid    = json.loads(ORPHA_TO_DOID.read_text(encoding="utf-8"))

    doid_to_prevalence: dict[str, float] = {}
    for orpha_id, doid in orpha_to_doid.items():
        if orpha_id in orpha_prevalence and doid not in doid_to_prevalence:
            doid_to_prevalence[doid] = orpha_prevalence[orpha_id]

    n_orpha       = 0
    n_icd_chapter = 0
    n_unmapped    = 0
    n_omim_prefix = 0
    n_ultra_rare  = 0

    for doid, entry in disease_index.items():
        orpha_val = doid_to_prevalence.get(doid)

        if orpha_val is not None and orpha_val != ORPHA_FALLBACK:
            entry["prevalence"]        = orpha_val
            entry["prevalence_source"] = "orpha"
            n_orpha += 1

        elif entry.get("icd"):
            icd_code = entry["icd"][0]
            chapter  = icd_code.split(":")[-1][0].upper()
            entry["prevalence"]        = ICD_CHAPTER_PREVALENCE.get(chapter, ORPHA_FALLBACK)
            entry["prevalence_source"] = "icd_chapter"
            n_icd_chapter += 1

        elif entry.get("orpha"):
            entry["prevalence"]        = ORPHA_FALLBACK
            entry["prevalence_source"] = "orpha_unmapped"
            n_unmapped += 1

        elif entry.get("omim"):
            omim_num    = entry["omim"][0].replace("OMIM:", "").strip()
            first_digit = omim_num[0] if omim_num else "9"
            entry["prevalence"]        = OMIM_PREFIX_PREVALENCE.get(first_digit, 0.000001)
            entry["prevalence_source"] = "omim_prefix"
            n_omim_prefix += 1

        else:
            entry["prevalence"]        = 0.000001
            entry["prevalence_source"] = "ultra_rare_default"
            n_ultra_rare += 1

    tmp = DISEASE_INDEX.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(disease_index, fh, ensure_ascii=False, indent=2)
        tmp.replace(DISEASE_INDEX)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    total = n_orpha + n_icd_chapter + n_unmapped + n_omim_prefix + n_ultra_rare
    print("Prevalence source breakdown:")
    print(f"  orpha              : {n_orpha} diseases")
    print(f"  icd_chapter        : {n_icd_chapter} diseases")
    print(f"  orpha_unmapped     : {n_unmapped} diseases")
    print(f"  omim_prefix        : {n_omim_prefix} diseases")
    print(f"  ultra_rare_default : {n_ultra_rare} diseases")
    print(f"  Total              : {total:,}")
    if total != 12127:
        print(f"WARNING: expected 12127 entries, got {total}", file=sys.stderr)


if __name__ == "__main__":
    main()
