#!/usr/bin/env python3
"""
scripts/1_parse_orpha_prevalence.py
--------------------------------------------------------------------------------
Parse en_product9_prev.xml (Orphanet prevalence data) and produce
data/processed/orpha_prevalence.json.

Output format:
    { "ORPHA:166024": 0.0000005, "ORPHA:413": 0.00005, ... }

Uses iterparse to avoid loading the full XML into memory.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
RAW_DIR    = DATA_DIR / "raw"
PROCESSED  = DATA_DIR / "processed"

XML_FILE   = RAW_DIR  / "en_product9_prev.xml"
OUTPUT     = PROCESSED / "orpha_prevalence.json"

PREVALENCE_CLASS_MAP: dict[str, float] = {
    ">1 / 1 000":           0.005,
    "1-9 / 1 000":          0.005,
    "1-9 / 10 000":         0.0005,
    "1-9 / 100 000":        0.00005,
    "1-9 / 1 000 000":      0.000005,
    "<1 / 1 000 000":       0.0000005,
}
FALLBACK = 0.00001


def _parse_disorder(disorder_el: ET.Element) -> tuple[str, float, bool]:
    """
    Extract ORPHA key and numeric prevalence from a <Disorder> element.
    Returns (key, value, used_fallback).
    """
    code_el = disorder_el.find("OrphaCode")
    if code_el is None or not code_el.text:
        return "", FALLBACK, True
    key = f"ORPHA:{code_el.text.strip()}"

    prev_list = disorder_el.find("PrevalenceList")
    if prev_list is None:
        return key, FALLBACK, True

    # Collect all Point prevalence entries
    point_entries: list[ET.Element] = []
    for prev in prev_list.findall("Prevalence"):
        ptype_el = prev.find("PrevalenceType/Name")
        if ptype_el is not None and ptype_el.text and ptype_el.text.strip() == "Point prevalence":
            point_entries.append(prev)

    if not point_entries:
        return key, FALLBACK, True

    # Prefer Worldwide geographic scope; otherwise take the first available
    chosen = None
    for entry in point_entries:
        geo_el = entry.find("PrevalenceGeographic/Name")
        if geo_el is not None and geo_el.text and geo_el.text.strip() == "Worldwide":
            chosen = entry
            break
    if chosen is None:
        chosen = point_entries[0]

    cls_el = chosen.find("PrevalenceClass/Name")
    if cls_el is None or not cls_el.text:
        return key, FALLBACK, True

    cls_str = cls_el.text.strip()
    value = PREVALENCE_CLASS_MAP.get(cls_str)
    if value is None:
        return key, FALLBACK, True

    return key, value, False


def main() -> None:
    if not XML_FILE.exists():
        print(f"ERROR: XML file not found: {XML_FILE}", file=sys.stderr)
        sys.exit(1)

    result: dict[str, float] = {}
    n_total    = 0
    n_real     = 0
    n_fallback = 0

    # Build complete <Disorder> elements via iterparse to avoid full DOM load
    context = ET.iterparse(str(XML_FILE), events=("end",))
    for event, elem in context:
        if elem.tag != "Disorder":
            continue
        n_total += 1
        key, value, used_fallback = _parse_disorder(elem)
        if not key:
            elem.clear()
            continue
        result[key] = value
        if used_fallback:
            n_fallback += 1
        else:
            n_real += 1
        elem.clear()

    PROCESSED.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print(f"Total disorders parsed:           {n_total}")
    print(f"With real Point prevalence:       {n_real}")
    print(f"Using fallback ({FALLBACK}):    {n_fallback}")
    if n_total > 0:
        pct = 100.0 * n_fallback / n_total
        print(f"Fallback percentage:              {pct:.1f}%")
        if pct > 80:
            print("WARNING: >80% fallback — Point prevalence filter may not be matching!", file=sys.stderr)
    print(f"Output written to:                {OUTPUT}")


if __name__ == "__main__":
    main()
