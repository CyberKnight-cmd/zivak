"""
scripts/utils/obo_parser.py
────────────────────────────────────────────────────────────────────────────────
OBO format parser built on pronto.
Handles doid.obo (Disease Ontology) and hp.obo (Human Phenotype Ontology).

Exported functions
──────────────────
    parse_doid(obo_path)  →  (disease_index, omim_to_doid)
    parse_hpo(obo_path)   →  symptom_index

Key note on MIM: vs OMIM:
──────────────────────────
    The Disease Ontology stores OMIM cross-references with the prefix "MIM:"
    (e.g.  xref: MIM:606391).  The phenotype.hpoa file uses "OMIM:" as its
    database_id prefix (e.g.  OMIM:606391).  All "MIM:XXXXXX" xref IDs are
    normalised to "OMIM:XXXXXX" so that the omim_to_doid bridge keys match
    the HPOA disease IDs exactly.  Without this normalisation the bridge
    produces zero matches.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import pronto  # pronto >=2.7.x

logger = logging.getLogger(__name__)


def _normalise_omim(raw_id: str) -> str:
    """
    Normalise a raw xref ID to the canonical "OMIM:XXXXXX" form.

    Disease Ontology xrefs use "MIM:XXXXXX"; HPOA uses "OMIM:XXXXXX".
    Both are mapped to "OMIM:XXXXXX" so bridge keys are uniform.
    """
    if raw_id.startswith("MIM:"):
        return "OMIM:" + raw_id[4:]
    return raw_id


# ── Disease Ontology ──────────────────────────────────────────────────────────

def parse_doid(obo_path: Path) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    """
    Parse the Disease Ontology OBO file.

    Parameters
    ----------
    obo_path : Path
        Absolute or relative path to ``doid.obo``.

    Returns
    -------
    disease_index : dict
        ``{ doid_curie: {"name": str, "icd": [str], "omim": [str], "orpha": [str]} }``
        e.g. ``{"DOID:3083": {"name": "chronic obstructive pulmonary disease",
                              "icd":  ["ICD10CM:J44"],
                              "omim": ["OMIM:606391"],
                              "orpha": []}}``

    omim_to_doid : dict
        ``{ omim_curie: doid_curie }``
        e.g. ``{"OMIM:606391": "DOID:3083"}``
        This is the bridge that replaces MONDO for HPOA ↔ DO resolution.
        When one OMIM ID appears on multiple non-obsolete DO terms the
        first encountered mapping is kept and a DEBUG message is emitted.

    orpha_to_doid : dict
        ``{ orpha_curie: doid_curie }``
        e.g. ``{"ORPHA:586": "DOID:0050425"}``
        Built from ``ORPHA:`` xrefs in doid.obo; bridges ORPHA disease IDs
        from HPOA to Disease Ontology terms.
    """
    logger.info("Loading Disease Ontology from %s …", obo_path)

    # Pronto emits UserWarnings for unknown OBO tags — suppress them so the
    # caller's logs stay clean.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ont = pronto.Ontology(str(obo_path))

    logger.info("  Ontology loaded. Iterating terms …")

    disease_index: dict[str, Any] = {}
    omim_to_doid: dict[str, str] = {}
    orpha_to_doid: dict[str, str] = {}
    n_skipped = 0

    for term in ont.terms():
        # Skip obsolete terms and terms without a name (both are expected in DO)
        if term.obsolete or not term.name:
            n_skipped += 1
            continue

        doid: str = term.id          # e.g. "DOID:0001816"
        icd_refs: list[str] = []
        omim_refs: list[str] = []
        orpha_refs: list[str] = []

        for xref in term.xrefs:
            xid: str = xref.id       # e.g. "ICD10CM:C49.0" | "MIM:612600" | "ORPHA:586"
            if xid.startswith("ICD"):
                icd_refs.append(xid)
            elif xid.startswith("MIM:") or xid.startswith("OMIM:"):
                # Normalise MIM: → OMIM: so keys match HPOA database_id values
                normalised = _normalise_omim(xid)
                omim_refs.append(normalised)
            elif xid.startswith("ORDO:"):
                # doid.obo uses "ORDO:XXXXXX"; HPOA uses "ORPHA:XXXXXX"
                # The numeric IDs are identical — normalise the prefix.
                orpha_refs.append("ORPHA:" + xid[5:])

        disease_index[doid] = {
            "name":  term.name,
            "icd":   icd_refs,
            "omim":  omim_refs,     # stored as normalised OMIM: CURIEs
            "orpha": orpha_refs,    # stored as ORPHA: CURIEs
        }

        # Build OMIM → DOID bridge (normalised keys)
        for omim_id in omim_refs:
            if omim_id in omim_to_doid:
                logger.debug(
                    "Duplicate OMIM mapping: %s already maps to %s — skipping %s",
                    omim_id, omim_to_doid[omim_id], doid,
                )
            else:
                omim_to_doid[omim_id] = doid

        # Build ORPHA → DOID bridge
        for orpha_id in orpha_refs:
            if orpha_id not in orpha_to_doid:
                orpha_to_doid[orpha_id] = doid

    logger.info(
        "  Parsed %d diseases | %d OMIM→DOID mappings | %d ORPHA→DOID mappings | %d terms skipped (obsolete/unnamed).",
        len(disease_index),
        len(omim_to_doid),
        len(orpha_to_doid),
        n_skipped,
    )
    # Print a few bridge entries so you can visually verify the prefix is right
    sample = list(omim_to_doid.items())[:4]
    for omim_key, doid_val in sample:
        logger.info("    bridge sample: %s → %s  (%s)",
                    omim_key, doid_val, disease_index[doid_val]["name"])
    return disease_index, omim_to_doid, orpha_to_doid


# ── Human Phenotype Ontology ──────────────────────────────────────────────────

def parse_hpo(obo_path: Path) -> dict[str, Any]:
    """
    Parse the Human Phenotype Ontology OBO file.

    Parameters
    ----------
    obo_path : Path
        Absolute or relative path to ``hp.obo``.

    Returns
    -------
    symptom_index : dict
        ``{ hp_id_curie: {"name": str, "synonyms": [str]} }``
        e.g. ``{"HP:0002094": {"name": "Dyspnea",
                               "synonyms": ["Breathlessness",
                                            "Difficulty breathing",
                                            "Shortness of breath"]}}``

        Synonyms are stored as plain strings (the ``.description`` field from
        pronto ``Synonym`` objects).  All synonym scopes (EXACT, BROAD,
        NARROW, RELATED) are included — the more terms Qdrant can match the
        better.
    """
    logger.info("Loading HPO from %s …", obo_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ont = pronto.Ontology(str(obo_path))

    logger.info("  Ontology loaded. Iterating terms …")

    symptom_index: dict[str, Any] = {}
    n_skipped = 0

    for term in ont.terms():
        if term.obsolete or not term.name:
            n_skipped += 1
            continue

        hp_id: str = term.id         # e.g. "HP:0002094"

        # Collect all synonym descriptions regardless of scope
        synonyms: list[str] = [syn.description for syn in term.synonyms]

        symptom_index[hp_id] = {
            "name":     term.name,
            "synonyms": synonyms,
        }

    logger.info(
        "  Parsed %d HPO terms | %d terms skipped (obsolete/unnamed).",
        len(symptom_index),
        n_skipped,
    )
    return symptom_index
