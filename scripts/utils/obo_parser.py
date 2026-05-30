"""
scripts/utils/obo_parser.py
────────────────────────────────────────────────────────────────────────────────
OBO format parser built on pronto.
Handles doid.obo (Disease Ontology) and hp.obo (Human Phenotype Ontology).

Exported functions
──────────────────
    parse_doid(obo_path)  →  (disease_index, omim_to_doid, orpha_to_doid, obsolete_doid_map)
    parse_hpo(obo_path)   →  symptom_index

Key note on MIM: vs OMIM:
──────────────────────────
    The Disease Ontology stores OMIM cross-references with the prefix "MIM:"
    (e.g.  xref: MIM:606391).  The phenotype.hpoa file uses "OMIM:" as its
    database_id prefix (e.g.  OMIM:606391).  All "MIM:XXXXXX" xref IDs are
    normalised to "OMIM:XXXXXX" so that the omim_to_doid bridge keys match
    the HPOA disease IDs exactly.  Without this normalisation the bridge
    produces zero matches.

Gap 1 fix:
    Obsolete doid.obo terms are processed for their xrefs. Their xrefs are
    mapped to the active DOID they were replaced by (following the replaced_by
    chain until a non-obsolete term is found). This eliminates ~6,680 missing
    edges caused by HPOA annotations that reference OMIM/ORPHA IDs whose only
    DOID xref was on an obsolete (and therefore non-loaded) term.
"""

from __future__ import annotations

import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pronto  # pronto >=2.7.x

logger = logging.getLogger(__name__)


def load_ontology(path: Path) -> pronto.Ontology:
    """Load a pronto Ontology, suppressing warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pronto.Ontology(str(path))


def _normalise_omim(raw_id: str) -> str:
    """Normalise MIM: → OMIM: so bridge keys match HPOA database_id values."""
    if raw_id.startswith("MIM:"):
        return "OMIM:" + raw_id[4:]
    return raw_id


def _follow_replaced_by(term: pronto.Term, ont: pronto.Ontology, depth: int = 0) -> str | None:
    """
    Follow the replaced_by chain until a non-obsolete term is found.
    Returns the active DOID string, or None if the chain cannot be resolved.
    Depth limit of 5 prevents infinite loops on circular references.
    """
    if depth > 5:
        return None
    for replaced in term.replaced_by:
        target_id = str(replaced.id) if hasattr(replaced, "id") else str(replaced)
        target = ont.get(target_id)
        if target is None:
            continue
        if not target.obsolete:
            return target.id
        # Recurse if the replacement is itself obsolete
        result = _follow_replaced_by(target, ont, depth + 1)
        if result:
            return result
    return None


# ── Disease Ontology ──────────────────────────────────────────────────────────

def parse_doid(
    obo_path: Path, ont: pronto.Ontology | None = None
) -> tuple[dict[str, Any], dict[str, str], dict[str, str], dict[str, str]]:
    """
    Parse the Disease Ontology OBO file.

    Returns
    -------
    disease_index : dict
        Active (non-obsolete) terms: { doid → {name, icd, omim, orpha} }
    omim_to_doid : dict
        { omim_curie → active_doid }  — includes xrefs from obsolete terms
        resolved to their replacement active DOID.
    orpha_to_doid : dict
        { orpha_curie → active_doid }  — same resolution.
    obsolete_doid_map : dict
        { old_doid → active_doid }  — alt_id and replaced_by mappings for audit.
    """
    logger.info("Loading Disease Ontology from %s …", obo_path)

    if ont is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ont = pronto.Ontology(str(obo_path))

    logger.info("  Ontology loaded. Iterating terms …")

    disease_index: dict[str, Any] = {}
    omim_to_doid: dict[str, str] = {}
    orpha_to_doid: dict[str, str] = {}
    obsolete_doid_map: dict[str, str] = {}

    # First pass: collect all active terms into disease_index and build
    # alt_id → active DOID portion of obsolete_doid_map.
    for term in ont.terms():
        if term.obsolete or not term.name:
            continue

        doid: str = term.id
        icd_refs: list[str] = []
        omim_refs: list[str] = []
        orpha_refs: list[str] = []

        for xref in term.xrefs:
            xid: str = xref.id
            if xid.startswith("ICD"):
                icd_refs.append(xid)
            elif xid.startswith("MIM:") or xid.startswith("OMIM:"):
                omim_refs.append(_normalise_omim(xid))
            elif xid.startswith("ORDO:"):
                orpha_refs.append("ORPHA:" + xid[5:])

        disease_index[doid] = {
            "name":  term.name,
            "icd":   icd_refs,
            "omim":  omim_refs,
            "orpha": orpha_refs,
        }

        # Bridge maps for active terms
        for omim_id in omim_refs:
            if omim_id not in omim_to_doid:
                omim_to_doid[omim_id] = doid
            else:
                logger.debug("Duplicate OMIM mapping: %s already maps to %s — skipping %s",
                             omim_id, omim_to_doid[omim_id], doid)

        for orpha_id in orpha_refs:
            if orpha_id not in orpha_to_doid:
                orpha_to_doid[orpha_id] = doid

        # alt_id entries: old DOIDs that were merged into this active term
        for alt_id in getattr(term, "alternate_ids", []):
            alt_str = str(alt_id)
            if alt_str.startswith("DOID:"):
                obsolete_doid_map[alt_str] = doid

    # Second pass: obsolete terms — resolve their xrefs to active replacements.
    n_obsolete_resolved = 0
    n_obsolete_unresolved = 0
    for term in ont.terms():
        if not term.obsolete:
            continue

        # Build replaced_by map entry
        active_target = _follow_replaced_by(term, ont)
        if active_target:
            obsolete_doid_map[term.id] = active_target
        else:
            # No replacement found; cannot resolve this term's xrefs
            n_obsolete_unresolved += 1
            continue

        n_obsolete_resolved += 1

        # Collect xrefs from the obsolete term
        omim_refs = []
        orpha_refs = []
        for xref in term.xrefs:
            xid = xref.id
            if xid.startswith("MIM:") or xid.startswith("OMIM:"):
                omim_refs.append(_normalise_omim(xid))
            elif xid.startswith("ORDO:"):
                orpha_refs.append("ORPHA:" + xid[5:])

        # Map obsolete term's xrefs to the active replacement DOID
        # Only add if not already present from an active term (active takes priority)
        for omim_id in omim_refs:
            if omim_id not in omim_to_doid:
                omim_to_doid[omim_id] = active_target

        for orpha_id in orpha_refs:
            if orpha_id not in orpha_to_doid:
                orpha_to_doid[orpha_id] = active_target

    logger.info(
        "  Parsed %d active diseases | %d OMIM→DOID | %d ORPHA→DOID | "
        "%d obsolete resolved | %d obsolete unresolved | %d alt_id entries.",
        len(disease_index),
        len(omim_to_doid),
        len(orpha_to_doid),
        n_obsolete_resolved,
        n_obsolete_unresolved,
        sum(1 for v in obsolete_doid_map.values() if v != ""),
    )

    sample = list(omim_to_doid.items())[:4]
    for omim_key, doid_val in sample:
        if doid_val in disease_index:
            logger.info("    bridge sample: %s → %s  (%s)",
                        omim_key, doid_val, disease_index[doid_val]["name"])

    return disease_index, omim_to_doid, orpha_to_doid, obsolete_doid_map


# ── Human Phenotype Ontology ──────────────────────────────────────────────────

def parse_hpo(obo_path: Path, ont: pronto.Ontology | None = None) -> dict[str, Any]:
    """
    Parse the Human Phenotype Ontology OBO file.

    Returns symptom_index: { hp_id → {name, synonyms} }
    """
    logger.info("Loading HPO from %s …", obo_path)

    if ont is None:
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

        hp_id: str = term.id
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


def parse_all(
    doid_path: Path,
    hpo_path: Path,
) -> tuple[
    tuple[dict[str, Any], dict[str, str], dict[str, str], dict[str, str]],
    dict[str, Any],
]:
    """
    Load both ontologies in parallel (I/O-bound), then parse each.

    Returns
    -------
    (doid_result, hpo_result)
        doid_result : (disease_index, omim_to_doid, orpha_to_doid, obsolete_doid_map)
        hpo_result  : symptom_index
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_doid = executor.submit(load_ontology, doid_path)
        fut_hpo  = executor.submit(load_ontology, hpo_path)
        doid_ont = fut_doid.result()
        hpo_ont  = fut_hpo.result()

    doid_result = parse_doid(doid_path, ont=doid_ont)
    hpo_result  = parse_hpo(hpo_path,  ont=hpo_ont)
    return doid_result, hpo_result
