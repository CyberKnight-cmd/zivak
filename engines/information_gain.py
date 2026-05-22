"""
Information Gain Engine — pure math, zero LLM calls, zero I/O.

Ranks diagnostic tests by Expected Information Gain (EIG) over the current
Bayesian differential so the QuestionSelectorAgent only sees the 5 highest-
value tests rather than every available HPO symptom term.

EIG formula for test T given differential P = {P(D_i)}:
  Simulate positive: P_pos[D_i] ∝ P(D_i) × LR+(D_i), normalize → Z_pos
  Simulate negative: P_neg[D_i] ∝ P(D_i) × LR-(D_i), normalize → Z_neg
  P(positive) ≈ Z_pos / (Z_pos + Z_neg)
  EIG(T) = H_before − P(+)×H(P_pos) − P(−)×H(P_neg)

LR derivation from graph edges:
  RULES_IN  edge lr:  LR+(D) = lr,      LR-(D) = 1/lr
  RULES_OUT edge lr:  LR+(D) = lr,      LR-(D) = 1/lr
  No edge for D:      LR+(D) = 1.0,     LR-(D) = 1.0  (neutral — no shift)
"""

import math
from typing import Dict, List

_LR_FLOOR = 1e-9  # guard against log(0) if a malformed LR of 0 arrives


def _entropy(probs: List[float]) -> float:
    """Shannon entropy in bits (log base 2)."""
    return -sum(p * math.log2(p) for p in probs if p > _LR_FLOOR)


def _simulate(disease_probs: Dict[str, float], lr_map: Dict[str, float]) -> List[float]:
    """Apply a likelihood ratio map to a prior and return the normalized posterior."""
    unnorm = {name: prob * lr_map.get(name, 1.0) for name, prob in disease_probs.items()}
    z = sum(unnorm.values())
    if z <= 0:
        n = len(unnorm)
        return [1.0 / n] * n
    return [v / z for v in unnorm.values()]


def rank_by_eig(
    differential: List[Dict],
    test_lr_map: Dict[str, List[Dict]],
    top_n: int = 5,
) -> List[str]:
    """
    Rank tests by Expected Information Gain and return the top_n test IDs.

    Args:
        differential:  Current disease list from DiagnosticState — each entry
                       must have 'name' and 'probability'.
        test_lr_map:   Mapping of test_id → list of edge dicts from Neo4j.
                       Each edge: {disease, relationship, lr}.
                       Only diseases present in the differential are expected
                       (caller should pass scoped edges).
        top_n:         Number of top-ranked test IDs to return.

    Returns:
        List of test_id strings sorted by EIG descending (highest first),
        length ≤ top_n. Returns all test IDs if len(test_lr_map) ≤ top_n.
    """
    if not differential or not test_lr_map:
        return list(test_lr_map.keys())[:top_n]

    # Keyed by lowercase name for case-insensitive matching against edge disease names.
    disease_probs: Dict[str, float] = {
        d["name"].lower(): d["probability"]
        for d in differential
        if d.get("probability", 0) > 0
    }
    if not disease_probs:
        return list(test_lr_map.keys())[:top_n]

    h_before = _entropy(list(disease_probs.values()))

    scores: List[tuple] = []

    for test_id, edges in test_lr_map.items():
        # Build LR maps for both outcomes from graph edges.
        lr_pos: Dict[str, float] = {}
        lr_neg: Dict[str, float] = {}

        for e in edges:
            name = e["disease"].lower()
            lr   = max(float(e.get("lr", 1.0)), _LR_FLOOR)
            if e["relationship"] == "RULES_IN":
                lr_pos[name] = lr
                lr_neg[name] = 1.0 / lr
            else:  # RULES_OUT
                lr_pos[name] = lr
                lr_neg[name] = 1.0 / lr

        # Z values are the unnormalized sums — their ratio approximates P(T=+).
        unnorm_pos = {n: p * lr_pos.get(n, 1.0) for n, p in disease_probs.items()}
        unnorm_neg = {n: p * lr_neg.get(n, 1.0) for n, p in disease_probs.items()}
        z_pos = sum(unnorm_pos.values())
        z_neg = sum(unnorm_neg.values())
        total = z_pos + z_neg

        if total <= 0:
            scores.append((test_id, 0.0))
            continue

        p_positive = z_pos / total
        p_negative = z_neg / total

        post_pos = [v / z_pos for v in unnorm_pos.values()] if z_pos > 0 else []
        post_neg = [v / z_neg for v in unnorm_neg.values()] if z_neg > 0 else []

        h_pos = _entropy(post_pos) if post_pos else 0.0
        h_neg = _entropy(post_neg) if post_neg else 0.0

        eig = h_before - (p_positive * h_pos + p_negative * h_neg)
        scores.append((test_id, eig))

    scores.sort(key=lambda x: -x[1])
    return [test_id for test_id, _ in scores[:top_n]]
