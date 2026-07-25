"""
Differential Engine - Bayesian probability updates
No LLM required - pure mathematics
"""

import logging
import math
from typing import Dict, List, Optional
from copy import deepcopy

logger = logging.getLogger(__name__)

# Probability floor for rare disease seeder (blueprint spec: 2% minimum prior)
RARE_DISEASE_FLOOR = 0.02
MIN_LR = 1e-10  # Guard against log(0) if a badly-formed LR of 0 arrives
DEFAULT_MAX_NEW_CANDIDATES = 8  # Phase-1 expansion cap per event


def _disease_key(d: Dict) -> str:
    """Stable identity key — prefer disease_id, fall back to normalized name."""
    did = d.get("disease_id")
    return f"id:{did}" if did else f"name:{d['name'].lower().strip()}"


class DifferentialEngine:
    """
    Maintains and updates disease probability distribution using Bayes' theorem.

    Works in log-probability space to prevent floating-point underflow when
    many sequential LR updates push rare-disease probabilities toward zero.

    Core operation: log P_new = log P_old + log(LR), then log-sum-exp normalize.

    Each disease dict also tracks `cum_log_lr` — the running sum of every
    log(LR) it has ever received. This lets `expand()` fairly introduce a
    disease discovered mid-session by replaying the same evidence history
    onto it, instead of injecting it at an arbitrary probability.
    """

    def __init__(self):
        self.differential: List[Dict] = []
        self.evidence_history: List[Dict] = []

    # ------------------------------------------------------------------ #
    #  Prior computation (shared by initialize() and expand())            #
    # ------------------------------------------------------------------ #

    def _floor_normalize(self, diseases: List[Dict]) -> None:
        """
        Compute raw_score = specificity * prevalence for each disease, apply
        the rare-disease floor, and renormalize so probabilities sum to 1.

        Mutates each dict in place, setting 'raw_score' and 'probability'.
        Does NOT set 'log_prob' or touch 'cum_log_lr' — callers combine
        this prior with cum_log_lr themselves (see initialize() vs expand()).
        """
        for d in diseases:
            d["raw_score"] = d["specificity"] * d["prevalence"]

        total = sum(d["raw_score"] for d in diseases)
        if total <= 0:
            # Degenerate case (shouldn't happen with valid specificity/prevalence) —
            # fall back to a uniform prior rather than dividing by zero.
            for d in diseases:
                d["probability"] = 1.0 / len(diseases)
            return

        for d in diseases:
            d["probability"] = max(d["raw_score"] / total, RARE_DISEASE_FLOOR)

        total2 = sum(d["probability"] for d in diseases)
        for d in diseases:
            d["probability"] /= total2

    def initialize(self, diseases: List[Dict]) -> List[Dict]:
        """
        Create initial differential from symptom-disease matches.

        Initial probability = (specificity × prevalence) / total,
        floored at RARE_DISEASE_FLOOR so rare diseases can never be
        eliminated from the differential by a single update.

        Args:
            diseases: List of dicts with name, specificity, prevalence

        Returns:
            Sorted list of diseases with probability and log_prob fields
        """
        if not diseases:
            return []

        self._floor_normalize(diseases)
        for d in diseases:
            d["log_prob"] = math.log(d["probability"])
            d["cum_log_lr"] = 0.0

        self.differential = sorted(diseases, key=lambda x: x["probability"], reverse=True)
        return self.differential

    # ------------------------------------------------------------------ #
    #  Per-turn Bayesian update                                            #
    # ------------------------------------------------------------------ #

    def update(
        self,
        evidence: Dict,
        test_id: Optional[str] = None,
        polarity: Optional[str] = None,
    ) -> List[Dict]:
        """
        Update probabilities using Bayes' theorem in log space.

        log P_new = log P_old + log(LR) for each disease,
        then log-sum-exp normalize back to probabilities.

        Args:
            evidence:  Dict with 'rules_in' and 'rules_out' lists.
            test_id:   HP id (or mock test id) this evidence came from.
                       Stored in evidence_history so a disease introduced
                       later (via expand()) can replay this exact event.
            polarity:  "positive" | "negative" | None. Stored alongside
                       test_id for the same replay purpose. Optional and
                       backward-compatible — existing callers that don't
                       pass it simply produce a non-replayable history entry.

        Returns:
            Updated differential sorted by probability.
        """
        self.evidence_history.append({
            "test_id":  test_id,
            "polarity": polarity,
            "rules_in":  deepcopy(evidence.get("rules_in", [])),
            "rules_out": deepcopy(evidence.get("rules_out", [])),
        })
        self._validate_evidence(evidence)

        rules_in_keys  = {r["disease"].lower().strip() for r in evidence.get("rules_in",  [])}
        rules_out_keys = {r["disease"].lower().strip() for r in evidence.get("rules_out", [])}
        overlap = rules_in_keys & rules_out_keys
        if overlap:
            logger.warning(
                "DifferentialEngine: disease(s) %s appear in both rules_in and rules_out — "
                "both LRs applied; net log-LR is their sum (likely an authoring error)",
                overlap,
            )

        log_lr_map: Dict[str, float] = {}
        for rule in evidence.get("rules_in", []) + evidence.get("rules_out", []):
            key = rule["disease"].lower().strip()
            lr = max(rule["likelihood_ratio"], MIN_LR)
            log_lr_map[key] = log_lr_map.get(key, 0.0) + math.log(lr)

        for disease in self.differential:
            key = disease["name"].lower().strip()
            delta = log_lr_map.get(key, 0.0)
            disease["log_prob"] += delta
            disease["cum_log_lr"] = disease.get("cum_log_lr", 0.0) + delta

        max_log = max(d["log_prob"] for d in self.differential)
        log_total = max_log + math.log(
            sum(math.exp(d["log_prob"] - max_log) for d in self.differential)
        )

        for disease in self.differential:
            disease["log_prob"] -= log_total
            disease["probability"] = math.exp(disease["log_prob"])

        self.differential = sorted(self.differential, key=lambda x: x["probability"], reverse=True)
        return self.differential

    # ------------------------------------------------------------------ #
    #  Dynamic expansion (Phase 1 — late-symptom-discovery fix)           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_lr(edges: List[Dict], disease_name: str, polarity: str) -> Optional[float]:
        """
        Given the raw edges for one historical test (scoped to a single new
        disease), return the LR that disease would have received for that
        turn's result, using the same positive/negative inversion rule
        EvidenceEvaluatorAgent already applies:
          positive result → use edge lr as-is (RULES_IN boosts, RULES_OUT penalises)
          negative result → invert (1/lr)
        Returns None if no edge exists for this disease (neutral — no
        contribution, equivalent to LR=1).
        """
        key = disease_name.lower().strip()
        for e in edges:
            if e["disease"].lower().strip() == key:
                lr = e["lr"]
                if polarity == "positive":
                    return lr
                else:
                    return 1.0 / lr if lr else None
        return None

    def expand(
        self,
        new_diseases: List[Dict],
        replay_edges: Dict[str, List[Dict]],
        max_new: int = DEFAULT_MAX_NEW_CANDIDATES,
    ) -> List[Dict]:
        """
        Merge newly discovered disease candidates into the differential,
        giving each one a fair prior AND a fair replay of every evidence
        event already collected — never an arbitrary injected probability.

        Args:
            new_diseases:  Raw candidates from neo4j.get_initial_differential()
                           for a newly discovered HPO symptom. Each dict has
                           name, disease_id, specificity, prevalence.
            replay_edges:  {test_id: edges} — for every test_id present in
                           self.evidence_history, the caller must supply the
                           edges (from neo4j.get_test_edges(test_id, new_ids))
                           scoped to these new candidate diseases only.
            max_new:       Cap on how many new diseases one expansion event
                           may introduce (controls candidate-pool growth).

        Math:
          1. Drop candidates already present in self.differential.
          2. Cap to top `max_new` by (specificity × prevalence).
          3. For each surviving new disease, replay evidence_history to
             compute cum_log_lr — the same cumulative LR sum an old disease
             already carries.
          4. Recompute a FRESH prior over the union (old + new) diseases
             using the same specificity×prevalence + floor formula
             initialize() uses. This is the only valid way to compare priors
             drawn from two different symptom queries on one probability
             simplex. Old diseases' relative ratios to each other are
             provably unchanged by this (they all get the same treatment,
             a uniform shift that cancels in the final normalization) —
             only their absolute share shrinks slightly to make room for
             the new candidates, which is the mathematically correct
             consequence of the hypothesis space having grown.
          5. log_value(D) = fresh_log_prior(D) + cum_log_lr(D) for every
             disease, then a single log-sum-exp normalization over the union.

        Returns:
            The updated, merged, re-sorted differential. If no genuinely
            new candidates survive filtering, returns self.differential
            unchanged.
        """
        existing_keys = {_disease_key(d) for d in self.differential}

        seen_new = set()
        candidates = []
        for d in new_diseases:
            k = _disease_key(d)
            if k in existing_keys or k in seen_new:
                continue
            seen_new.add(k)
            candidates.append(deepcopy(d))

        if not candidates:
            return self.differential

        candidates.sort(key=lambda d: d["specificity"] * d["prevalence"], reverse=True)
        candidates = candidates[:max_new]

        # Replay evidence_history for each new candidate.
        for d in candidates:
            cum = 0.0
            for entry in self.evidence_history:
                test_id  = entry.get("test_id")
                polarity = entry.get("polarity")
                if not test_id or polarity not in ("positive", "negative"):
                    continue  # untagged historical entry — can't replay, treat as neutral
                edges = replay_edges.get(test_id, [])
                lr = self._resolve_lr(edges, d["name"], polarity)
                if lr is not None:
                    cum += math.log(max(lr, MIN_LR))
            d["cum_log_lr"] = cum

        union = self.differential + candidates
        self._floor_normalize(union)  # sets fresh 'probability' (prior only) + 'raw_score'

        for d in union:
            d["log_prob"] = math.log(d["probability"]) + d.get("cum_log_lr", 0.0)

        max_log = max(d["log_prob"] for d in union)
        log_total = max_log + math.log(
            sum(math.exp(d["log_prob"] - max_log) for d in union)
        )
        for d in union:
            d["log_prob"] -= log_total
            d["probability"] = math.exp(d["log_prob"])

        self.differential = sorted(union, key=lambda x: x["probability"], reverse=True)
        logger.info(
            "expand: introduced %d new candidate(s): %s",
            len(candidates), [d["name"] for d in candidates],
        )
        return self.differential

    def _validate_evidence(self, evidence: Dict) -> None:
        for rule in evidence.get("rules_in", []):
            lr = rule.get("likelihood_ratio", 1.0)
            if lr < 1.0:
                logger.warning(
                    "DifferentialEngine: rules_in entry for %r has LR=%.4f < 1 "
                    "(expected > 1 for boosting evidence — check knowledge base)",
                    rule.get("disease"), lr,
                )
        for rule in evidence.get("rules_out", []):
            lr = rule.get("likelihood_ratio", 1.0)
            if lr > 1.0:
                logger.warning(
                    "DifferentialEngine: rules_out entry for %r has LR=%.4f > 1 "
                    "(expected < 1 for penalising evidence — check knowledge base)",
                    rule.get("disease"), lr,
                )

    def get_top_n(self, n: int = 3) -> List[Dict]:
        return self.differential[:n]

    def get_evidence_count(self) -> int:
        return len(self.evidence_history)

    def get_summary(self) -> str:
        lines = ["Current Differential:"]
        for i, disease in enumerate(self.differential[:5], 1):
            lines.append(f"  {i}. {disease['name']}: {disease['probability']*100:.1f}%")
        return "\n".join(lines)