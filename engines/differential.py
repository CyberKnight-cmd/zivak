"""
Differential Engine - Bayesian probability updates
No LLM required - pure mathematics
"""

import logging
import math
from typing import List, Dict
from copy import deepcopy

logger = logging.getLogger(__name__)

# Probability floor for rare disease seeder (blueprint spec: 2% minimum prior)
RARE_DISEASE_FLOOR = 0.02
MIN_LR = 1e-10  # Guard against log(0) if a badly-formed LR of 0 arrives


class DifferentialEngine:
    """
    Maintains and updates disease probability distribution using Bayes' theorem.

    Works in log-probability space to prevent floating-point underflow when
    many sequential LR updates push rare-disease probabilities toward zero.

    Core operation: log P_new = log P_old + log(LR), then log-sum-exp normalize.
    """

    def __init__(self):
        self.differential: List[Dict] = []
        self.evidence_history: List[Dict] = []

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

        for disease in diseases:
            disease['raw_score'] = disease['specificity'] * disease['prevalence']

        total = sum(d['raw_score'] for d in diseases)
        if total == 0:
            return []

        # Normalize then apply rare-disease floor, then re-normalize
        for disease in diseases:
            disease['probability'] = max(disease['raw_score'] / total, RARE_DISEASE_FLOOR)

        total = sum(d['probability'] for d in diseases)
        for disease in diseases:
            disease['probability'] /= total
            disease['log_prob'] = math.log(disease['probability'])

        self.differential = sorted(diseases, key=lambda x: x['probability'], reverse=True)
        return self.differential

    def update(self, evidence: Dict) -> List[Dict]:
        """
        Update probabilities using Bayes' theorem in log space.

        log P_new = log P_old + log(LR) for each disease,
        then log-sum-exp normalize back to probabilities.

        Multiple LRs for the same disease (independent evidence) are
        summed in log space (equivalent to multiplying in probability space).
        Name matching is case-insensitive and whitespace-stripped.

        Args:
            evidence: Dict with 'rules_in' and 'rules_out' lists.
                      Each entry has 'disease' and 'likelihood_ratio'.
                      rules_in  LRs should be > 1 (evidence boosting the disease).
                      rules_out LRs should be < 1 (evidence penalising the disease).

        Returns:
            Updated differential sorted by probability.
        """
        self.evidence_history.append(deepcopy(evidence))
        self._validate_evidence(evidence)

        # Warn when the same disease appears in both lists — both LRs still applied
        # (net = log(lr_in) + log(lr_out)), but this is almost always an authoring error.
        rules_in_keys  = {r['disease'].lower().strip() for r in evidence.get('rules_in',  [])}
        rules_out_keys = {r['disease'].lower().strip() for r in evidence.get('rules_out', [])}
        overlap = rules_in_keys & rules_out_keys
        if overlap:
            logger.warning(
                "DifferentialEngine: disease(s) %s appear in both rules_in and rules_out — "
                "both LRs applied; net log-LR is their sum (likely an authoring error)",
                overlap,
            )

        # Build log-LR map: sum log(LR) per disease (= multiply LRs in prob space)
        log_lr_map: Dict[str, float] = {}
        for rule in evidence.get('rules_in', []) + evidence.get('rules_out', []):
            key = rule['disease'].lower().strip()
            lr = max(rule['likelihood_ratio'], MIN_LR)
            log_lr_map[key] = log_lr_map.get(key, 0.0) + math.log(lr)

        # Update in log space
        for disease in self.differential:
            key = disease['name'].lower().strip()
            disease['log_prob'] += log_lr_map.get(key, 0.0)  # 0.0 = log(1.0) = neutral

        # Log-sum-exp normalization (numerically stable)
        max_log = max(d['log_prob'] for d in self.differential)
        log_total = max_log + math.log(
            sum(math.exp(d['log_prob'] - max_log) for d in self.differential)
        )

        for disease in self.differential:
            disease['log_prob'] -= log_total
            disease['probability'] = math.exp(disease['log_prob'])

        self.differential = sorted(self.differential, key=lambda x: x['probability'], reverse=True)
        return self.differential

    def _validate_evidence(self, evidence: Dict) -> None:
        """
        Warn when LR values violate the expected direction invariant.

        rules_in  entries should have LR > 1 (positive evidence boosting a disease).
        rules_out entries should have LR < 1 (positive evidence penalising a disease).

        Violations don't crash the engine — the math still runs — but they almost
        certainly indicate a knowledge-base authoring error (e.g. a penalising LR
        accidentally placed in rules_in), so each violation gets a warning.
        """
        for rule in evidence.get('rules_in', []):
            lr = rule.get('likelihood_ratio', 1.0)
            if lr < 1.0:
                logger.warning(
                    "DifferentialEngine: rules_in entry for %r has LR=%.4f < 1 "
                    "(expected > 1 for boosting evidence — check knowledge base)",
                    rule.get('disease'), lr,
                )
        for rule in evidence.get('rules_out', []):
            lr = rule.get('likelihood_ratio', 1.0)
            if lr > 1.0:
                logger.warning(
                    "DifferentialEngine: rules_out entry for %r has LR=%.4f > 1 "
                    "(expected < 1 for penalising evidence — check knowledge base)",
                    rule.get('disease'), lr,
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
