"""
Confidence Judge - Decides when to finalize diagnosis
No LLM required - simple threshold logic
"""

from typing import List, Dict, Tuple, Optional


class ConfidenceJudge:
    """
    Determines whether enough evidence has been collected to finalize diagnosis.
    
    Criteria:
    1. Minimum evidence count (≥4 questions answered)
    2. Top diagnosis confidence (≥75%)
    3. Runner-up sufficiently low (<40%)
    """
    
    def __init__(
        self,
        min_evidence: int = 4,
        min_top_confidence: float = 0.75,
        max_runner_up: float = 0.40
    ):
        self.min_evidence = min_evidence
        self.min_top_confidence = min_top_confidence
        self.max_runner_up = max_runner_up
    
    def should_finalize(
        self,
        differential: List[Dict],
        evidence_count: int,
        previous_top_prob: Optional[float] = None,
    ) -> Tuple[bool, Dict]:
        """
        Check if diagnosis should be finalized.

        Args:
            differential:      Current disease probabilities (sorted, post-last-answer)
            evidence_count:    Number of Q&A turns completed
            previous_top_prob: Top probability BEFORE the last answer was applied.
                               None on the first turn (no history yet).

        Returns:
            Tuple of (should_finalize: bool, details: Dict)
        """
        if len(differential) < 2:
            return False, {"reason": "Insufficient diseases in differential"}

        top       = differential[0]
        runner_up = differential[1]

        # Stability gate: if the last answer pushed the leader DOWN by more than
        # 10 percentage points the differential is actively shifting — keep asking.
        if previous_top_prob is not None:
            delta = top['probability'] - previous_top_prob
            if delta < -0.10:
                return False, {
                    "should_finalize":        False,
                    "evidence_count":         evidence_count,
                    "top_disease":            top['name'],
                    "top_probability":        top['probability'],
                    "runner_up_disease":      runner_up['name'],
                    "runner_up_probability":  runner_up['probability'],
                    "criteria_met":           {"stability": False},
                    "reason": (
                        f"Unstable — '{top['name']}' dropped "
                        f"{abs(delta) * 100:.1f}% on last answer; need more evidence"
                    ),
                }

        criteria = {
            "min_evidence":   evidence_count >= self.min_evidence,
            "top_confidence": top['probability'] >= self.min_top_confidence,
            "runner_up_low":  runner_up['probability'] < self.max_runner_up,
        }

        all_met = all(criteria.values())

        details = {
            "should_finalize":       all_met,
            "evidence_count":        evidence_count,
            "min_required":          self.min_evidence,
            "top_disease":           top['name'],
            "top_probability":       top['probability'],
            "runner_up_disease":     runner_up['name'],
            "runner_up_probability": runner_up['probability'],
            "criteria_met":          criteria,
            "reason":                self._get_reason(criteria, evidence_count),
        }

        return all_met, details
    
    def _get_reason(self, criteria: Dict, evidence_count: int) -> str:
        """Generate human-readable reason for decision"""
        if all(criteria.values()):
            return "All criteria met - ready to finalize"
        
        unmet = [k for k, v in criteria.items() if not v]
        
        if not criteria['min_evidence']:
            return f"Need more evidence ({evidence_count}/{self.min_evidence})"
        elif not criteria['runner_up_low']:
            return "Runner-up diagnosis still too likely - need discriminating test"
        elif not criteria['top_confidence']:
            return "Top diagnosis confidence too low"
        
        return f"Unmet criteria: {', '.join(unmet)}"