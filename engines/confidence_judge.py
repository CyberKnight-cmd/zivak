"""
Confidence Judge - Decides when to finalize diagnosis
No LLM required - simple threshold logic
"""

from typing import List, Dict, Tuple, Optional


class ConfidenceJudge:
    """
    Determines whether enough evidence has been collected to finalize diagnosis.

    All three criteria must hold before finalizing:
    1. Minimum evidence count (>= min_evidence questions answered)
    2. Top diagnosis confidence (>= min_top_confidence)
    3. Runner-up sufficiently separated (< max_runner_up)
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
            differential:      Current disease probabilities (sorted, post-last-answer).
            evidence_count:    Number of Q&A turns completed.
            previous_top_prob: Top probability BEFORE the last answer was applied.
                               Pass None on the first turn (no prior snapshot exists).

        Returns:
            Tuple of (should_finalize: bool, details: Dict).

            details always contains:
              should_finalize, evidence_count, min_required,
              top_disease, top_probability,
              runner_up_disease, runner_up_probability,
              criteria_met, reason, confidence_warning.

            confidence_warning is True when top_probability < min_top_confidence.
            Callers that terminate the session via "no_tests" or "max_questions"
            should propagate this flag to the API response so consumers can
            distinguish a high-confidence diagnosis from a forced termination.
        """
        if len(differential) < 2:
            top_prob = differential[0]["probability"] if differential else 0.0
            return False, {
                "should_finalize":        False,
                "evidence_count":         evidence_count,
                "min_required":           self.min_evidence,
                "top_disease":            differential[0]["name"] if differential else None,
                "top_probability":        top_prob,
                "runner_up_disease":      None,
                "runner_up_probability":  None,
                "criteria_met":           {},
                "reason":                 "Insufficient diseases in differential",
                "confidence_warning":     top_prob < self.min_top_confidence,
            }

        top       = differential[0]
        runner_up = differential[1]
        confidence_warning = top["probability"] < self.min_top_confidence

        # Stability gate: if the last answer pushed the leader DOWN by more than
        # 10 percentage points the differential is actively shifting — keep asking.
        if previous_top_prob is not None:
            delta = top["probability"] - previous_top_prob
            if delta < -0.10:
                return False, {
                    "should_finalize":        False,
                    "evidence_count":         evidence_count,
                    "min_required":           self.min_evidence,
                    "top_disease":            top["name"],
                    "top_probability":        top["probability"],
                    "runner_up_disease":      runner_up["name"],
                    "runner_up_probability":  runner_up["probability"],
                    "criteria_met":           {"stability": False},
                    "reason": (
                        f"Unstable — '{top['name']}' dropped "
                        f"{abs(delta) * 100:.1f}% on last answer; need more evidence"
                    ),
                    "confidence_warning":     confidence_warning,
                }

        criteria = {
            "min_evidence":   evidence_count >= self.min_evidence,
            "top_confidence": top["probability"] >= self.min_top_confidence,
            "runner_up_low":  runner_up["probability"] < self.max_runner_up,
        }

        all_met = all(criteria.values())

        return all_met, {
            "should_finalize":       all_met,
            "evidence_count":        evidence_count,
            "min_required":          self.min_evidence,
            "top_disease":           top["name"],
            "top_probability":       top["probability"],
            "runner_up_disease":     runner_up["name"],
            "runner_up_probability": runner_up["probability"],
            "criteria_met":          criteria,
            "reason":                self._get_reason(criteria, evidence_count),
            "confidence_warning":    confidence_warning,
        }

    def _get_reason(self, criteria: Dict, evidence_count: int) -> str:
        if all(criteria.values()):
            return "All criteria met - ready to finalize"
        if not criteria["min_evidence"]:
            return f"Need more evidence ({evidence_count}/{self.min_evidence})"
        if not criteria["runner_up_low"]:
            return "Runner-up diagnosis still too likely - need discriminating test"
        return "Top diagnosis confidence too low"
