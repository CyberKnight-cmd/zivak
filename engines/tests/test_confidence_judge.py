"""
Unit tests for Confidence Judge
Run: pytest engines/tests/test_confidence_judge.py
"""

from engines.confidence_judge import ConfidenceJudge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _diff(top_p, runner_up_p, top_name="COPD", runner_name="Asthma"):
    return [
        {"name": top_name,   "probability": top_p},
        {"name": runner_name, "probability": runner_up_p},
    ]


# ---------------------------------------------------------------------------
# Original tests (kept intact)
# ---------------------------------------------------------------------------

def test_early_stage():
    """Should not finalize with only 1 evidence point"""
    judge = ConfidenceJudge()

    differential = [
        {"name": "COPD",   "probability": 0.86},
        {"name": "Asthma", "probability": 0.12},
    ]

    should_finalize, details = judge.should_finalize(differential, evidence_count=1)

    assert not should_finalize
    assert details["reason"] == "Need more evidence (1/4)"
    print("PASS: Correctly requires minimum evidence")


def test_high_confidence():
    """Should finalize when all criteria met"""
    judge = ConfidenceJudge()

    differential = [
        {"name": "COPD",          "probability": 0.92},
        {"name": "Heart Failure", "probability": 0.06},
    ]

    should_finalize, details = judge.should_finalize(differential, evidence_count=4)

    assert should_finalize
    assert details["reason"] == "All criteria met - ready to finalize"
    print("PASS: Correctly finalizes with strong evidence")


def test_ambiguous_differential():
    """Should not finalize if runner-up is still high"""
    judge = ConfidenceJudge()

    differential = [
        {"name": "COPD",   "probability": 0.55},
        {"name": "Asthma", "probability": 0.45},
    ]

    should_finalize, details = judge.should_finalize(differential, evidence_count=4)

    assert not should_finalize
    assert "runner-up" in details["reason"].lower()
    print("PASS: Correctly continues when differential is ambiguous")


# ---------------------------------------------------------------------------
# New tests — missing coverage from report
# ---------------------------------------------------------------------------

def test_stability_gate_fires():
    """Leader dropped >10pp on last answer — keep asking despite high confidence"""
    judge = ConfidenceJudge()

    # Post-update: COPD has fallen from 80% to 55% (delta = -25%)
    should_finalize, details = judge.should_finalize(
        differential=_diff(0.55, 0.40),
        evidence_count=5,
        previous_top_prob=0.80,
    )

    assert not should_finalize
    assert "Unstable" in details["reason"]
    assert "dropped" in details["reason"]
    assert "25.0%" in details["reason"]
    print("PASS: Stability gate fires on >10pp leader drop")


def test_stability_gate_does_not_fire_on_small_drop():
    """Leader dropped 5pp — within tolerance, normal criteria apply"""
    judge = ConfidenceJudge()

    # Post-update: COPD at 80%, was 85% (delta = -5%) — gate should NOT fire
    should_finalize, details = judge.should_finalize(
        differential=_diff(0.80, 0.15),
        evidence_count=4,
        previous_top_prob=0.85,
    )

    assert should_finalize
    assert details["reason"] == "All criteria met - ready to finalize"
    print("PASS: Stability gate ignores small drops (≤10pp)")


def test_exact_thresholds_finalize():
    """Exactly at every boundary (evidence=4, top=0.75, runner_up=0.399) should finalize"""
    judge = ConfidenceJudge()

    should_finalize, details = judge.should_finalize(
        differential=_diff(0.75, 0.399),
        evidence_count=4,
    )

    assert should_finalize
    assert details["criteria_met"]["min_evidence"] is True
    assert details["criteria_met"]["top_confidence"] is True
    assert details["criteria_met"]["runner_up_low"] is True
    print("PASS: Exactly at thresholds → finalize")


def test_runner_up_at_boundary_blocks():
    """runner_up exactly at 0.40 should NOT finalize (strict less-than)"""
    judge = ConfidenceJudge()

    should_finalize, details = judge.should_finalize(
        differential=_diff(0.76, 0.40),
        evidence_count=4,
    )

    assert not should_finalize
    assert not details["criteria_met"]["runner_up_low"]
    print("PASS: runner_up = 0.40 (not strictly < 0.40) blocks finalization")


def test_single_disease_guard():
    """Differential with only one disease returns False with informative reason"""
    judge = ConfidenceJudge()

    should_finalize, details = judge.should_finalize(
        differential=[{"name": "COPD", "probability": 1.0}],
        evidence_count=5,
    )

    assert not should_finalize
    assert "Insufficient" in details["reason"]
    assert details["runner_up_disease"] is None
    assert details["runner_up_probability"] is None
    assert "min_required" in details
    print("PASS: Single-disease differential returns guard response")


def test_empty_differential_guard():
    """Empty differential returns False without KeyError"""
    judge = ConfidenceJudge()

    should_finalize, details = judge.should_finalize(
        differential=[],
        evidence_count=0,
    )

    assert not should_finalize
    assert "Insufficient" in details["reason"]
    assert details["top_disease"] is None
    print("PASS: Empty differential handled without error")


def test_previous_top_prob_none_skips_stability():
    """First turn (previous_top_prob=None) skips stability gate, normal criteria apply"""
    judge = ConfidenceJudge()

    # High confidence, but first turn — stability gate must not fire
    should_finalize, details = judge.should_finalize(
        differential=_diff(0.85, 0.10),
        evidence_count=4,
        previous_top_prob=None,
    )

    assert should_finalize
    assert details["reason"] == "All criteria met - ready to finalize"
    print("PASS: previous_top_prob=None skips stability gate")


def test_confidence_warning_true_when_low():
    """confidence_warning is True when top_probability < min_top_confidence"""
    judge = ConfidenceJudge()

    _, details = judge.should_finalize(
        differential=_diff(0.60, 0.30),
        evidence_count=2,
    )

    assert details["confidence_warning"] is True
    print("PASS: confidence_warning=True when top_probability < 0.75")


def test_confidence_warning_false_when_met():
    """confidence_warning is False when all criteria met"""
    judge = ConfidenceJudge()

    _, details = judge.should_finalize(
        differential=_diff(0.90, 0.08),
        evidence_count=4,
    )

    assert details["confidence_warning"] is False
    print("PASS: confidence_warning=False when top_probability >= 0.75")


def test_details_keys_consistent_across_all_paths():
    """Every return path emits the same set of top-level keys"""
    judge = ConfidenceJudge()
    expected_keys = {
        "should_finalize", "evidence_count", "min_required",
        "top_disease", "top_probability",
        "runner_up_disease", "runner_up_probability",
        "criteria_met", "reason", "confidence_warning",
    }

    cases = [
        # Normal False
        judge.should_finalize(_diff(0.50, 0.40), evidence_count=2)[1],
        # Normal True
        judge.should_finalize(_diff(0.90, 0.05), evidence_count=4)[1],
        # Stability gate
        judge.should_finalize(_diff(0.55, 0.35), evidence_count=5, previous_top_prob=0.80)[1],
        # Single disease guard
        judge.should_finalize([{"name": "COPD", "probability": 1.0}], evidence_count=3)[1],
    ]

    for i, details in enumerate(cases):
        missing = expected_keys - details.keys()
        assert not missing, f"Case {i} missing keys: {missing}"

    print("PASS: All return paths emit consistent detail keys")


if __name__ == "__main__":
    test_early_stage()
    test_high_confidence()
    test_ambiguous_differential()
    test_stability_gate_fires()
    test_stability_gate_does_not_fire_on_small_drop()
    test_exact_thresholds_finalize()
    test_runner_up_at_boundary_blocks()
    test_single_disease_guard()
    test_empty_differential_guard()
    test_previous_top_prob_none_skips_stability()
    test_confidence_warning_true_when_low()
    test_confidence_warning_false_when_met()
    test_details_keys_consistent_across_all_paths()
    print("\nPASS: All ConfidenceJudge tests passed")
