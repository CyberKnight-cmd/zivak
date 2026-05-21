"""
Unit tests for Confidence Judge
Run: pytest engines/tests/test_confidence_judge.py
"""

from engines.confidence_judge import ConfidenceJudge


def test_early_stage():
    """Should not finalize with only 1 evidence point"""
    judge = ConfidenceJudge()
    
    differential = [
        {"name": "COPD", "probability": 0.86},
        {"name": "Asthma", "probability": 0.12},
    ]
    
    should_finalize, details = judge.should_finalize(differential, evidence_count=1)
    
    assert not should_finalize
    assert details['reason'] == "Need more evidence (1/4)"
    print("PASS: Correctly requires minimum evidence")


def test_high_confidence():
    """Should finalize when all criteria met"""
    judge = ConfidenceJudge()
    
    differential = [
        {"name": "COPD", "probability": 0.92},  # >75%
        {"name": "Heart Failure", "probability": 0.06},  # <40%
    ]
    
    should_finalize, details = judge.should_finalize(differential, evidence_count=4)
    
    assert should_finalize
    assert details['reason'] == "All criteria met - ready to finalize"
    print("PASS: Correctly finalizes with strong evidence")


def test_ambiguous_differential():
    """Should not finalize if runner-up is still high"""
    judge = ConfidenceJudge()
    
    differential = [
        {"name": "COPD", "probability": 0.55},
        {"name": "Asthma", "probability": 0.45},  # Too high!
    ]
    
    should_finalize, details = judge.should_finalize(differential, evidence_count=4)
    
    assert not should_finalize
    assert "runner-up" in details['reason'].lower()
    print("PASS: Correctly continues when differential is ambiguous")


if __name__ == "__main__":
    test_early_stage()
    test_high_confidence()
    test_ambiguous_differential()
    print("\nPASS: All Confidence Judge tests passed")