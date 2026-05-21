"""
Unit tests for Differential Engine
Run: pytest engines/tests/test_differential.py
"""

import logging
import math

from engines.differential import DifferentialEngine, RARE_DISEASE_FLOOR


def test_initialize():
    """Probabilities sum to 1, sorted descending"""
    engine = DifferentialEngine()

    diseases = [
        {"name": "COPD",          "specificity": 0.85, "prevalence": 0.065},
        {"name": "Asthma",        "specificity": 0.80, "prevalence": 0.08},
        {"name": "Heart Failure", "specificity": 0.90, "prevalence": 0.02},
    ]

    result = engine.initialize(diseases)

    total = sum(d['probability'] for d in result)
    assert abs(total - 1.0) < 1e-9

    assert result[0]['probability'] >= result[1]['probability']
    assert result[1]['probability'] >= result[2]['probability']

    print("PASS: Initial differential created")
    for d in result:
        print(f"  {d['name']}: {d['probability']*100:.1f}%")


def test_update():
    """Strong FEV1 evidence pushes COPD above 80%"""
    engine = DifferentialEngine()

    diseases = [
        {"name": "COPD",   "specificity": 0.85, "prevalence": 0.065},
        {"name": "Asthma", "specificity": 0.80, "prevalence": 0.08},
    ]

    engine.initialize(diseases)
    print("\nBefore evidence:")
    print(engine.get_summary())

    result = engine.update({
        'rules_in':  [{"disease": "COPD",   "likelihood_ratio": 8.5}],
        'rules_out': [{"disease": "Asthma", "likelihood_ratio": 0.3}],
    })
    print("\nAfter FEV1 < 0.7:")
    print(engine.get_summary())

    assert result[0]['name'] == 'COPD'
    assert result[0]['probability'] > 0.8
    print("\nPASS: Bayesian update works correctly")


def test_floor_triggers():
    """Disease with near-zero prevalence is floored to RARE_DISEASE_FLOOR"""
    engine = DifferentialEngine()

    diseases = [
        {"name": "Common", "specificity": 0.90, "prevalence": 0.50},
        {"name": "Rare",   "specificity": 0.80, "prevalence": 0.0001},
    ]

    result = engine.initialize(diseases)
    rare = next(d for d in result if d['name'] == 'Rare')

    # Without floor: 0.80×0.0001 / (0.90×0.50 + 0.80×0.0001) ≈ 0.018%
    # The floor is applied before re-normalization, so the final probability is
    # RARE_DISEASE_FLOOR / renorm_total, slightly below RARE_DISEASE_FLOOR.
    # With one dominant disease the renorm denominator ≈ 1 + RARE_DISEASE_FLOOR.
    no_floor_prob = (0.80 * 0.0001) / (0.90 * 0.50 + 0.80 * 0.0001)
    assert rare['probability'] > no_floor_prob * 50   # floor boosted it >50×
    assert rare['probability'] > RARE_DISEASE_FLOOR * 0.95  # close to 2% floor

    total = sum(d['probability'] for d in result)
    assert abs(total - 1.0) < 1e-9

    print(f"PASS: Rare disease floored: {no_floor_prob*100:.4f}% → {rare['probability']*100:.2f}%")


def test_sequential_updates():
    """Three consecutive positive results compound correctly"""
    engine = DifferentialEngine()

    diseases = [
        {"name": "COPD",   "specificity": 0.85, "prevalence": 0.065},
        {"name": "Asthma", "specificity": 0.80, "prevalence": 0.08},
    ]
    engine.initialize(diseases)

    for _ in range(3):
        engine.update({
            'rules_in':  [{"disease": "COPD",   "likelihood_ratio": 3.0}],
            'rules_out': [{"disease": "Asthma", "likelihood_ratio": 0.5}],
        })

    assert engine.differential[0]['name'] == 'COPD'
    assert engine.differential[0]['probability'] > 0.95
    assert engine.get_evidence_count() == 3

    print(f"PASS: COPD after 3 updates: {engine.differential[0]['probability']*100:.1f}%")


def test_evidence_for_unknown_disease_is_noop():
    """Evidence for a disease absent from the differential does not change any probability"""
    engine = DifferentialEngine()

    diseases = [
        {"name": "COPD",   "specificity": 0.85, "prevalence": 0.065},
        {"name": "Asthma", "specificity": 0.80, "prevalence": 0.08},
    ]
    engine.initialize(diseases)
    copd_before = next(d['probability'] for d in engine.differential if d['name'] == 'COPD')

    engine.update({
        'rules_in':  [{"disease": "Phantom Disease", "likelihood_ratio": 50.0}],
        'rules_out': [],
    })

    copd_after = next(d['probability'] for d in engine.differential if d['name'] == 'COPD')
    assert abs(copd_after - copd_before) < 1e-9

    print("PASS: Evidence for unknown disease is a no-op")


def test_single_disease():
    """Single-disease differential stays at 100% regardless of evidence"""
    engine = DifferentialEngine()

    diseases = [{"name": "COPD", "specificity": 0.85, "prevalence": 0.065}]
    result = engine.initialize(diseases)
    assert abs(result[0]['probability'] - 1.0) < 1e-9

    engine.update({'rules_in': [{"disease": "COPD", "likelihood_ratio": 8.5}], 'rules_out': []})
    assert abs(engine.differential[0]['probability'] - 1.0) < 1e-9

    print("PASS: Single-disease differential stays at 100%")


def test_extreme_lr_stability():
    """LR of 100 (the validator ceiling) does not produce NaN, inf, or non-unit sum"""
    engine = DifferentialEngine()

    diseases = [
        {"name": "COPD",          "specificity": 0.85, "prevalence": 0.065},
        {"name": "Heart Failure",  "specificity": 0.90, "prevalence": 0.020},
    ]
    engine.initialize(diseases)

    engine.update({
        'rules_in':  [{"disease": "Heart Failure", "likelihood_ratio": 100.0}],
        'rules_out': [],
    })

    for d in engine.differential:
        assert math.isfinite(d['probability']), f"non-finite probability for {d['name']}"
        assert math.isfinite(d['log_prob']),    f"non-finite log_prob for {d['name']}"

    total = sum(d['probability'] for d in engine.differential)
    assert abs(total - 1.0) < 1e-9

    print("PASS: Extreme LR=100 does not produce NaN/inf")


def test_direction_invariant_warning(caplog):
    """rules_in LR < 1 and rules_out LR > 1 each produce a logger warning"""
    engine = DifferentialEngine()

    diseases = [
        {"name": "COPD",   "specificity": 0.85, "prevalence": 0.065},
        {"name": "Asthma", "specificity": 0.80, "prevalence": 0.08},
    ]
    engine.initialize(diseases)

    with caplog.at_level(logging.WARNING, logger="engines.differential"):
        engine.update({
            'rules_in':  [{"disease": "COPD",   "likelihood_ratio": 0.2}],  # wrong: < 1
            'rules_out': [{"disease": "Asthma", "likelihood_ratio": 5.0}],  # wrong: > 1
        })

    assert "rules_in" in caplog.text and "0.2000" in caplog.text
    assert "rules_out" in caplog.text and "5.0000" in caplog.text

    print("PASS: Direction invariant violations produce warnings")


def test_overlap_warning(caplog):
    """Disease in both rules_in and rules_out triggers a warning"""
    engine = DifferentialEngine()

    diseases = [
        {"name": "COPD",   "specificity": 0.85, "prevalence": 0.065},
        {"name": "Asthma", "specificity": 0.80, "prevalence": 0.08},
    ]
    engine.initialize(diseases)

    with caplog.at_level(logging.WARNING, logger="engines.differential"):
        engine.update({
            'rules_in':  [{"disease": "COPD", "likelihood_ratio": 8.5}],
            'rules_out': [{"disease": "COPD", "likelihood_ratio": 0.15}],
        })

    assert "both rules_in and rules_out" in caplog.text

    print("PASS: Overlap between rules_in and rules_out produces warning")


if __name__ == "__main__":
    test_initialize()
    test_update()
    test_floor_triggers()
    test_sequential_updates()
    test_evidence_for_unknown_disease_is_noop()
    test_single_disease()
    test_extreme_lr_stability()
    # Note: test_direction_invariant_warning and test_overlap_warning require
    # the pytest caplog fixture — run them with: pytest engines/tests/test_differential.py
    print("\nPASS: All standalone Differential Engine tests passed")
