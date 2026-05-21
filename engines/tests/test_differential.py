"""
Unit tests for Differential Engine
Run: pytest engines/tests/test_differential.py
"""

from engines.differential import DifferentialEngine


def test_initialize():
    """Test initial probability calculation"""
    engine = DifferentialEngine()
    
    diseases = [
        {"name": "COPD", "specificity": 0.85, "prevalence": 0.065},
        {"name": "Asthma", "specificity": 0.80, "prevalence": 0.08},
        {"name": "Heart Failure", "specificity": 0.90, "prevalence": 0.02},
    ]
    
    result = engine.initialize(diseases)
    
    # Check probabilities sum to 1
    total = sum(d['probability'] for d in result)
    assert abs(total - 1.0) < 0.001
    
    # Check sorted by probability
    assert result[0]['probability'] >= result[1]['probability']
    
    print("PASS: Initial differential created")
    for d in result:
        print(f"  {d['name']}: {d['probability']*100:.1f}%")


def test_update():
    """Test Bayesian update with evidence"""
    engine = DifferentialEngine()
    
    diseases = [
        {"name": "COPD", "specificity": 0.85, "prevalence": 0.065},
        {"name": "Asthma", "specificity": 0.80, "prevalence": 0.08},
    ]
    
    engine.initialize(diseases)
    print("\nBefore evidence:")
    print(engine.get_summary())
    
    # Apply strong evidence for COPD
    evidence = {
        'rules_in': [{"disease": "COPD", "likelihood_ratio": 8.5}],
        'rules_out': [{"disease": "Asthma", "likelihood_ratio": 0.3}]
    }
    
    result = engine.update(evidence)
    print("\nAfter FEV1 < 0.7:")
    print(engine.get_summary())
    
    # COPD should now be much higher
    assert result[0]['name'] == 'COPD'
    assert result[0]['probability'] > 0.8  # Should be >80%
    
    print("\nPASS: Bayesian update works correctly")


if __name__ == "__main__":
    test_initialize()
    test_update()