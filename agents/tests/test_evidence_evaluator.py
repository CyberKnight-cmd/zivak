"""
Test Evidence Evaluator Agent
Requires: Ollama running with llama3.1:8b
"""

from agents.evidence_evaluator import EvidenceEvaluatorAgent


def test_evidence_evaluator():
    """Test LLM-based evidence evaluation"""
    agent = EvidenceEvaluatorAgent()
    
    question = "What is the patient's FEV1/FVC ratio?"
    answer = "0.62 (below 0.7, indicating obstruction)"
    
    edges = [
        {"disease": "COPD", "relationship": "RULES_IN", "lr": 8.5},
        {"disease": "Asthma", "relationship": "RULES_OUT", "lr": 0.3},
        {"disease": "Heart Failure", "relationship": "NEUTRAL", "lr": 1.0},
    ]
    
    print("\nCalling Evidence Evaluator Agent...")
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    
    result = agent.evaluate(question, answer, edges)
    
    print(f"\nPASS: Evaluation complete:")
    print(f"  Rules in: {result['rules_in']}")
    print(f"  Rules out: {result['rules_out']}")
    
    assert 'rules_in' in result
    assert 'rules_out' in result
    assert len(result['rules_in']) > 0  # Should have at least one


if __name__ == "__main__":
    test_evidence_evaluator()