"""
Test Question Selector Agent
Requires: Ollama running with llama3.1:8b
"""

from agents.question_selector import QuestionSelectorAgent


def test_question_selector():
    """Test LLM-based question selection"""
    agent = QuestionSelectorAgent()
    
    differential = [
        {"name": "COPD", "probability": 0.40},
        {"name": "Asthma", "probability": 0.35},
        {"name": "Heart Failure", "probability": 0.25},
    ]
    
    available_tests = [
        {"id": "test_fev1", "name": "FEV1/FVC ratio", "diseases": ["COPD", "Asthma"]},
        {"id": "test_bnp", "name": "BNP blood test", "diseases": ["Heart Failure"]},
    ]
    
    print("\nCalling Question Selector Agent...")
    result = agent.select_question(differential, available_tests)
    
    print(f"\nPASS: Selected question: {result['question']}")
    print(f"  Test ID: {result['test_id']}")
    print(f"  Reasoning: {result['reasoning']}")
    
    assert 'question' in result
    assert 'test_id' in result


if __name__ == "__main__":
    test_question_selector()