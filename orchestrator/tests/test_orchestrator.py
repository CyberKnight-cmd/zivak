"""
Test Orchestrator - LangGraph + mock clients
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from orchestrator.orchestrator import DiagnosticOrchestrator
from orchestrator.mock_clients import MockQdrantClient, MockNeo4jClient


def test_full_workflow():
    """Test complete diagnostic workflow with mock clients and LangGraph graph."""

    qdrant = MockQdrantClient()
    neo4j  = MockNeo4jClient()
    orch   = DiagnosticOrchestrator(qdrant, neo4j)

    print("\n" + "=" * 60)
    print("ZIVAK - LangGraph Orchestrator Test (Mock Data)")
    print("=" * 60)

    # Step 1: Start session - runs seed node + first qa interrupt
    print("\n[1] Starting session...")
    user_input = "My chest feels heavy when I walk upstairs"
    session_id, result = orch.start_session(user_input)

    print(f"\n  Session ID : {session_id}")
    print(f"  Symptom    : {result['symptom_match']['clinical_term']}")
    print(f"\nInitial Differential:")
    for d in result["initial_differential"][:3]:
        print(f"  {d['name']}: {d['probability'] * 100:.1f}%")

    question = result.get("next_question")
    assert question is not None, "Expected a first question after session start"
    assert "question" in question
    assert "test_id"  in question

    print(f"\n  First question : {question['question']}")
    print(f"  Test ID        : {question['test_id']}")
    print(f"  Reasoning      : {question.get('reasoning', 'N/A')}")

    # Step 2: Submit answer - resumes graph, processes evidence, selects next question
    print("\n[2] Submitting answer...")
    answer = "0.62 (below 0.7 - obstruction pattern)"
    result = orch.submit_answer(session_id, answer)

    print(f"\n  Updated Differential:")
    for d in result["updated_differential"][:3]:
        print(f"  {d['name']}: {d['probability'] * 100:.1f}%")

    print(f"\n  Should continue : {result['should_continue']}")
    if result.get("next_question"):
        print(f"  Next question   : {result['next_question']['question']}")
    if result.get("judge_details"):
        print(f"  Judge reason    : {result['judge_details'].get('reason', 'N/A')}")

    assert "updated_differential" in result
    assert "should_continue"      in result

    print("\n" + "=" * 60)
    print("Orchestrator test complete")
    print("=" * 60)


if __name__ == "__main__":
    test_full_workflow()
