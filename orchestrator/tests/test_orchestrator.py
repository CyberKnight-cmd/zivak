"""
Tests for DiagnosticOrchestrator
Run: pytest orchestrator/tests/test_orchestrator.py

Integration test (test_full_workflow) requires Ollama.
All other tests are pure-Python unit tests — no LLM needed.
"""

import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from orchestrator.orchestrator import (
    DiagnosticOrchestrator,
    DiagnosticState,
    SessionNotFoundError,
    _sanitize,
    finalize_node,
    question_node,
    MAX_QUESTIONS,
)
from orchestrator.mock_clients import MockQdrantClient, MockNeo4jClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _base_state(**overrides) -> DiagnosticState:
    """Minimal valid DiagnosticState for unit testing individual nodes."""
    state: DiagnosticState = {
        "user_input":          "chest feels heavy",
        "symptom_match":       {"clinical_term": "Exertional Dyspnoea", "symptom_id": "HP:0002875", "score": 0.95},
        "differential": [
            {"name": "COPD",          "probability": 0.40, "log_prob": -0.92},
            {"name": "Asthma",        "probability": 0.35, "log_prob": -1.05},
            {"name": "Heart Failure", "probability": 0.25, "log_prob": -1.39},
        ],
        "evidence_history":    [],
        "questions_asked":     [],
        "should_finalize":     False,
        "judge_details":       {},
        "final_diagnosis":     None,
        "previous_top_prob":   None,
        "pending_question":    None,
        "finalization_reason": None,
        "confidence_warning":  False,
    }
    state.update(overrides)
    return state


def _mock_config(neo4j=None, selector=None, evaluator=None):
    """Build a minimal config dict for node calls."""
    neo4j_client = neo4j or MockNeo4jClient()
    if selector is None:
        selector = MagicMock()
        selector.select_question.return_value = {
            "test_id": "test_fev1", "question": "What is FEV1?", "reasoning": "discriminates COPD/Asthma",
        }
    return {
        "configurable": {
            "neo4j":     neo4j_client,
            "qdrant":    MockQdrantClient(),
            "selector":  selector,
            "evaluator": evaluator or MagicMock(),
        }
    }


# ---------------------------------------------------------------------------
# _sanitize — unit tests
# ---------------------------------------------------------------------------

def test_sanitize_strips_control_chars():
    assert _sanitize("hello\x00world") == "helloworld"
    assert _sanitize("hello\x01\x02\x1f") == "hello"
    print("PASS: control characters stripped")


def test_sanitize_preserves_newlines_and_tabs():
    text = "line one\nline two\ttabbed"
    assert _sanitize(text) == text
    print("PASS: newlines and tabs preserved")


def test_sanitize_caps_at_max_input_len():
    from orchestrator.orchestrator import MAX_INPUT_LEN
    long_text = "a" * (MAX_INPUT_LEN + 500)
    result = _sanitize(long_text)
    assert len(result) == MAX_INPUT_LEN
    print(f"PASS: output capped at {MAX_INPUT_LEN} chars")


def test_sanitize_strips_whitespace():
    assert _sanitize("  hello  ") == "hello"
    print("PASS: leading/trailing whitespace stripped")


def test_sanitize_empty_string():
    assert _sanitize("") == ""
    print("PASS: empty string returns empty string")


def test_sanitize_preserves_unicode():
    text = "température élevée"
    assert _sanitize(text) == text
    print("PASS: unicode characters preserved")


# ---------------------------------------------------------------------------
# question_node — termination path unit tests
# ---------------------------------------------------------------------------

def test_question_node_confidence_gate():
    """High-confidence state → question_node returns should_finalize=True."""
    # evidence_count >= 4, top >= 0.75, runner_up < 0.40
    state = _base_state(
        differential=[
            {"name": "COPD",   "probability": 0.92, "log_prob": -0.08},
            {"name": "Asthma", "probability": 0.08, "log_prob": -2.53},
        ],
        evidence_history=[{}, {}, {}, {}],   # 4 turns
        questions_asked=[
            {"test_id": f"test_{i}", "question": f"Q{i}", "reasoning": ""}
            for i in range(4)
        ],
    )
    result = question_node(state, _mock_config())

    assert result["should_finalize"] is True
    assert result["finalization_reason"] == "confidence_gate"
    assert "confidence_warning" in result
    print("PASS: confidence gate sets should_finalize + finalization_reason")


def test_question_node_max_questions():
    """MAX_QUESTIONS reached → finalization_reason='max_questions'."""
    state = _base_state(
        questions_asked=[
            {"test_id": f"test_{i}", "question": f"Q{i}", "reasoning": ""}
            for i in range(MAX_QUESTIONS)
        ],
    )
    result = question_node(state, _mock_config())

    assert result["should_finalize"] is True
    assert result["finalization_reason"] == "max_questions"
    assert "confidence_warning" in result
    print("PASS: MAX_QUESTIONS cap sets finalization_reason='max_questions'")


def test_question_node_no_tests_available():
    """All tests already asked → finalization_reason='no_tests'.

    Uses the vertigo cluster which has 7 unique tests — fewer than MAX_QUESTIONS(10)
    so the MAX_QUESTIONS cap does not fire before the no_tests check.
    """
    neo4j = MockNeo4jClient()
    vertigo_diseases = ["BPPV", "Vestibular Neuritis", "Meniere's Disease", "Central Vertigo"]
    all_test_ids = {
        t["id"]
        for disease in vertigo_diseases
        for t in neo4j.get_available_tests([disease])
    }
    assert len(all_test_ids) < MAX_QUESTIONS, (
        f"Vertigo test count ({len(all_test_ids)}) must be < MAX_QUESTIONS ({MAX_QUESTIONS})"
    )
    state = _base_state(
        symptom_match={"clinical_term": "Vertigo", "symptom_id": "HP:0002321", "score": 0.95},
        differential=[
            {"name": "BPPV",                "probability": 0.50, "log_prob": -0.69},
            {"name": "Vestibular Neuritis",  "probability": 0.25, "log_prob": -1.39},
            {"name": "Meniere's Disease",    "probability": 0.15, "log_prob": -1.90},
            {"name": "Central Vertigo",      "probability": 0.10, "log_prob": -2.30},
        ],
        questions_asked=[
            {"test_id": tid, "question": "Q?", "reasoning": ""}
            for tid in all_test_ids
        ],
    )
    result = question_node(state, _mock_config(neo4j=neo4j))

    assert result["should_finalize"] is True
    assert result["finalization_reason"] == "no_tests"
    print("PASS: no available tests sets finalization_reason='no_tests'")


def test_question_node_selects_question_and_saves_to_state():
    """Normal path → should_finalize=False, pending_question set."""
    state  = _base_state()
    result = question_node(state, _mock_config())

    assert result["should_finalize"] is False
    assert result["pending_question"] is not None
    assert "test_id"  in result["pending_question"]
    assert "question" in result["pending_question"]
    print("PASS: normal path sets pending_question in state")


def test_question_node_all_diseases_in_pool():
    """Test pool uses all differential diseases, not just top-5."""
    # ILD is rank 7 in dyspnoea — its test (test_hrct) should be available
    neo4j = MockNeo4jClient()
    state = _base_state(
        differential=[
            {"name": "COPD",                    "probability": 0.30, "log_prob": -1.20},
            {"name": "Asthma",                  "probability": 0.25, "log_prob": -1.39},
            {"name": "Heart Failure",            "probability": 0.15, "log_prob": -1.90},
            {"name": "Pulmonary Embolism",       "probability": 0.10, "log_prob": -2.30},
            {"name": "Pneumonia",                "probability": 0.08, "log_prob": -2.53},
            {"name": "Anemia",                   "probability": 0.07, "log_prob": -2.66},
            {"name": "Interstitial Lung Disease","probability": 0.05, "log_prob": -3.00},
        ],
    )

    # Capture what available_tests the selector receives
    captured = {}
    mock_selector = MagicMock()
    def capture_select(diff, tests, **kwargs):
        captured["tests"] = tests
        return {"test_id": tests[0]["id"], "question": "Q?", "reasoning": ""}
    mock_selector.select_question.side_effect = capture_select

    question_node(state, _mock_config(neo4j=neo4j, selector=mock_selector))

    test_ids = {t["id"] for t in captured.get("tests", [])}
    # test_hrct is ILD's key test — must appear even though ILD is rank 7
    assert "test_hrct" in test_ids, f"test_hrct missing from pool: {test_ids}"
    print("PASS: test_hrct (ILD, rank 7) included in test pool")


def test_question_node_passes_lr_map_to_selector():
    """question_node passes test_lr_map to selector (N1 wiring)."""
    captured = {}
    mock_selector = MagicMock()
    def capture_select(diff, tests, **kwargs):
        captured.update(kwargs)
        return {"test_id": tests[0]["id"], "question": "Q?", "reasoning": ""}
    mock_selector.select_question.side_effect = capture_select

    question_node(_base_state(), _mock_config(selector=mock_selector))

    assert "test_lr_map" in captured
    assert isinstance(captured["test_lr_map"], dict)
    assert len(captured["test_lr_map"]) > 0
    print("PASS: test_lr_map passed to selector")


def test_question_node_passes_symptom_and_history():
    """question_node passes symptom and qa_history to selector (N4 wiring)."""
    captured = {}
    mock_selector = MagicMock()
    def capture_select(diff, tests, **kwargs):
        captured.update(kwargs)
        return {"test_id": tests[0]["id"], "question": "Q?", "reasoning": ""}
    mock_selector.select_question.side_effect = capture_select

    state = _base_state(
        questions_asked=[{"test_id": "test_bnp", "question": "What is BNP?", "reasoning": ""}],
    )
    question_node(state, _mock_config(selector=mock_selector))

    assert captured.get("symptom") == "Exertional Dyspnoea"
    assert captured.get("qa_history") is not None
    assert any("BNP" in qa["question"] for qa in captured["qa_history"])
    print("PASS: symptom and qa_history passed to selector")


def test_question_node_fallback_on_invalid_test_id():
    """Selector returns unknown test_id → fallback to first available test."""
    mock_selector = MagicMock()
    mock_selector.select_question.return_value = {
        "test_id": "test_does_not_exist", "question": "Q?", "reasoning": "",
    }
    result = question_node(_base_state(), _mock_config(selector=mock_selector))

    assert result["pending_question"]["test_id"] != "test_does_not_exist"
    assert "fallback" in result["pending_question"]["reasoning"]
    print("PASS: invalid test_id falls back to first available test")


def test_question_node_confidence_warning_true_when_low():
    """confidence_warning=True on no_tests when top probability < 0.75."""
    neo4j = MockNeo4jClient()
    all_test_ids = {
        t["id"]
        for disease in ["COPD", "Asthma", "Heart Failure", "Pulmonary Embolism",
                        "Pneumonia", "Anemia", "Interstitial Lung Disease"]
        for t in neo4j.get_available_tests([disease])
    }
    state = _base_state(
        differential=[
            {"name": "COPD",   "probability": 0.55, "log_prob": -0.60},
            {"name": "Asthma", "probability": 0.45, "log_prob": -0.80},
        ],
        questions_asked=[
            {"test_id": tid, "question": "Q?", "reasoning": ""}
            for tid in all_test_ids
        ],
    )
    result = question_node(state, _mock_config(neo4j=neo4j))
    assert result["confidence_warning"] is True
    print("PASS: confidence_warning=True when top < 0.75 at no_tests termination")


# ---------------------------------------------------------------------------
# finalize_node — unit tests
# ---------------------------------------------------------------------------

def test_finalize_node_includes_new_fields():
    """finalize_node report contains confidence_warning and finalization_reason."""
    state = _base_state(
        differential=[
            {"name": "COPD",   "probability": 0.70, "log_prob": -0.36},
            {"name": "Asthma", "probability": 0.30, "log_prob": -1.20},
        ],
        finalization_reason="no_tests",
        confidence_warning=True,
        questions_asked=[
            {"test_id": "test_fev1", "question": "What is FEV1?", "reasoning": "discriminates"},
        ],
    )
    result = finalize_node(state)
    report = result["final_diagnosis"]

    assert report["finalization_reason"] == "no_tests"
    assert report["confidence_warning"]  is True
    assert report["primary_diagnosis"]   == "COPD"
    assert report["total_questions"]     == 1
    print("PASS: finalize_node includes finalization_reason and confidence_warning")


def test_finalize_node_defaults_reason_to_confidence_gate():
    """finalize_node defaults finalization_reason to 'confidence_gate' if not set."""
    state = _base_state(
        finalization_reason=None,
        confidence_warning=False,
    )
    result = finalize_node(state)
    assert result["final_diagnosis"]["finalization_reason"] == "confidence_gate"
    print("PASS: missing finalization_reason defaults to 'confidence_gate'")


def test_finalize_node_empty_differential():
    """Empty differential returns error dict without raising."""
    state = _base_state(differential=[])
    result = finalize_node(state)
    assert "error" in result["final_diagnosis"]
    print("PASS: empty differential returns error dict")


# ---------------------------------------------------------------------------
# SessionNotFoundError — C3 fix
# ---------------------------------------------------------------------------

def test_submit_answer_raises_on_unknown_session():
    """submit_answer raises SessionNotFoundError for a non-existent session_id."""
    orch = DiagnosticOrchestrator(MockQdrantClient(), MockNeo4jClient())
    with pytest.raises(SessionNotFoundError):
        orch.submit_answer("non-existent-session-id-00000", "some answer")
    print("PASS: SessionNotFoundError raised for unknown session_id")


# ---------------------------------------------------------------------------
# Integration test (requires Ollama)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="requires Ollama — run manually when Ollama is available")
def test_full_workflow():
    """Full end-to-end session with mock clients and real LLM via Ollama."""
    qdrant = MockQdrantClient()
    neo4j  = MockNeo4jClient()
    orch   = DiagnosticOrchestrator(qdrant, neo4j)

    print("\n" + "=" * 60)
    print("ZIVAK - LangGraph Orchestrator Test (Mock Data)")
    print("=" * 60)

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

    answer = "0.62 (below 0.7 - obstruction pattern)"
    result = orch.submit_answer(session_id, answer)

    print(f"\n  Updated Differential:")
    for d in result["updated_differential"][:3]:
        print(f"  {d['name']}: {d['probability'] * 100:.1f}%")

    print(f"\n  Should continue : {result['should_continue']}")
    if result.get("next_question"):
        print(f"  Next question   : {result['next_question']['question']}")

    assert "updated_differential" in result
    assert "should_continue"      in result

    print("\n" + "=" * 60)
    print("Orchestrator test complete")
    print("=" * 60)


if __name__ == "__main__":
    test_sanitize_strips_control_chars()
    test_sanitize_preserves_newlines_and_tabs()
    test_sanitize_caps_at_max_input_len()
    test_sanitize_strips_whitespace()
    test_sanitize_empty_string()
    test_sanitize_preserves_unicode()
    test_question_node_confidence_gate()
    test_question_node_max_questions()
    test_question_node_no_tests_available()
    test_question_node_selects_question_and_saves_to_state()
    test_question_node_all_diseases_in_pool()
    test_question_node_passes_lr_map_to_selector()
    test_question_node_passes_symptom_and_history()
    test_question_node_fallback_on_invalid_test_id()
    test_question_node_confidence_warning_true_when_low()
    test_finalize_node_includes_new_fields()
    test_finalize_node_defaults_reason_to_confidence_gate()
    test_finalize_node_empty_differential()
    test_submit_answer_raises_on_unknown_session()
    print("\nPASS: All standalone Orchestrator tests passed")
