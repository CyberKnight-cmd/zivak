"""
Tests for QuestionSelectorAgent
Run: pytest agents/tests/test_question_selector.py

LLM integration test (test_question_selector) requires Ollama.
All other tests are pure-Python unit tests — no LLM needed.
"""

import pytest
from unittest.mock import MagicMock

from agents.question_selector import QuestionSelectorAgent, _extract_json


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def differential():
    return [
        {"name": "COPD",          "probability": 0.40},
        {"name": "Asthma",        "probability": 0.35},
        {"name": "Heart Failure", "probability": 0.25},
    ]


@pytest.fixture
def available_tests():
    return [
        {"id": "test_fev1",  "name": "FEV1/FVC spirometry", "diseases": ["COPD", "Asthma"]},
        {"id": "test_bnp",   "name": "BNP / NT-proBNP",     "diseases": ["Heart Failure"]},
        {"id": "test_hrct",  "name": "HRCT chest",          "diseases": ["COPD", "Interstitial Lung Disease"]},
    ]


@pytest.fixture
def test_lr_map():
    return {
        "test_fev1": [
            {"disease": "COPD",   "relationship": "RULES_IN",  "lr": 8.5},
            {"disease": "Asthma", "relationship": "RULES_IN",  "lr": 3.0},
            {"disease": "Heart Failure", "relationship": "RULES_OUT", "lr": 0.5},
        ],
        "test_bnp": [
            {"disease": "Heart Failure", "relationship": "RULES_IN",  "lr": 9.2},
            {"disease": "COPD",          "relationship": "RULES_OUT", "lr": 0.5},
        ],
        "test_hrct": [
            {"disease": "Interstitial Lung Disease", "relationship": "RULES_IN", "lr": 20.0},
            {"disease": "COPD",                      "relationship": "RULES_IN", "lr": 3.5},
        ],
    }


def _agent_with_mock_llm(response_text: str) -> QuestionSelectorAgent:
    """Return an agent whose LLM returns a fixed string."""
    agent = QuestionSelectorAgent.__new__(QuestionSelectorAgent)
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content=response_text)
    agent.llm = mock_llm
    return agent


# ---------------------------------------------------------------------------
# _extract_json — unit tests (no LLM)
# ---------------------------------------------------------------------------

def test_extract_json_clean():
    raw = '{"question": "What is FEV1?", "test_id": "test_fev1", "reasoning": "best"}'
    result = _extract_json(raw)
    assert result["test_id"] == "test_fev1"
    print("PASS: clean JSON parsed")


def test_extract_json_markdown_fences():
    raw = '```json\n{"question": "Q?", "test_id": "test_bnp", "reasoning": "r"}\n```'
    result = _extract_json(raw)
    assert result["test_id"] == "test_bnp"
    print("PASS: markdown-fenced JSON extracted")


def test_extract_json_reasoning_prefix():
    raw = (
        "Let me think about this carefully.\n"
        'The best test is BNP. {"question": "What is BNP?", "test_id": "test_bnp", "reasoning": "HF"}'
    )
    result = _extract_json(raw)
    assert result["test_id"] == "test_bnp"
    print("PASS: JSON extracted from response with leading reasoning text")


def test_extract_json_invalid_raises():
    with pytest.raises((ValueError, Exception)):
        _extract_json("no json here at all")
    print("PASS: invalid content raises ValueError")


# ---------------------------------------------------------------------------
# _build_prompt — structural unit tests (no LLM)
# ---------------------------------------------------------------------------

def test_prompt_contains_differential(differential, available_tests):
    agent = QuestionSelectorAgent.__new__(QuestionSelectorAgent)
    prompt = agent._build_prompt(differential, available_tests)

    assert "COPD" in prompt
    assert "40.0%" in prompt
    assert "Asthma" in prompt
    assert "35.0%" in prompt
    print("PASS: prompt contains differential with percentages")


def test_prompt_default_no_lr_hints(differential, available_tests):
    agent = QuestionSelectorAgent.__new__(QuestionSelectorAgent)
    prompt = agent._build_prompt(differential, available_tests)

    assert "relevant for" in prompt
    assert "LR≈" not in prompt
    assert "Available tests:" in prompt
    print("PASS: default prompt uses 'relevant for' format, no LR values")


def test_prompt_with_lr_map_shows_lr_values(differential, available_tests, test_lr_map):
    agent = QuestionSelectorAgent.__new__(QuestionSelectorAgent)
    prompt = agent._build_prompt(differential, available_tests, test_lr_map=test_lr_map)

    # High-LR test should be clearly visible
    assert "LR≈20.0" in prompt   # test_hrct for ILD
    assert "LR≈9.2" in prompt    # test_bnp for Heart Failure
    assert "LR≈8.5" in prompt    # test_fev1 for COPD
    # RULES_OUT edges should NOT appear (only RULES_IN are shown)
    assert "LR≈0.5" not in prompt
    # Header updated
    assert "LR values show" in prompt
    assert "relevant for" not in prompt
    print("PASS: LR map injects numeric LR values for RULES_IN edges only")


def test_prompt_lr_criterion_updated_when_lr_map(differential, available_tests, test_lr_map):
    agent = QuestionSelectorAgent.__new__(QuestionSelectorAgent)

    prompt_no_lr = agent._build_prompt(differential, available_tests)
    prompt_with_lr = agent._build_prompt(differential, available_tests, test_lr_map=test_lr_map)

    assert "Has strong likelihood ratios" in prompt_no_lr
    assert "Prefer tests with higher LR values" in prompt_with_lr
    print("PASS: selection criterion wording updated when LR map is provided")


def test_prompt_with_symptom(differential, available_tests):
    agent = QuestionSelectorAgent.__new__(QuestionSelectorAgent)
    prompt = agent._build_prompt(
        differential, available_tests,
        symptom="I get breathless climbing stairs",
    )

    assert "I get breathless climbing stairs" in prompt
    assert "Patient presenting complaint" in prompt
    print("PASS: symptom injected into prompt")


def test_prompt_with_qa_history_with_answers(differential, available_tests):
    agent = QuestionSelectorAgent.__new__(QuestionSelectorAgent)
    history = [
        {"question": "What is the BNP result?",       "answer": "Normal — 45 pg/mL"},
        {"question": "What is the FEV1/FVC ratio?",   "answer": "0.88 — normal"},
    ]
    prompt = agent._build_prompt(differential, available_tests, qa_history=history)

    assert "Evidence collected so far:" in prompt
    assert "Turn 1" in prompt
    assert "Normal — 45 pg/mL" in prompt
    assert "Turn 2" in prompt
    assert "0.88 — normal" in prompt
    print("PASS: Q&A history with answers rendered in prompt")


def test_prompt_with_qa_history_without_answers(differential, available_tests):
    agent = QuestionSelectorAgent.__new__(QuestionSelectorAgent)
    history = [
        {"question": "What is the BNP result?"},
        {"question": "What is the FEV1/FVC ratio?"},
    ]
    prompt = agent._build_prompt(differential, available_tests, qa_history=history)

    assert "Evidence collected so far:" in prompt
    assert "Turn 1" in prompt
    assert "Turn 2" in prompt
    # No answer text — no colon suffix
    assert "Turn 1 — What is the BNP result?\n" in prompt
    print("PASS: Q&A history without answers renders question only")


def test_prompt_caps_differential_at_top5(available_tests):
    agent = QuestionSelectorAgent.__new__(QuestionSelectorAgent)
    long_diff = [
        {"name": f"Disease{i}", "probability": 1 / 8}
        for i in range(8)
    ]
    prompt = agent._build_prompt(long_diff, available_tests)

    # Only first 5 diseases should appear
    assert "Disease0" in prompt
    assert "Disease4" in prompt
    assert "Disease5" not in prompt
    assert "Disease7" not in prompt
    print("PASS: differential capped at top 5 in prompt")


def test_prompt_no_context_block_when_not_provided(differential, available_tests):
    agent = QuestionSelectorAgent.__new__(QuestionSelectorAgent)
    prompt = agent._build_prompt(differential, available_tests)

    assert "Patient presenting complaint" not in prompt
    assert "Evidence collected so far" not in prompt
    print("PASS: no context block when symptom and qa_history are None")


# ---------------------------------------------------------------------------
# select_question — retry and error behaviour (mocked LLM)
# ---------------------------------------------------------------------------

def test_select_question_success(differential, available_tests):
    valid_json = '{"question": "What is FEV1/FVC?", "test_id": "test_fev1", "reasoning": "top discriminator"}'
    agent = _agent_with_mock_llm(valid_json)

    result = agent.select_question(differential, available_tests)

    assert result["test_id"] == "test_fev1"
    assert result["question"] == "What is FEV1/FVC?"
    agent.llm.invoke.assert_called_once()
    print("PASS: valid response returned on first attempt")


def test_select_question_raises_after_all_retries_fail(differential, available_tests):
    agent = QuestionSelectorAgent.__new__(QuestionSelectorAgent)
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("LLM unavailable")
    agent.llm = mock_llm

    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        agent.select_question(differential, available_tests)

    assert mock_llm.invoke.call_count == 3
    print("PASS: RuntimeError raised after all 3 attempts fail")


def test_select_question_retries_on_bad_json(differential, available_tests):
    agent = QuestionSelectorAgent.__new__(QuestionSelectorAgent)
    mock_llm = MagicMock()
    # First two calls return garbage; third returns valid JSON
    valid_json = '{"question": "What is BNP?", "test_id": "test_bnp", "reasoning": "HF"}'
    mock_llm.invoke.side_effect = [
        MagicMock(content="not json at all"),
        MagicMock(content="still bad"),
        MagicMock(content=valid_json),
    ]
    agent.llm = mock_llm

    result = agent.select_question(differential, available_tests)

    assert result["test_id"] == "test_bnp"
    assert mock_llm.invoke.call_count == 3
    print("PASS: retries until valid JSON returned on attempt 3")


def test_select_question_passes_lr_map_and_symptom(differential, available_tests, test_lr_map):
    """Verify new parameters are forwarded into the prompt sent to the LLM."""
    valid_json = '{"question": "What is HRCT?", "test_id": "test_hrct", "reasoning": "highest LR"}'
    agent = _agent_with_mock_llm(valid_json)

    agent.select_question(
        differential, available_tests,
        test_lr_map=test_lr_map,
        symptom="dry cough for 6 months",
        qa_history=[{"question": "BNP?", "answer": "Normal"}],
    )

    call_args = agent.llm.invoke.call_args[0][0]
    assert "LR≈20.0" in call_args          # LR map wired through
    assert "dry cough for 6 months" in call_args  # symptom wired through
    assert "BNP?" in call_args             # history wired through
    print("PASS: test_lr_map, symptom, and qa_history all reach the LLM prompt")


# ---------------------------------------------------------------------------
# LLM integration test (requires Ollama)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="requires Ollama — run manually when Ollama is available")
def test_question_selector_llm():
    """End-to-end test — requires Ollama running with the configured local model."""
    agent = QuestionSelectorAgent()

    differential = [
        {"name": "COPD",          "probability": 0.40},
        {"name": "Asthma",        "probability": 0.35},
        {"name": "Heart Failure", "probability": 0.25},
    ]
    available_tests = [
        {"id": "test_fev1", "name": "FEV1/FVC ratio", "diseases": ["COPD", "Asthma"]},
        {"id": "test_bnp",  "name": "BNP blood test", "diseases": ["Heart Failure"]},
    ]

    print("\nCalling Question Selector Agent...")
    result = agent.select_question(differential, available_tests)

    print(f"\nPASS: Selected question: {result['question']}")
    print(f"  Test ID:   {result['test_id']}")
    print(f"  Reasoning: {result['reasoning']}")

    assert "question" in result
    assert "test_id"  in result


if __name__ == "__main__":
    test_extract_json_clean()
    test_extract_json_markdown_fences()
    test_extract_json_reasoning_prefix()
    test_extract_json_invalid_raises()
    test_prompt_contains_differential(
        [{"name": "COPD", "probability": 0.40}, {"name": "Asthma", "probability": 0.35}, {"name": "HF", "probability": 0.25}],
        [{"id": "test_fev1", "name": "FEV1", "diseases": ["COPD"]}, {"id": "test_bnp", "name": "BNP", "diseases": ["HF"]}],
    )
    print("\nPASS: All standalone QuestionSelector tests passed")
    print("Note: LLM integration test (test_question_selector_llm) requires Ollama.")
