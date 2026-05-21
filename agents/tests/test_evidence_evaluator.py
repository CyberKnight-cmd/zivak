"""
Tests for EvidenceEvaluatorAgent
Run: pytest agents/tests/test_evidence_evaluator.py

LLM integration test (test_evidence_evaluator) requires Ollama.
All other tests are pure-Python unit tests — no LLM needed.
"""

import logging
import pytest
from unittest.mock import MagicMock

from agents.evidence_evaluator import EvidenceEvaluatorAgent, _extract_json


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fev1_edges():
    """test_fev1: RULES_IN COPD/Asthma, RULES_OUT HF/PE"""
    return [
        {"disease": "COPD",              "relationship": "RULES_IN",  "lr": 8.5},
        {"disease": "Asthma",            "relationship": "RULES_IN",  "lr": 3.0},
        {"disease": "Heart Failure",     "relationship": "RULES_OUT", "lr": 0.5},
        {"disease": "Pulmonary Embolism","relationship": "RULES_OUT", "lr": 0.6},
    ]


@pytest.fixture
def agent():
    a = EvidenceEvaluatorAgent.__new__(EvidenceEvaluatorAgent)
    return a


def _agent_with_mock_llm(response_text: str) -> EvidenceEvaluatorAgent:
    a = EvidenceEvaluatorAgent.__new__(EvidenceEvaluatorAgent)
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content=response_text)
    a.llm = mock_llm
    return a


# ---------------------------------------------------------------------------
# _extract_json — unit tests (identical pattern to selector)
# ---------------------------------------------------------------------------

def test_extract_json_clean():
    raw = '{"rules_in": [], "rules_out": []}'
    assert _extract_json(raw) == {"rules_in": [], "rules_out": []}
    print("PASS: clean JSON parsed")


def test_extract_json_markdown_fences():
    raw = '```json\n{"rules_in": [], "rules_out": []}\n```'
    result = _extract_json(raw)
    assert "rules_in" in result
    print("PASS: markdown-fenced JSON extracted")


def test_extract_json_reasoning_prefix():
    raw = (
        "Based on the answer, the result is positive.\n"
        '{"rules_in": [{"disease": "COPD", "likelihood_ratio": 8.5}], "rules_out": []}'
    )
    result = _extract_json(raw)
    assert result["rules_in"][0]["disease"] == "COPD"
    print("PASS: JSON extracted from response with leading reasoning text")


def test_extract_json_invalid_raises():
    with pytest.raises((ValueError, Exception)):
        _extract_json("no json here")
    print("PASS: invalid content raises error")


# ---------------------------------------------------------------------------
# _validate_output — unit tests
# ---------------------------------------------------------------------------

def test_validate_drops_unknown_disease(agent, fev1_edges):
    result = {
        "rules_in":  [
            {"disease": "COPD",            "likelihood_ratio": 8.5},
            {"disease": "Phantom Disease", "likelihood_ratio": 50.0},  # not in edges
        ],
        "rules_out": [],
    }
    cleaned = agent._validate_output(result, fev1_edges)
    names = [r["disease"] for r in cleaned["rules_in"]]
    assert "COPD" in names
    assert "Phantom Disease" not in names
    print("PASS: unknown disease dropped from output")


def test_validate_clamps_lr_above_100(agent, fev1_edges):
    result = {
        "rules_in":  [{"disease": "COPD", "likelihood_ratio": 9999.0}],
        "rules_out": [],
    }
    cleaned = agent._validate_output(result, fev1_edges)
    assert cleaned["rules_in"][0]["likelihood_ratio"] == 100.0
    print("PASS: LR > 100 clamped to 100.0")


def test_validate_clamps_lr_below_0001(agent, fev1_edges):
    result = {
        "rules_in":  [],
        "rules_out": [{"disease": "Heart Failure", "likelihood_ratio": 0.0}],
    }
    cleaned = agent._validate_output(result, fev1_edges)
    assert cleaned["rules_out"][0]["likelihood_ratio"] == 0.001
    print("PASS: LR <= 0 clamped to 0.001")


def test_validate_passes_through_valid_lr(agent, fev1_edges):
    result = {
        "rules_in":  [{"disease": "COPD", "likelihood_ratio": 8.5}],
        "rules_out": [{"disease": "Heart Failure", "likelihood_ratio": 0.5}],
    }
    cleaned = agent._validate_output(result, fev1_edges)
    assert cleaned["rules_in"][0]["likelihood_ratio"] == 8.5
    assert cleaned["rules_out"][0]["likelihood_ratio"] == 0.5
    print("PASS: in-range LR values pass through unchanged")


def test_validate_warns_on_empty_output(agent, fev1_edges, caplog):
    result = {"rules_in": [], "rules_out": []}
    with caplog.at_level(logging.WARNING, logger="agents.evidence_evaluator"):
        cleaned = agent._validate_output(result, fev1_edges)
    assert cleaned == {"rules_in": [], "rules_out": []}
    assert "zero information" in caplog.text
    print("PASS: empty output produces warning")


def test_validate_case_insensitive_disease_match(agent):
    edges = [{"disease": "Interstitial Lung Disease", "relationship": "RULES_IN", "lr": 20.0}]
    result = {
        "rules_in":  [{"disease": "interstitial lung disease", "likelihood_ratio": 20.0}],
        "rules_out": [],
    }
    cleaned = agent._validate_output(result, edges)
    assert len(cleaned["rules_in"]) == 1
    print("PASS: disease name matching is case-insensitive")


# ---------------------------------------------------------------------------
# _build_prompt — structural unit tests
# ---------------------------------------------------------------------------

def test_prompt_contains_both_polarity_branches(agent, fev1_edges):
    prompt = agent._build_prompt("What is FEV1?", "0.62", fev1_edges)
    assert "IF THE RESULT IS POSITIVE" in prompt
    assert "IF THE RESULT IS NEGATIVE" in prompt
    print("PASS: prompt contains both positive and negative branches")


def test_prompt_has_injection_guard(agent, fev1_edges):
    prompt = agent._build_prompt("Q?", "some answer", fev1_edges)
    assert "---BEGIN PATIENT INPUT---" in prompt
    assert "---END PATIENT INPUT---" in prompt
    print("PASS: patient answer wrapped in injection guard delimiters")


def test_prompt_positive_branch_uses_original_lr(agent, fev1_edges):
    prompt = agent._build_prompt("What is FEV1?", "0.62", fev1_edges)
    # RULES_IN COPD lr=8.5 → positive branch rules_in should show 8.5
    assert '"likelihood_ratio": 8.5' in prompt
    print("PASS: positive branch carries original RULES_IN LR")


def test_prompt_negative_branch_inverts_rules_in_lr(agent, fev1_edges):
    prompt = agent._build_prompt("What is FEV1?", "0.88", fev1_edges)
    # RULES_IN COPD lr=8.5 → negative branch rules_out should show 1/8.5 ≈ 0.1176
    assert "0.1176" in prompt
    print("PASS: negative branch inverts RULES_IN LR correctly")


def test_prompt_negative_branch_inverts_rules_out_lr(agent, fev1_edges):
    prompt = agent._build_prompt("What is FEV1?", "0.88", fev1_edges)
    # RULES_OUT HF lr=0.5 → negative branch rules_in should show 1/0.5 = 2.0
    assert '"likelihood_ratio": 2.0' in prompt
    print("PASS: negative branch inverts RULES_OUT LR correctly")


def test_prompt_without_threshold_has_no_reference(agent, fev1_edges):
    prompt = agent._build_prompt("What is FEV1?", "0.62", fev1_edges)
    assert "Reference threshold" not in prompt
    print("PASS: no threshold line when test_threshold is None")


def test_prompt_with_threshold_injected_into_step1(agent, fev1_edges):
    threshold = "FEV1/FVC < 0.70 = POSITIVE (obstruction), >= 0.70 = NEGATIVE"
    prompt = agent._build_prompt("What is FEV1?", "0.62", fev1_edges, test_threshold=threshold)
    assert "Reference threshold" in prompt
    assert "FEV1/FVC < 0.70" in prompt
    # Must appear inside Step 1, before Step 2
    step1_pos = prompt.index("Step 1")
    step2_pos = prompt.index("Step 2")
    threshold_pos = prompt.index("Reference threshold")
    assert step1_pos < threshold_pos < step2_pos
    print("PASS: threshold injected inside Step 1, before Step 2")


def test_prompt_empty_edges_renders_none_placeholders(agent):
    prompt = agent._build_prompt("What is X?", "result", [])
    assert "(none)" in prompt
    print("PASS: empty edge list renders '(none)' placeholders")


# ---------------------------------------------------------------------------
# evaluate() — retry and error behaviour (mocked LLM)
# ---------------------------------------------------------------------------

def test_evaluate_success_positive_result(fev1_edges):
    positive_json = (
        '{"rules_in":  [{"disease": "COPD", "likelihood_ratio": 8.5},'
        '               {"disease": "Asthma", "likelihood_ratio": 3.0}],'
        ' "rules_out": [{"disease": "Heart Failure", "likelihood_ratio": 0.5}]}'
    )
    agent = _agent_with_mock_llm(positive_json)
    result = agent.evaluate("What is FEV1?", "0.62 — obstruction", fev1_edges)

    assert any(r["disease"] == "COPD" for r in result["rules_in"])
    assert any(r["disease"] == "Heart Failure" for r in result["rules_out"])
    agent.llm.invoke.assert_called_once()
    print("PASS: positive result returns correct rules_in / rules_out")


def test_evaluate_raises_after_all_retries_fail(fev1_edges):
    a = EvidenceEvaluatorAgent.__new__(EvidenceEvaluatorAgent)
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("LLM unavailable")
    a.llm = mock_llm

    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        a.evaluate("Q?", "A", fev1_edges)

    assert mock_llm.invoke.call_count == 3
    print("PASS: RuntimeError raised after all 3 attempts fail")


def test_evaluate_retries_on_bad_json(fev1_edges):
    a = EvidenceEvaluatorAgent.__new__(EvidenceEvaluatorAgent)
    mock_llm = MagicMock()
    valid_json = '{"rules_in": [], "rules_out": [{"disease": "COPD", "likelihood_ratio": 0.12}]}'
    mock_llm.invoke.side_effect = [
        MagicMock(content="not json"),
        MagicMock(content="also bad"),
        MagicMock(content=valid_json),
    ]
    a.llm = mock_llm

    result = a.evaluate("Q?", "A", fev1_edges)
    assert mock_llm.invoke.call_count == 3
    print("PASS: retries until valid JSON returned on attempt 3")


def test_evaluate_passes_threshold_to_prompt(fev1_edges):
    """test_threshold appears in the prompt that reaches the LLM."""
    valid_json = '{"rules_in": [], "rules_out": []}'
    a = _agent_with_mock_llm(valid_json)

    threshold = "BNP >= 100 pg/mL = POSITIVE (heart failure), < 100 = NEGATIVE"
    a.evaluate("What is BNP?", "45 pg/mL", fev1_edges, test_threshold=threshold)

    call_prompt = a.llm.invoke.call_args[0][0]
    assert "BNP >= 100 pg/mL" in call_prompt
    assert "Reference threshold" in call_prompt
    print("PASS: test_threshold wired through to LLM prompt")


def test_evaluate_empty_output_does_not_raise(fev1_edges):
    """Empty arrays from LLM are valid — caller gets {} and a warning is logged."""
    valid_json = '{"rules_in": [], "rules_out": []}'
    a = _agent_with_mock_llm(valid_json)
    result = a.evaluate("Q?", "A", fev1_edges)
    assert result == {"rules_in": [], "rules_out": []}
    print("PASS: empty LLM output returns empty evidence without raising")


# ---------------------------------------------------------------------------
# LLM integration test (requires Ollama)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="requires Ollama — run manually when Ollama is available")
def test_evidence_evaluator():
    """End-to-end test — requires Ollama running with the configured local model."""
    agent = EvidenceEvaluatorAgent()

    question = "What is the patient's FEV1/FVC ratio?"
    answer   = "0.62 (below 0.7, indicating obstruction)"

    edges = [
        {"disease": "COPD",         "relationship": "RULES_IN",  "lr": 8.5},
        {"disease": "Asthma",       "relationship": "RULES_OUT", "lr": 0.3},
        {"disease": "Heart Failure","relationship": "RULES_OUT", "lr": 1.0},
    ]

    print("\nCalling Evidence Evaluator Agent...")
    print(f"Question: {question}")
    print(f"Answer:   {answer}")

    result = agent.evaluate(question, answer, edges)

    print(f"\nPASS: Evaluation complete:")
    print(f"  Rules in:  {result['rules_in']}")
    print(f"  Rules out: {result['rules_out']}")

    assert "rules_in"  in result
    assert "rules_out" in result
    assert len(result["rules_in"]) > 0


if __name__ == "__main__":
    test_extract_json_clean()
    test_extract_json_markdown_fences()
    test_extract_json_reasoning_prefix()
    test_extract_json_invalid_raises()
    print("\nPASS: All standalone EvidenceEvaluator tests passed")
    print("Note: LLM integration test (test_evidence_evaluator) requires Ollama.")
