"""
FastAPI layer tests — no real HTTP, no Groq calls, no embedding model.
DiagnosticOrchestrator is patched at the class level so the lifespan
receives our mock instead of building a real instance.
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from orchestrator.orchestrator import SessionNotFoundError


# ------------------------------------------------------------------ #
#  Fixtures                                                            #
# ------------------------------------------------------------------ #

@pytest.fixture
def mock_orch():
    m = MagicMock()
    m.start_session.return_value = (
        "test-session-123",
        {
            "session_id":            "test-session-123",
            "symptom_match":         {"clinical_term": "Exertional Dyspnoea", "symptom_id": "HP:0002875", "score": 0.90},
            "initial_differential":  [{"name": "COPD", "probability": 0.45}, {"name": "Asthma", "probability": 0.35}],
            "next_question":         {"question": "FEV1/FVC ratio?", "test_id": "test_fev1", "reasoning": "..."},
        },
    )
    m.submit_answer.return_value = {
        "updated_differential": [{"name": "COPD", "probability": 0.82}, {"name": "Asthma", "probability": 0.18}],
        "should_continue":      True,
        "next_question":        {"question": "Chest X-ray?", "test_id": "test_cxr_hyp", "reasoning": "..."},
        "judge_details":        {"reason": "Need more evidence (1/4)"},
        "final_diagnosis":      None,
    }
    m.get_final_diagnosis.return_value = {
        "primary_diagnosis": "COPD",
        "confidence":        0.87,
        "differential":      [{"name": "COPD", "probability": 0.87}],
        "evidence_chain":    [],
        "total_questions":   3,
    }
    return m


@pytest.fixture
def client(mock_orch):
    """
    TestClient whose lifespan builds a mock orchestrator.
    Patching DiagnosticOrchestrator (the class) means lifespan gets
    our mock when it calls DiagnosticOrchestrator(qdrant, neo4j).
    """
    with patch("main.DiagnosticOrchestrator", return_value=mock_orch), \
         patch("main.get_clients", return_value=(MagicMock(), MagicMock())):
        with TestClient(main.app) as c:
            yield c


import main  # noqa: E402 — imported after fixtures so patches apply correctly

HEADERS = {"Authorization": "Bearer zivak-dev-key"}


# ------------------------------------------------------------------ #
#  Tests                                                               #
# ------------------------------------------------------------------ #

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_health_no_auth_required(client):
    r = client.get("/health")
    assert r.status_code == 200


def test_start_session(client):
    r = client.post("/api/v1/sessions", json={"symptom": "chest feels heavy"}, headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert "session_id" in data
    assert "initial_differential" in data
    assert data["next_question"]["test_id"] == "test_fev1"


def test_start_session_requires_auth(client):
    r = client.post("/api/v1/sessions", json={"symptom": "chest feels heavy"})
    # FastAPI HTTPBearer returns 403 for missing creds, 401 for wrong token
    assert r.status_code in (401, 403)


def test_start_session_bad_token(client):
    r = client.post("/api/v1/sessions", json={"symptom": "chest feels heavy"},
                    headers={"Authorization": "Bearer wrong-key"})
    assert r.status_code == 401


def test_start_session_too_short(client):
    r = client.post("/api/v1/sessions", json={"symptom": "hi"}, headers=HEADERS)
    assert r.status_code == 422


def test_submit_answer(client):
    r = client.post(
        "/api/v1/sessions/test-session-123/answer",
        json={"answer": "0.62 below 0.7"},
        headers=HEADERS,
    )
    assert r.status_code == 200
    data = r.json()
    assert data["should_continue"] is True
    assert data["updated_differential"][0]["name"] == "COPD"
    assert data["next_question"]["test_id"] == "test_cxr_hyp"


def test_submit_answer_session_not_found(client, mock_orch):
    mock_orch.submit_answer.side_effect = SessionNotFoundError("gone")
    r = client.post(
        "/api/v1/sessions/no-such-session/answer",
        json={"answer": "yes"},
        headers=HEADERS,
    )
    assert r.status_code == 404
    mock_orch.submit_answer.side_effect = None  # reset for subsequent tests


def test_submit_answer_engine_failure(client, mock_orch):
    mock_orch.submit_answer.side_effect = RuntimeError("LLM unavailable")
    r = client.post(
        "/api/v1/sessions/any/answer",
        json={"answer": "yes"},
        headers=HEADERS,
    )
    assert r.status_code == 503
    mock_orch.submit_answer.side_effect = None


def test_get_diagnosis(client):
    r = client.get("/api/v1/sessions/test-session-123/diagnosis", headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert data["primary_diagnosis"] == "COPD"
    assert data["confidence"] == 0.87
