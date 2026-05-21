"""
ZIVAK FastAPI — diagnostic session API

Run dev server:  uvicorn main:app --reload

Endpoints:
  GET   /health
  POST  /api/v1/sessions                 — start session, returns first question
  POST  /api/v1/sessions/{id}/answer     — submit answer, returns next question or final dx
  GET   /api/v1/sessions/{id}/diagnosis  — final report (once session is complete)

Auth: Bearer token checked against API_KEY env var.
Swap mock clients for real ones by setting USE_MOCK=false in .env.
"""

import logging
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from urllib.request import urlopen
from urllib.error import URLError

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Ollama lifecycle — kill stale process, start fresh, wait for ready #
# ------------------------------------------------------------------ #

def _kill_ollama() -> None:
    """Terminate any running Ollama process."""
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/IM", "ollama.exe"],
                capture_output=True, timeout=10,
            )
        else:
            subprocess.run(["pkill", "-f", "ollama serve"], capture_output=True, timeout=10)
        time.sleep(1)  # let the port free up
    except Exception:
        pass


def _start_ollama() -> subprocess.Popen:
    """Start `ollama serve` and wait up to 30 s for it to answer HTTP."""
    env = {**os.environ, "OLLAMA_NUM_PARALLEL": "1"}  # one inference slot → no VRAM contention
    proc = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    ollama_url = os.getenv("LOCAL_LLM_URL", "http://localhost:11434/v1").rsplit("/v1", 1)[0]
    for _ in range(30):
        try:
            urlopen(ollama_url, timeout=1)
            logger.info("Ollama ready at %s", ollama_url)
            return proc
        except (URLError, OSError):
            time.sleep(1)
    proc.kill()
    raise RuntimeError("Ollama failed to start within 30 s")


# ------------------------------------------------------------------ #
#  App lifecycle — build orchestrator once at startup                  #
# ------------------------------------------------------------------ #

from orchestrator.mock_clients import get_clients
from orchestrator.orchestrator import DiagnosticOrchestrator, SessionNotFoundError

_orchestrator: DiagnosticOrchestrator | None = None
_ollama_proc:  subprocess.Popen | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator, _ollama_proc

    use_local_llm = os.getenv("USE_LOCAL_LLM", "").lower() == "true"
    if use_local_llm:
        logger.info("USE_LOCAL_LLM=true — restarting Ollama for clean GPU state")
        _kill_ollama()
        _ollama_proc = _start_ollama()

    use_mock = os.getenv("USE_MOCK", "true").lower() != "false"
    qdrant, neo4j = get_clients(use_mock=use_mock)
    _orchestrator = DiagnosticOrchestrator(qdrant, neo4j)
    logger.info("Orchestrator ready (mock=%s)", use_mock)
    yield

    logger.info("Shutdown")
    if _ollama_proc is not None:
        _ollama_proc.terminate()
        _ollama_proc = None


app = FastAPI(title="ZIVAK Diagnostic API", version="0.1.0", lifespan=lifespan)

# ------------------------------------------------------------------ #
#  CORS — allow React dev server and same-origin in prod              #
# ------------------------------------------------------------------ #

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# ------------------------------------------------------------------ #
#  Auth — static Bearer token (swap for JWT when user accounts exist) #
# ------------------------------------------------------------------ #

_bearer  = HTTPBearer()
_API_KEY = os.getenv("API_KEY", "")


def _verify(creds: HTTPAuthorizationCredentials = Security(_bearer)) -> None:
    if not _API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY not set on server")
    if creds.credentials != _API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


# ------------------------------------------------------------------ #
#  Request / response models                                          #
# ------------------------------------------------------------------ #

class StartRequest(BaseModel):
    symptom: str = Field(..., min_length=3, max_length=1000,
                         description="Patient's symptom description in their own words")

class AnswerRequest(BaseModel):
    answer: str = Field(..., min_length=1, max_length=1000,
                        description="Patient's answer to the current diagnostic question")

# ------------------------------------------------------------------ #
#  Routes                                                             #
# ------------------------------------------------------------------ #

@app.get("/health", tags=["Meta"])
def health():
    return {"status": "ok"}


@app.post("/api/v1/sessions", dependencies=[Depends(_verify)], tags=["Diagnostic"])
def start_session(body: StartRequest):
    """
    Start a new diagnostic session.

    Returns the initial differential and the first question to show the patient.
    Store the returned `session_id` — all subsequent calls need it.
    """
    try:
        session_id, result = _orchestrator.start_session(body.symptom)
        return result
    except RuntimeError as e:
        logger.error("start_session failed: %s", e)
        raise HTTPException(status_code=503, detail="Diagnostic engine unavailable — try again shortly")


@app.post("/api/v1/sessions/{session_id}/answer", dependencies=[Depends(_verify)], tags=["Diagnostic"])
def submit_answer(session_id: str, body: AnswerRequest):
    """
    Submit the patient's answer to the current question.

    Returns the updated differential and the next question.
    When `should_continue` is false, `final_diagnosis` is populated and the session is complete.
    """
    try:
        return _orchestrator.submit_answer(session_id, body.answer)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    except RuntimeError as e:
        logger.error("submit_answer session=%s failed: %s", session_id, e)
        raise HTTPException(status_code=503, detail="Diagnostic engine unavailable — try again shortly")


@app.get("/api/v1/sessions/{session_id}/diagnosis", dependencies=[Depends(_verify)], tags=["Diagnostic"])
def get_diagnosis(session_id: str):
    """
    Retrieve the final diagnosis for a completed session.

    Returns 409 if the session has not yet reached a final diagnosis.
    """
    try:
        result = _orchestrator.get_final_diagnosis(session_id)
        if "error" in result:
            raise HTTPException(status_code=409, detail=result["error"])
        return result
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found or expired")
