"""
Orchestrator — LangGraph diagnostic workflow.

Graph topology:
  START → seed → qa ←─────────────┐
                  │                │
                  ├─ not done ─────┘  (loop: next question)
                  └─ done ──→ finalize → END

The qa node interrupts after selecting each question.
Caller resumes with Command(resume=answer) to continue.

Checkpointer (state persistence):
  MemorySaver()                  — dev / tests (default)
  langgraph-checkpoint-redis     — production (swap via checkpointer param)
"""

import logging
import re
import uuid
from typing import Dict, List, Optional, Tuple

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing import TypedDict

from agents.evidence_evaluator import EvidenceEvaluatorAgent
from agents.question_selector import QuestionSelectorAgent
from engines.confidence_judge import ConfidenceJudge
from engines.differential import DifferentialEngine

logger = logging.getLogger(__name__)

MAX_QUESTIONS = 10
MAX_INPUT_LEN  = 1000


def _sanitize(text: str) -> str:
    """Strip control characters and cap length before text enters any LLM prompt."""
    text = text.strip()[:MAX_INPUT_LEN]
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)


# ------------------------------------------------------------------ #
#  Graph state schema                                                  #
# ------------------------------------------------------------------ #

class DiagnosticState(TypedDict):
    user_input:        str
    symptom_match:     Dict
    differential:      List[Dict]
    evidence_history:  List[Dict]
    questions_asked:   List[Dict]
    should_finalize:   bool
    judge_details:     Dict
    final_diagnosis:   Optional[Dict]
    previous_top_prob: Optional[float]  # top probability snapshot before the last answer


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _restore_engine(state: DiagnosticState) -> DifferentialEngine:
    """Reconstruct DifferentialEngine from persisted state fields."""
    engine = DifferentialEngine()
    engine.differential    = state.get("differential", [])
    engine.evidence_history = state.get("evidence_history", [])
    return engine


# ------------------------------------------------------------------ #
#  Nodes                                                               #
# ------------------------------------------------------------------ #

def seed_node(state: DiagnosticState, config: RunnableConfig) -> Dict:
    """
    Symptom text → Qdrant semantic search → Neo4j differential seed.
    Produces the initial probability distribution.
    """
    cfg    = config["configurable"]
    qdrant = cfg["qdrant"]
    neo4j  = cfg["neo4j"]

    symptom_match = qdrant.search_symptom(state["user_input"])
    diseases      = neo4j.get_initial_differential(symptom_match["symptom_id"])

    engine       = DifferentialEngine()
    differential = engine.initialize(diseases)

    logger.info("seed: symptom=%r n_diseases=%d", symptom_match.get("clinical_term"), len(differential))

    return {
        "symptom_match":   symptom_match,
        "differential":    differential,
        "evidence_history": engine.evidence_history,
    }


def qa_node(state: DiagnosticState, config: RunnableConfig) -> Dict:
    """
    One Q&A turn:
      1. Check termination (confidence gate / max questions / no tests)
      2. Select the highest-yield question (QuestionSelectorAgent)
      3. interrupt() — pause and surface the question to the caller
      4. Resumed with the user's answer
      5. Map answer → evidence (EvidenceEvaluatorAgent)
      6. Bayesian update (DifferentialEngine)
    """
    cfg       = config["configurable"]
    neo4j     = cfg["neo4j"]
    selector: QuestionSelectorAgent  = cfg["selector"]
    evaluator: EvidenceEvaluatorAgent = cfg["evaluator"]

    engine = _restore_engine(state)
    judge  = ConfidenceJudge()

    # --- Termination checks (before asking the next question) ---
    should_stop, judge_details = judge.should_finalize(
        state["differential"],
        engine.get_evidence_count(),
        state.get("previous_top_prob"),
    )
    if should_stop:
        logger.info("qa: confidence gate — %s", judge_details.get("reason"))
        return {"should_finalize": True, "judge_details": judge_details}

    if len(state["questions_asked"]) >= MAX_QUESTIONS:
        logger.warning("qa: MAX_QUESTIONS cap")
        return {"should_finalize": True, "judge_details": {"reason": "Max questions reached"}}

    top_diseases    = [d["name"] for d in state["differential"][:5]]
    asked_ids       = {q["test_id"] for q in state["questions_asked"]}
    available_tests = [
        t for t in neo4j.get_available_tests(top_diseases)
        if t["id"] not in asked_ids
    ]
    if not available_tests:
        logger.warning("qa: no tests remain for %s", top_diseases)
        return {"should_finalize": True, "judge_details": {"reason": "No tests available"}}

    # --- Select question ---
    question  = selector.select_question(state["differential"], available_tests)
    valid_ids = {t["id"] for t in available_tests}
    if question.get("test_id") not in valid_ids:
        fallback = available_tests[0]
        logger.warning(
            "qa: selector returned out-of-scope test_id %r — falling back to %s",
            question.get("test_id"), fallback["id"],
        )
        question = {
            "test_id":   fallback["id"],
            "question":  f"What is the result of {fallback['name']}?",
            "reasoning": "fallback: selector returned a test outside the available list",
        }
    logger.info("qa: selected test=%s", question.get("test_id"))

    # --- Interrupt: surface question, wait for human answer ---
    answer: str = interrupt(question)

    # --- Process answer ---
    edges    = neo4j.get_test_edges(question["test_id"])
    evidence = evaluator.evaluate(question["question"], _sanitize(str(answer)), edges)
    updated  = engine.update(evidence)

    logger.info(
        "qa: n_evidence=%d top=%s(%.2f)",
        engine.get_evidence_count(),
        updated[0]["name"] if updated else "none",
        updated[0]["probability"] if updated else 0,
    )

    # Snapshot the top probability BEFORE this answer was applied;
    # next turn's judge uses it to detect a sudden drop in leader confidence.
    pre_update_top = state["differential"][0]["probability"] if state["differential"] else None

    return {
        "differential":      updated,
        "evidence_history":  engine.evidence_history,
        "questions_asked":   state["questions_asked"] + [question],
        "should_finalize":   False,
        "previous_top_prob": pre_update_top,
    }


def finalize_node(state: DiagnosticState) -> Dict:
    """Compile the final diagnosis report from completed state."""
    if not state.get("differential"):
        return {"final_diagnosis": {"error": "No differential available"}}

    top = state["differential"][0]
    report = {
        "primary_diagnosis": top["name"],
        "confidence":        top["probability"],
        "differential":      state["differential"][:3],
        "evidence_chain": [
            {"question": q["question"], "reasoning": q.get("reasoning", "")}
            for q in state.get("questions_asked", [])
        ],
        "total_questions": len(state.get("questions_asked", [])),
    }
    logger.info("finalize: %s confidence=%.2f", top["name"], top["probability"])
    return {"final_diagnosis": report}


# ------------------------------------------------------------------ #
#  Graph compilation                                                   #
# ------------------------------------------------------------------ #

def _build_graph(checkpointer):
    builder = StateGraph(DiagnosticState)

    builder.add_node("seed",     seed_node)
    builder.add_node("qa",       qa_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "seed")
    builder.add_edge("seed", "qa")
    builder.add_conditional_edges(
        "qa",
        lambda state: "finalize" if state.get("should_finalize") else "qa",
    )
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

class SessionNotFoundError(Exception):
    pass


class DiagnosticOrchestrator:
    """
    Thin wrapper around the LangGraph diagnostic graph.

    Agents are instantiated once here and injected via configurable
    so LangGraph nodes remain pure functions (no hidden state).

    Usage:
        orch = DiagnosticOrchestrator(qdrant, neo4j)
        session_id, result = orch.start_session("chest feels heavy")
        question = result["next_question"]

        while question:
            answer = get_answer_from_user(question)
            result = orch.submit_answer(session_id, answer)
            question = result["next_question"]

        report = result["final_diagnosis"]
    """

    def __init__(self, qdrant_client, neo4j_client, checkpointer=None):
        self._qdrant    = qdrant_client
        self._neo4j     = neo4j_client
        self._selector  = QuestionSelectorAgent()
        self._evaluator = EvidenceEvaluatorAgent()
        self._graph     = _build_graph(checkpointer or MemorySaver())

    # ---------------------------------------------------------------- #

    def start_session(self, user_input: str) -> Tuple[str, Dict]:
        """
        Start a new diagnostic session.

        Runs seed → first qa interrupt.
        Returns (session_id, result) where result['next_question']
        is the first question ready for the user.
        """
        user_input = _sanitize(user_input)
        session_id = str(uuid.uuid4())
        config     = self._config(session_id)

        initial: DiagnosticState = {
            "user_input":        user_input,
            "symptom_match":     {},
            "differential":      [],
            "evidence_history":  [],
            "questions_asked":   [],
            "should_finalize":   False,
            "judge_details":     {},
            "final_diagnosis":   None,
            "previous_top_prob": None,
        }

        snapshot = self._graph.invoke(initial, config)

        return session_id, {
            "session_id":          session_id,
            "symptom_match":       snapshot.get("symptom_match", {}),
            "initial_differential": snapshot.get("differential", []),
            "next_question":       self._pending_question(config),
        }

    def submit_answer(self, session_id: str, answer: str) -> Dict:
        """
        Resume the graph with the user's answer to the current question.

        Returns updated differential and the next question (None if done).
        """
        config   = self._config(session_id)
        snapshot = self._graph.invoke(Command(resume=_sanitize(answer)), config)

        next_q         = self._pending_question(config)
        final_diagnosis = snapshot.get("final_diagnosis")
        should_continue = next_q is not None and final_diagnosis is None

        return {
            "updated_differential": snapshot.get("differential", []),
            "should_continue":      should_continue,
            "next_question":        next_q,
            "judge_details":        snapshot.get("judge_details", {}),
            "final_diagnosis":      final_diagnosis,
        }

    def get_final_diagnosis(self, session_id: str) -> Dict:
        """Return the final diagnosis from a completed session."""
        snapshot = self._graph.get_state(self._config(session_id))
        result   = snapshot.values.get("final_diagnosis")
        return result or {"error": "Session not finalized yet"}

    # ---------------------------------------------------------------- #

    def _config(self, session_id: str) -> Dict:
        return {
            "configurable": {
                "thread_id": session_id,
                "qdrant":    self._qdrant,
                "neo4j":     self._neo4j,
                "selector":  self._selector,
                "evaluator": self._evaluator,
            }
        }

    def _pending_question(self, config: Dict) -> Optional[Dict]:
        """Extract the interrupt value (the pending question) from graph state."""
        state = self._graph.get_state(config)
        if state.tasks and state.tasks[0].interrupts:
            return state.tasks[0].interrupts[0].value
        return None
