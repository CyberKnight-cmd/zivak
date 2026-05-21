"""
Orchestrator — LangGraph diagnostic workflow.

Graph topology:
  START → seed → question ──────────────────────┐
                     │                           │
                     ├─ not done → answer ───────┘  (loop)
                     └─ done ──→ finalize → END

question_node selects the next test and saves it to state, then returns.
answer_node reads that question from state, interrupts to surface it to the
caller, and on resume evaluates the answer and applies the Bayesian update.

This two-node split is the C1 fix: question selection runs exactly once per
turn. Previously qa_node ran the selector TWICE per turn (once before the
interrupt, once again on resume), wasting LLM calls and risking mismatched
question/evidence pairs at temperature=1.

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
MAX_INPUT_LEN = 1000


def _sanitize(text: str) -> str:
    """Strip control characters and cap length before text enters any LLM prompt."""
    text = text.strip()[:MAX_INPUT_LEN]
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)


# ------------------------------------------------------------------ #
#  Graph state schema                                                  #
# ------------------------------------------------------------------ #

class DiagnosticState(TypedDict):
    user_input:          str
    symptom_match:       Dict
    differential:        List[Dict]
    evidence_history:    List[Dict]
    questions_asked:     List[Dict]
    should_finalize:     bool
    judge_details:       Dict
    final_diagnosis:     Optional[Dict]
    previous_top_prob:   Optional[float]   # top prob snapshot before the last answer
    pending_question:    Optional[Dict]    # question_node → answer_node handoff (C1 fix)
    finalization_reason: Optional[str]    # "confidence_gate" | "no_tests" | "max_questions"
    confidence_warning:  bool             # top_probability < threshold at termination (M4)


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _restore_engine(state: DiagnosticState) -> DifferentialEngine:
    """Reconstruct DifferentialEngine from persisted state fields."""
    engine = DifferentialEngine()
    engine.differential     = state.get("differential", [])
    engine.evidence_history = state.get("evidence_history", [])
    return engine


def _top_confidence_warning(state: DiagnosticState) -> bool:
    """True when the current leader is below the ConfidenceJudge threshold."""
    if not state.get("differential"):
        return True
    return state["differential"][0]["probability"] < ConfidenceJudge().min_top_confidence


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
    # Accept both single symptom_id (mock) and symptom_ids list (real Qdrant client).
    symptom_ids   = symptom_match.get("symptom_ids") or [symptom_match["symptom_id"]]
    diseases      = neo4j.get_initial_differential(symptom_ids)

    engine       = DifferentialEngine()
    differential = engine.initialize(diseases)

    logger.info(
        "seed: symptom=%r n_diseases=%d",
        symptom_match.get("clinical_term"), len(differential),
    )

    return {
        "symptom_match":    symptom_match,
        "differential":     differential,
        "evidence_history": engine.evidence_history,
    }


def question_node(state: DiagnosticState, config: RunnableConfig) -> Dict:
    """
    Termination check → test pool → question selection → save to state.

    Runs exactly once per question turn. Returns pending_question to state
    so answer_node can read the correct question on resume without re-running
    this node (C1 fix — no double selector call).
    """
    cfg      = config["configurable"]
    neo4j    = cfg["neo4j"]
    selector: QuestionSelectorAgent = cfg["selector"]

    engine = _restore_engine(state)
    judge  = ConfidenceJudge()

    # --- Confidence gate ---
    should_stop, judge_details = judge.should_finalize(
        state["differential"],
        engine.get_evidence_count(),
        state.get("previous_top_prob"),
    )
    if should_stop:
        logger.info("question: confidence gate — %s", judge_details.get("reason"))
        return {
            "should_finalize":     True,
            "judge_details":       judge_details,
            "finalization_reason": "confidence_gate",
            "confidence_warning":  judge_details.get("confidence_warning", False),
        }

    # --- MAX_QUESTIONS cap ---
    if len(state["questions_asked"]) >= MAX_QUESTIONS:
        logger.warning("question: MAX_QUESTIONS cap reached")
        return {
            "should_finalize":     True,
            "judge_details":       {"reason": "Max questions reached"},
            "finalization_reason": "max_questions",
            "confidence_warning":  _top_confidence_warning(state),
        }

    # --- Build test pool from ALL diseases (M1 fix: was top-5 only) ---
    # Use disease_id (DOID) when available (real Neo4j client); fall back to name
    # for backward compat with MockNeo4jClient which keys by disease name.
    all_disease_ids = [
        d.get("disease_id", d["name"]) for d in state["differential"]
    ]
    asked_ids       = {q["test_id"] for q in state["questions_asked"]}
    available_tests = [
        t for t in neo4j.get_available_tests(all_disease_ids)
        if t["id"] not in asked_ids
    ]
    if not available_tests:
        logger.warning("question: no tests remain for %s", all_disease_ids)
        return {
            "should_finalize":     True,
            "judge_details":       {"reason": "No tests available"},
            "finalization_reason": "no_tests",
            "confidence_warning":  _top_confidence_warning(state),
        }

    # --- Build selector context (N1 + N4 fixes) ---
    test_lr_map = {t["id"]: neo4j.get_test_edges(t["id"]) for t in available_tests}
    symptom     = state["symptom_match"].get("clinical_term") or None
    qa_history  = (
        [{"question": q["question"]} for q in state["questions_asked"]]
        or None
    )

    # --- Select question ---
    question  = selector.select_question(
        state["differential"],
        available_tests,
        test_lr_map=test_lr_map,
        symptom=symptom,
        qa_history=qa_history,
    )
    valid_ids = {t["id"] for t in available_tests}
    if question.get("test_id") not in valid_ids:
        fallback = available_tests[0]
        logger.warning(
            "question: selector returned out-of-scope test_id %r — falling back to %s",
            question.get("test_id"), fallback["id"],
        )
        question = {
            "test_id":   fallback["id"],
            "question":  f"What is the result of {fallback['name']}?",
            "reasoning": "fallback: selector returned a test outside the available list",
        }

    logger.info("question: selected test=%s", question.get("test_id"))
    return {
        "should_finalize":  False,
        "pending_question": question,
    }


def answer_node(state: DiagnosticState, config: RunnableConfig) -> Dict:
    """
    Interrupt → evaluate → Bayesian update.

    Reads pending_question from state (set by question_node), surfaces it to
    the caller via interrupt(), then on resume evaluates the answer and applies
    the Bayesian update. Always routes back to question_node.
    """
    cfg       = config["configurable"]
    neo4j     = cfg["neo4j"]
    evaluator: EvidenceEvaluatorAgent = cfg["evaluator"]

    question = state["pending_question"]

    # Interrupt: surface question to caller, wait for answer.
    # First execution: pauses here. On resume: returns the submitted answer.
    answer: str = interrupt(question)

    # Snapshot top prob BEFORE this update (for next turn's stability gate).
    pre_update_top = (
        state["differential"][0]["probability"]
        if state.get("differential") else None
    )

    edges = neo4j.get_test_edges(question["test_id"])

    # Wire test_threshold if the neo4j client provides one (C2 readiness).
    get_threshold  = getattr(neo4j, "get_test_threshold", None)
    test_threshold = get_threshold(question["test_id"]) if get_threshold else None

    evidence = evaluator.evaluate(
        question["question"],
        _sanitize(str(answer)),
        edges,
        test_threshold=test_threshold,
    )

    engine  = _restore_engine(state)
    updated = engine.update(evidence)

    logger.info(
        "answer: n_evidence=%d top=%s(%.2f)",
        engine.get_evidence_count(),
        updated[0]["name"] if updated else "none",
        updated[0]["probability"] if updated else 0,
    )

    return {
        "differential":      updated,
        "evidence_history":  engine.evidence_history,
        "questions_asked":   state["questions_asked"] + [question],
        "pending_question":  None,
        "previous_top_prob": pre_update_top,
    }


def finalize_node(state: DiagnosticState) -> Dict:
    """Compile the final diagnosis report from completed state."""
    if not state.get("differential"):
        return {"final_diagnosis": {"error": "No differential available"}}

    top    = state["differential"][0]
    reason = state.get("finalization_reason") or "confidence_gate"
    warn   = state.get("confidence_warning", False)

    report = {
        "primary_diagnosis":   top["name"],
        "confidence":          top["probability"],
        "confidence_warning":  warn,
        "finalization_reason": reason,
        "differential":        state["differential"][:3],
        "evidence_chain": [
            {"question": q["question"], "reasoning": q.get("reasoning", "")}
            for q in state.get("questions_asked", [])
        ],
        "total_questions": len(state.get("questions_asked", [])),
    }

    logger.info(
        "finalize: %s confidence=%.2f reason=%s warning=%s",
        top["name"], top["probability"], reason, warn,
    )
    return {"final_diagnosis": report}


# ------------------------------------------------------------------ #
#  Graph compilation                                                   #
# ------------------------------------------------------------------ #

def _build_graph(checkpointer):
    builder = StateGraph(DiagnosticState)

    builder.add_node("seed",     seed_node)
    builder.add_node("question", question_node)
    builder.add_node("answer",   answer_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START,      "seed")
    builder.add_edge("seed",     "question")
    builder.add_conditional_edges(
        "question",
        lambda state: "finalize" if state.get("should_finalize") else "answer",
    )
    builder.add_edge("answer",   "question")
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

        Runs seed → question_node (select Q1) → answer_node interrupt.
        Returns (session_id, result) where result['next_question'] is the
        first question ready for the user.
        """
        user_input = _sanitize(user_input)
        session_id = str(uuid.uuid4())
        config     = self._config(session_id)

        initial: DiagnosticState = {
            "user_input":          user_input,
            "symptom_match":       {},
            "differential":        [],
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

        snapshot = self._graph.invoke(initial, config)

        return session_id, {
            "session_id":           session_id,
            "symptom_match":        snapshot.get("symptom_match", {}),
            "initial_differential": snapshot.get("differential", []),
            "next_question":        self._pending_question(config),
        }

    def submit_answer(self, session_id: str, answer: str) -> Dict:
        """
        Resume the graph with the user's answer to the current question.

        Raises:
            SessionNotFoundError: if session_id does not exist in the
                                  checkpointer (C3 fix — was returning 503).

        Returns updated differential and the next question (None if done).
        """
        config = self._config(session_id)

        # C3 fix: detect dead/unknown sessions before attempting resume.
        state = self._graph.get_state(config)
        if not state or not state.values:
            raise SessionNotFoundError(f"No active session: {session_id}")

        snapshot = self._graph.invoke(Command(resume=_sanitize(answer)), config)

        next_q          = self._pending_question(config)
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
