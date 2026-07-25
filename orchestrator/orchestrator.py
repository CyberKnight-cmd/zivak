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

"""
Orchestrator — LangGraph diagnostic workflow.
[... unchanged module docstring ...]
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
from agents.symptom_extractor import SymptomExtractorAgent
from engines.confidence_judge import ConfidenceJudge
from engines.differential import DifferentialEngine
from engines.information_gain import rank_by_eig

logger = logging.getLogger(__name__)

MAX_QUESTIONS = 10
MAX_INPUT_LEN = 1000

# --- Phase 1: dynamic differential expansion controls ---
MAX_NEW_CANDIDATES_PER_EXPANSION = 8   # cap per single expansion event
MAX_EXPANSIONS_PER_SESSION       = 3   # safety cap on expansion events per session
MAX_DIFFERENTIAL_SIZE            = 40  # trim lowest-probability overflow after merge

NEGATION_RE = re.compile(
    r"\b(no|not|never|denies|denied|without|n't|nothing|none)\b",
    re.IGNORECASE,
)


def _sanitize(text: str) -> str:
    text = text.strip()[:MAX_INPUT_LEN]
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)


def _detect_negation(text: str) -> bool:
    """
    Conservative, deterministic negation check for the whole answer.

    Phase-1 approximation: if the answer contains a negation cue anywhere,
    treat any newly-matched symptom as unconfirmed and skip expansion
    rather than risk a false-positive candidate injection. This will
    under-trigger on compound answers like "no weakness, but I do have a
    rash" — refining per-clause polarity is Phase-3 scope (conversational
    extraction), noted as a known limitation, not blocking this fix.
    """
    return bool(NEGATION_RE.search(text))


def _infer_polarity(edges: List[Dict], evidence: Dict) -> Optional[str]:
    """
    Deterministically infer whether a test result was POSITIVE or NEGATIVE
    from the evaluator's already-computed output — no LLM change needed.

    A positive result copies RULES_IN/RULES_OUT edges through unchanged;
    a negative result inverts and swaps them. So: if any edge's disease
    appears in evidence['rules_in'] with an edge of type RULES_IN (or in
    'rules_out' with type RULES_OUT), the result was POSITIVE. If it
    appears in the opposite list, the result was NEGATIVE.

    Returns None if no edge gives an unambiguous signal (e.g. edges is
    empty) — replay for that turn then contributes nothing (LR=1), which
    is the correct neutral fallback anyway.
    """
    rules_in  = {r["disease"].lower().strip() for r in evidence.get("rules_in", [])}
    rules_out = {r["disease"].lower().strip() for r in evidence.get("rules_out", [])}

    for e in edges:
        key = e["disease"].lower().strip()
        if e["relationship"] == "RULES_IN":
            if key in rules_in:
                return "positive"
            if key in rules_out:
                return "negative"
        else:  # RULES_OUT
            if key in rules_out:
                return "positive"
            if key in rules_in:
                return "negative"
    return None


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
    previous_top_prob:   Optional[float]
    pending_question:    Optional[Dict]
    finalization_reason: Optional[str]
    confidence_warning:  bool
    error_message:       Optional[str]
    seen_symptom_ids:    List[str]   # every HP id already folded into the differential
    expansions_used:     int         # count of dynamic-expansion events this session


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _restore_engine(state: DiagnosticState) -> DifferentialEngine:
    engine = DifferentialEngine()
    engine.differential     = state.get("differential", [])
    engine.evidence_history = state.get("evidence_history", [])
    return engine


def _top_confidence_warning(state: DiagnosticState) -> bool:
    if not state.get("differential"):
        return True
    return state["differential"][0]["probability"] < ConfidenceJudge().min_top_confidence


def _active_ids(differential: List[Dict], threshold: float = 0.03, min_count: int = 8) -> List[str]:
    above = [d for d in differential if d["probability"] >= threshold]
    active = above if len(above) >= min_count else differential[:min_count]
    return [d.get("disease_id", d["name"]) for d in active]


# ------------------------------------------------------------------ #
#  Nodes                                                               #
# ------------------------------------------------------------------ #

def seed_node(state: DiagnosticState, config: RunnableConfig) -> Dict:
    cfg    = config["configurable"]
    qdrant = cfg["qdrant"]
    neo4j  = cfg["neo4j"]
    extractor = cfg["extractor"]

    clinical_terms = extractor.extract_symptoms(state["user_input"])
    # If extraction fails entirely, fallback to the raw user input
    search_terms = clinical_terms if clinical_terms else state["user_input"]

    symptom_match = qdrant.search(search_terms)

    logger.info(
        "seed: qdrant matched=%s terms=%r -> clinical_term=%r score=%.3f symptom_ids=%s",
        symptom_match.get("matched"),
        search_terms,
        symptom_match.get("clinical_term"),
        symptom_match.get("score", 0.0),
        symptom_match.get("symptom_ids", []),
    )

    if not symptom_match["matched"]:
        return {
            "symptom_match":       symptom_match,
            "final_diagnosis":     None,
            "should_finalize":     True,
            "confidence_warning":  True,
            "finalization_reason": "no_symptom_match",
            "error_message": (
                "I couldn't identify any medical symptoms in your description. "
                "Please describe your symptoms more specifically — for example, "
                "'chest pain', 'shortness of breath', or 'severe headache'."
            ),
        }

    if symptom_match.get("ambiguous"):
        return {
            "symptom_match":       symptom_match,
            "final_diagnosis":     None,
            "should_finalize":     True,
            "confidence_warning":  True,
            "finalization_reason": "ambiguous_symptom",
            "error_message": (
                "I want to make sure I understand your main concern correctly. "
                "Could you describe it in a few words? For example: "
                "'chest pain', 'shortness of breath', 'dizziness'."
            ),
        }

    symptom_ids = symptom_match.get("symptom_ids") or [symptom_match["symptom_id"]]
    diseases    = neo4j.get_initial_differential(symptom_ids, term=symptom_match.get("clinical_term"))

    engine       = DifferentialEngine()
    differential = engine.initialize(diseases)

    logger.info(
        "seed: symptom=%r n_diseases=%d hp_ids=%s",
        symptom_match.get("clinical_term"), len(differential), symptom_ids,
    )

    if not differential:
        logger.warning(
            "seed: Qdrant matched %r (hp_ids=%s) but Neo4j returned 0 diseases — "
            "run scripts/7_verify_neo4j.py to check PRESENTS_WITH edge counts",
            symptom_match.get("clinical_term"), symptom_ids,
        )
        return {
            "symptom_match":       symptom_match,
            "differential":        [],
            "evidence_history":    [],
            "should_finalize":     True,
            "confidence_warning":  True,
            "finalization_reason": "no_diseases_found",
            "error_message": (
                "Your symptoms were recognised but no diagnostic diseases were found "
                "in the knowledge base. Please describe your symptoms differently."
            ),
            "seen_symptom_ids": list(symptom_ids),
        }

    return {
        "symptom_match":    symptom_match,
        "differential":     differential,
        "evidence_history": engine.evidence_history,
        "seen_symptom_ids": list(symptom_ids),
        "expansions_used":  0,
    }


def question_node(state: DiagnosticState, config: RunnableConfig) -> Dict:
    """UNCHANGED from your current file."""
    cfg      = config["configurable"]
    neo4j    = cfg["neo4j"]
    selector: QuestionSelectorAgent = cfg["selector"]

    if not state.get("differential"):
        return {
            "should_finalize":     True,
            "judge_details":       {"reason": "No diseases in differential after seeding"},
            "finalization_reason": state.get("finalization_reason") or "no_diseases_found",
            "confidence_warning":  True,
        }

    engine = _restore_engine(state)
    judge  = ConfidenceJudge()

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

    if len(state["questions_asked"]) >= MAX_QUESTIONS:
        logger.warning("question: MAX_QUESTIONS cap reached")
        return {
            "should_finalize":     True,
            "judge_details":       {"reason": "Max questions reached"},
            "finalization_reason": "max_questions",
            "confidence_warning":  _top_confidence_warning(state),
        }

    active_ids = _active_ids(state["differential"])

    asked_ids       = {q["test_id"] for q in state["questions_asked"]}
    available_tests = [
        t for t in neo4j.get_available_tests(active_ids)
        if t["id"] not in asked_ids
    ]
    if not available_tests:
        logger.warning("question: no tests remain for active diseases %s", active_ids)
        return {
            "should_finalize":     True,
            "judge_details":       {"reason": "No tests available"},
            "finalization_reason": "no_tests",
            "confidence_warning":  _top_confidence_warning(state),
        }

    test_lr_map = {
        t["id"]: neo4j.get_test_edges(t["id"], active_ids)
        for t in available_tests
    }
    top_ids   = rank_by_eig(state["differential"], test_lr_map, top_n=5)
    top_tests = [t for t in available_tests if t["id"] in top_ids]
    top_lr    = {tid: test_lr_map[tid] for tid in top_ids if tid in test_lr_map}

    if not top_tests:
        top_tests = available_tests[:1]
        top_lr    = {top_tests[0]["id"]: test_lr_map.get(top_tests[0]["id"], [])}

    logger.info(
        "question: %d candidates → top-%d by EIG: %s",
        len(available_tests), len(top_tests), [t["id"] for t in top_tests],
    )

    symptom    = state["symptom_match"].get("clinical_term") or None
    qa_history = (
        [{"question": q["question"]} for q in state["questions_asked"]]
        or None
    )

    question  = selector.select_question(
        state["differential"],
        top_tests,
        test_lr_map=top_lr,
        symptom=symptom,
        qa_history=qa_history,
    )
    valid_ids = {t["id"] for t in top_tests}
    if question.get("test_id") not in valid_ids:
        fallback = top_tests[0]
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
    Interrupt → evaluate → Bayesian update → [NEW] dynamic differential
    expansion from volunteered symptoms.
    """
    cfg       = config["configurable"]
    qdrant    = cfg["qdrant"]
    neo4j     = cfg["neo4j"]
    evaluator: EvidenceEvaluatorAgent = cfg["evaluator"]
    extractor: SymptomExtractorAgent  = cfg["extractor"]

    question = state["pending_question"]
    answer: str = interrupt(question)
    clean_answer = _sanitize(str(answer))

    pre_update_top = (
        state["differential"][0]["probability"]
        if state.get("differential") else None
    )

    active_ids = _active_ids(state["differential"]) or None
    edges = neo4j.get_test_edges(question["test_id"], active_ids)

    get_threshold  = getattr(neo4j, "get_test_threshold", None)
    test_threshold = get_threshold(question["test_id"]) if get_threshold else None

    evidence = evaluator.evaluate(
        question["question"],
        clean_answer,
        edges,
        test_threshold=test_threshold,
    )

    # Deterministic, no LLM: infer polarity from the evaluator's own output
    # so this turn's evidence can be replayed later against a new disease.
    polarity = _infer_polarity(edges, evidence)

    engine  = _restore_engine(state)
    updated = engine.update(evidence, test_id=question["test_id"], polarity=polarity)

    # ---------------------------------------------------------------- #
    #  NEW: dynamic expansion — check whether the answer volunteered a
    #  genuinely new symptom that should introduce new disease candidates.
    # ---------------------------------------------------------------- #
    seen_symptom_ids = list(state.get("seen_symptom_ids", []))
    expansions_used  = state.get("expansions_used", 0)

    if expansions_used < MAX_EXPANSIONS_PER_SESSION and not _detect_negation(clean_answer):
        extracted_terms = extractor.extract_symptoms(clean_answer)
        search_terms = extracted_terms if extracted_terms else clean_answer
        match = qdrant.search(search_terms)
        
        new_hp_ids = [
            hp for hp in match.get("symptom_ids", [])
            if match.get("matched") and hp not in seen_symptom_ids
        ]

        for hp_id in new_hp_ids:
            if expansions_used >= MAX_EXPANSIONS_PER_SESSION:
                break

            raw_candidates = neo4j.get_initial_differential([hp_id])
            existing_keys  = {d.get("disease_id", d["name"]) for d in updated}
            new_candidates = [
                d for d in raw_candidates
                if d.get("disease_id", d["name"]) not in existing_keys
            ]
            seen_symptom_ids.append(hp_id)

            if not new_candidates:
                continue  # nothing genuinely new for this symptom — still mark it seen

            new_ids = [d.get("disease_id", d["name"]) for d in new_candidates][:MAX_NEW_CANDIDATES_PER_EXPANSION]

            # Fetch replay edges: for every historical evidence event with a
            # known test_id, get the LR these new candidates would have had.
            replay_edges: Dict[str, List[Dict]] = {}
            for entry in engine.evidence_history:
                t_id = entry.get("test_id")
                if t_id and t_id not in replay_edges:
                    replay_edges[t_id] = neo4j.get_test_edges(t_id, new_ids)

            updated = engine.expand(
                new_candidates,
                replay_edges,
                max_new=MAX_NEW_CANDIDATES_PER_EXPANSION,
            )
            expansions_used += 1

            # Apply the volunteered symptom itself as an ordinary positive
            # finding, scoped to the full (now-expanded) differential.
            full_ids  = [d.get("disease_id", d["name"]) for d in updated]
            new_edges = neo4j.get_test_edges(hp_id, full_ids)
            if new_edges:
                symptom_evidence = {
                    "rules_in":  [{"disease": e["disease"], "likelihood_ratio": e["lr"]}
                                  for e in new_edges if e["relationship"] == "RULES_IN"],
                    "rules_out": [{"disease": e["disease"], "likelihood_ratio": e["lr"]}
                                  for e in new_edges if e["relationship"] == "RULES_OUT"],
                }
                updated = engine.update(symptom_evidence, test_id=hp_id, polarity="positive")

            # Bound total differential size — trim lowest-probability overflow.
            if len(updated) > MAX_DIFFERENTIAL_SIZE:
                updated = sorted(updated, key=lambda d: d["probability"], reverse=True)
                updated = updated[:MAX_DIFFERENTIAL_SIZE]
                engine.differential = updated

            logger.info(
                "answer: expansion #%d via %s introduced %d candidate(s)",
                expansions_used, hp_id, len(new_candidates[:MAX_NEW_CANDIDATES_PER_EXPANSION]),
            )

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
        "seen_symptom_ids":  seen_symptom_ids,
        "expansions_used":   expansions_used,
    }


def finalize_node(state: DiagnosticState) -> Dict:
    """UNCHANGED from your current file."""
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
#  Graph compilation — UNCHANGED                                       #
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


class SessionNotFoundError(Exception):
    pass


class DiagnosticOrchestrator:
    """UNCHANGED except start_session's initial state dict below."""

    def __init__(self, qdrant_client, neo4j_client, checkpointer=None):
        self._qdrant    = qdrant_client
        self._neo4j     = neo4j_client
        self._selector  = QuestionSelectorAgent()
        self._evaluator = EvidenceEvaluatorAgent()
        self._extractor = SymptomExtractorAgent()
        self._graph     = _build_graph(checkpointer or MemorySaver())

    def start_session(self, user_input: str) -> Tuple[str, Dict]:
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
            "error_message":       None,
            "seen_symptom_ids":    [],
            "expansions_used":     0,
        }

        snapshot = self._graph.invoke(initial, config)

        return session_id, {
            "session_id":           session_id,
            "symptom_match":        snapshot.get("symptom_match", {}),
            "initial_differential": snapshot.get("differential", []),
            "next_question":        self._pending_question(config),
            "finalization_reason":  snapshot.get("finalization_reason"),
            "error_message":        snapshot.get("error_message"),
            "final_diagnosis":      snapshot.get("final_diagnosis"),
        }

    def submit_answer(self, session_id: str, answer: str) -> Dict:
        config = self._config(session_id)
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
        snapshot = self._graph.get_state(self._config(session_id))
        result   = snapshot.values.get("final_diagnosis")
        return result or {"error": "Session not finalized yet"}

    def _config(self, session_id: str) -> Dict:
        return {
            "configurable": {
                "thread_id": session_id,
                "qdrant":    self._qdrant,
                "neo4j":     self._neo4j,
                "selector":  self._selector,
                "evaluator": self._evaluator,
                "extractor": self._extractor,
            }
        }

    def _pending_question(self, config: Dict) -> Optional[Dict]:
        state = self._graph.get_state(config)
        if state.tasks and state.tasks[0].interrupts:
            return state.tasks[0].interrupts[0].value
        return None