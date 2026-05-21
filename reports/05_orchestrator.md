# DiagnosticOrchestrator — Deep Dive Report

**File:** `orchestrator/orchestrator.py`
**Type:** LangGraph state machine + public API wrapper
**Tests:** `orchestrator/tests/test_orchestrator.py`, `tests/test_api.py`

---

## 1. Purpose

Ties all components together into a coherent, stateful diagnostic session.
Manages the LangGraph graph, session lifecycle, and exposes a clean Python API
that the FastAPI layer calls.

---

## 2. LangGraph Graph Topology

```
START
  │
  ▼
seed_node           — symptom search + initial differential (no LLM)
  │
  ▼
qa_node  ◄──────────────────────────────────┐
  │                                          │
  ├── should_finalize = False ───────────────┘  (loop: ask another question)
  │
  └── should_finalize = True
           │
           ▼
       finalize_node                         — compile final report (no LLM)
           │
           ▼
          END
```

The `qa_node` is an **interrupt node** — it pauses execution and surfaces a question
to the caller. The caller resumes it with `Command(resume=answer)`.

---

## 3. DiagnosticState — The Full State Schema

```python
class DiagnosticState(TypedDict):
    user_input:        str           # original patient complaint (sanitized)
    symptom_match:     Dict          # {clinical_term, symptom_id, score}
    differential:      List[Dict]    # [{name, probability, log_prob, ...}] sorted desc
    evidence_history:  List[Dict]    # [{rules_in, rules_out}, ...] one per turn
    questions_asked:   List[Dict]    # [{test_id, question, reasoning}, ...] answered
    should_finalize:   bool          # gate flag for conditional edge
    judge_details:     Dict          # ConfidenceJudge output for debugging
    final_diagnosis:   Optional[Dict]# populated by finalize_node only
    previous_top_prob: Optional[float]# top prob snapshot before last answer
```

### How state flows between turns

```
start_session():
  initial_state = {all fields empty/None}
  graph.invoke(initial_state, config)
    → seed_node runs: fills symptom_match, differential, evidence_history=[]
    → qa_node runs:   selects first question, calls interrupt()
    → graph pauses, returns snapshot
  Returns: (session_id, {symptom_match, initial_differential, next_question})

submit_answer(session_id, answer):
  graph.invoke(Command(resume=answer), config)
    → qa_node re-executes from top (see Double LLM Call bug below)
    → evaluator.evaluate() maps answer to evidence
    → engine.update() applies Bayesian update
    → ConfidenceJudge checks finalization
    → if not done: interrupt() with next question
    → if done: sets should_finalize=True, routes to finalize_node
  Returns: {updated_differential, should_continue, next_question, final_diagnosis}
```

---

## 4. seed_node — Symptom Mapping & Initial Differential

```python
def seed_node(state, config):
    qdrant = config["configurable"]["qdrant"]
    neo4j  = config["configurable"]["neo4j"]

    symptom_match = qdrant.search_symptom(state["user_input"])
    diseases      = neo4j.get_initial_differential(symptom_match["symptom_id"])

    engine       = DifferentialEngine()
    differential = engine.initialize(diseases)

    return {
        "symptom_match":    symptom_match,
        "differential":     differential,
        "evidence_history": engine.evidence_history,   # empty list []
    }
```

**What it does NOT return:** `user_input`, `questions_asked`, `should_finalize`,
`judge_details`, `final_diagnosis`, `previous_top_prob`. LangGraph merges the returned
dict into the existing state — unmentioned keys keep their initial values.

**No LLM involved.** This node is pure semantic search + DB lookup + math.

---

## 5. qa_node — The Core Loop Node

This is the most complex node and contains the critical double-execution bug.

### Full execution path

```python
def qa_node(state, config):
    # --- Extract dependencies from config ---
    neo4j     = config["configurable"]["neo4j"]
    selector  = config["configurable"]["selector"]
    evaluator = config["configurable"]["evaluator"]

    # --- Reconstruct engine from state ---
    engine = _restore_engine(state)   # DifferentialEngine with current differential
    judge  = ConfidenceJudge()

    # ═══ TERMINATION CHECK (runs on EVERY execution, including re-execution on resume) ═══
    should_stop, judge_details = judge.should_finalize(
        state["differential"],
        engine.get_evidence_count(),     # len(state["evidence_history"])
        state.get("previous_top_prob"),
    )
    if should_stop:
        return {"should_finalize": True, "judge_details": judge_details}

    # ═══ MAX QUESTIONS CAP ═══
    if len(state["questions_asked"]) >= MAX_QUESTIONS:   # MAX_QUESTIONS = 10
        return {"should_finalize": True, "judge_details": {"reason": "Max questions reached"}}

    # ═══ BUILD AVAILABLE TESTS ═══
    top_diseases    = [d["name"] for d in state["differential"][:5]]
    asked_ids       = {q["test_id"] for q in state["questions_asked"]}
    available_tests = [
        t for t in neo4j.get_available_tests(top_diseases)
        if t["id"] not in asked_ids
    ]
    if not available_tests:
        return {"should_finalize": True, "judge_details": {"reason": "No tests available"}}

    # ═══ SELECT QUESTION — LLM CALL (runs twice per turn due to interrupt re-execution) ═══
    question  = selector.select_question(state["differential"], available_tests)
    valid_ids = {t["id"] for t in available_tests}
    if question.get("test_id") not in valid_ids:
        fallback  = available_tests[0]
        question  = {
            "test_id":  fallback["id"],
            "question": f"What is the result of {fallback['name']}?",
            "reasoning": "fallback",
        }

    # ═══ INTERRUPT — PAUSES HERE, SURFACES QUESTION TO CALLER ═══
    answer: str = interrupt(question)

    # ═══ PROCESS ANSWER (only runs when resuming) ═══
    edges    = neo4j.get_test_edges(question["test_id"])
    evidence = evaluator.evaluate(question["question"], _sanitize(str(answer)), edges)
    updated  = engine.update(evidence)

    # Snapshot top prob BEFORE this update (for next turn's stability gate)
    pre_update_top = state["differential"][0]["probability"] if state["differential"] else None

    return {
        "differential":      updated,
        "evidence_history":  engine.evidence_history,
        "questions_asked":   state["questions_asked"] + [question],
        "should_finalize":   False,
        "previous_top_prob": pre_update_top,
    }
```

### The interrupt() mechanism

```
graph.invoke(initial_state, config):
    seed_node() runs completely
    qa_node() runs:
        - termination check
        - question selection (LLM call #1)
        - interrupt(question) → GRAPH PAUSES, returns snapshot to caller
    Caller receives snapshot where tasks[0].interrupts[0].value = question

graph.invoke(Command(resume=answer), config):
    qa_node() RUNS AGAIN FROM THE TOP:
        - termination check (same result, state unchanged)
        - question selection (LLM call #2 — WASTED, possibly different!)
        - interrupt(question) → returns 'answer' immediately (the resume value)
        - evaluator.evaluate() runs with question from LLM call #2
        - engine.update() applies evidence
        - returns updated state
    If should_finalize=False:
        qa_node() runs again:
            - question selection (LLM call #3 for the NEW question)
            - interrupt(next_question) → GRAPH PAUSES again
```

---

## 6. finalize_node — Report Compilation

```python
def finalize_node(state):
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
    return {"final_diagnosis": report}
```

**What's missing from the blueprint:** The `ExplainabilityAgent` should run here,
generating a grounded reasoning chain ("Ruled out X because [evidence]. Confirmed Y
because [evidence]."). Currently, `evidence_chain` just echoes back the questions
asked and the selector's brief `reasoning` string — not the evaluator's clinical
interpretation.

---

## 7. Session Management

### _config() — Dependency injection via LangGraph configurable

```python
def _config(self, session_id):
    return {
        "configurable": {
            "thread_id": session_id,   # LangGraph uses this to key the checkpoint
            "qdrant":    self._qdrant,
            "neo4j":     self._neo4j,
            "selector":  self._selector,
            "evaluator": self._evaluator,
        }
    }
```

Agents are instantiated ONCE in `__init__` and shared across all sessions. They are
stateless (each call is independent), so this is safe.

### _pending_question() — Extracting the interrupt value

```python
def _pending_question(self, config):
    state = self._graph.get_state(config)
    if state.tasks and state.tasks[0].interrupts:
        return state.tasks[0].interrupts[0].value
    return None
```

This reads the pending interrupt from the graph checkpoint. If the graph is not
currently interrupted (either just started or just finalized), returns None.

### should_continue logic in submit_answer

```python
should_continue = next_q is not None and final_diagnosis is None
```

This is correct in the normal flow. Edge case: if somehow `final_diagnosis` is set
AND `next_q` is not None, `should_continue=False` correctly. If both are None
(graph errored silently), `should_continue=False` — the caller would think the session
is done but `final_diagnosis` is also None. The API layer should handle this.

---

## 8. CRITICAL BUG: SessionNotFoundError Never Raised

### The bug

The FastAPI layer has a handler for `SessionNotFoundError`:
```python
except SessionNotFoundError:
    raise HTTPException(status_code=404, detail="Session not found or expired")
```

But the orchestrator's `submit_answer()` never raises `SessionNotFoundError`.
If `graph.invoke(Command(resume=answer), config)` is called with a non-existent
`session_id`, LangGraph's MemorySaver raises a different internal error.

### Consequence
Submitting an answer to a non-existent session returns **503** ("Diagnostic engine
unavailable") instead of **404** ("Session not found"). The client can't distinguish
between a dead session and a server error.

### Fix
Wrap the graph invoke in try/except and detect "no checkpoint for thread_id":
```python
def submit_answer(self, session_id, answer):
    config = self._config(session_id)
    state  = self._graph.get_state(config)
    if state is None or not state.values:
        raise SessionNotFoundError(f"No session: {session_id}")
    # ... rest of submit
```

---

## 9. _sanitize() — Input Safety

```python
MAX_INPUT_LEN = 1000

def _sanitize(text):
    text = text.strip()[:MAX_INPUT_LEN]           # cap length
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
```

Strips: null bytes, backspace, vertical tab, form feed, other control chars.
Keeps: newlines (`\n` = `\x0a`), tabs (`\t` = `\x09`), all printable ASCII, Unicode.

Applied to:
- Patient's initial complaint (in `start_session`)
- Patient's answers to questions (in `submit_answer`, before passing to evaluator)
- Also inside the prompt as `_sanitize(str(answer))` — double-sanitised for the LLM

---

## 10. MemorySaver vs. Redis Checkpointer

Current default:
```python
_graph = _build_graph(checkpointer or MemorySaver())
```

`MemorySaver` stores all session state in memory — per-process, lost on restart.
For production, swap to `langgraph-checkpoint-redis`:
```python
from langgraph.checkpoint.redis import RedisSaver
checkpointer = RedisSaver.from_conn_string(os.getenv("REDIS_URL"))
```

The `thread_id = session_id` (UUID) is the key. All state including `log_prob` floats
must serialize cleanly to JSON/Redis. The current state schema is fully JSON-serialisable.

---

## 11. Known Issues Summary

| Issue                            | Severity | File/Line              | Impact                                      |
|----------------------------------|----------|------------------------|---------------------------------------------|
| Double LLM call per turn         | CRITICAL | qa_node, interrupt()   | 2× cost, potential wrong question/evidence  |
| SessionNotFoundError never raised| HIGH     | submit_answer()        | 503 instead of 404 for dead sessions        |
| No ExplainabilityAgent           | HIGH     | finalize_node()        | Reasoning chain is stub (just Q list)       |
| finalize with low confidence     | MEDIUM   | MAX_QUESTIONS cap      | Can output 55% confidence "diagnosis"       |
| pre_update_top after update call | LOW      | qa_node line ~177      | Snapshot is taken from incoming state correctly — actually fine |

---

## 12. Test Coverage

| Test                 | What it validates                                     | Status          |
|----------------------|-------------------------------------------------------|-----------------|
| test_full_workflow   | Start session → answer → get question                 | Requires Ollama |
| test_health          | GET /health returns 200                               | PASS (mock)     |
| test_start_session   | POST /sessions returns session_id + question          | PASS (mock)     |
| test_submit_answer   | POST /sessions/{id}/answer returns updated diff       | PASS (mock)     |
| test_get_diagnosis   | GET /sessions/{id}/diagnosis returns final dx         | PASS (mock)     |
| test_404_session     | SessionNotFoundError → 404                            | PASS (mock)     |
| test_503_engine      | RuntimeError → 503                                    | PASS (mock)     |

**Missing tests:**
- End-to-end session reaching finalization (real Ollama, real math)
- Session not found returns 503 (demonstrates the bug)
- MAX_QUESTIONS cap triggers finalization
- No available tests triggers finalization
- Double-question selection consistency check
