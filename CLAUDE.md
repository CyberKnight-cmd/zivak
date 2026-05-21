# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests (no LLM required — Ollama tests are @pytest.mark.skip)
pytest

# Run a single test file
pytest engines/tests/test_differential.py -v

# Run a single test by name
pytest -k "test_stability_gate_fires" -v

# Run the full mock scenario suite (requires LLM — uses Groq or Ollama)
python run_scenarios.py
python run_scenarios.py --filter COPD Asthma BPPV   # run specific scenarios
python run_scenarios.py --verbose --delay 90 --turn-delay 5

# Start the API server
uvicorn main:app --reload
```

Environment variables live in `.env`. Copy `.env.example` if it exists. Minimum required:
- `GROQ_API_KEY` — used by both LLM agents unless `USE_LOCAL_LLM=true`
- `API_KEY` — Bearer token for all `/api/v1/*` endpoints
- `USE_LOCAL_LLM=true` + `LOCAL_MODEL` + `LOCAL_LLM_URL` — to use Ollama instead of Groq

## Architecture

### LangGraph graph (orchestrator/orchestrator.py)

```
START → seed_node → question_node ◄──────────────┐
                         │                        │
                         ├─ not done → answer_node┘
                         └─ done    → finalize_node → END
```

**Why two nodes instead of one `qa_node`:** LangGraph re-runs the interrupted node from scratch on resume. A single combined node would call the LLM selector twice per turn (once before `interrupt()`, once on resume) and risk mismatched question/evidence pairs at `temperature=1`. The split ensures the selector fires exactly once per question turn.

- **seed_node** — Qdrant semantic search → Neo4j differential seed → `DifferentialEngine.initialize()`. No LLM.
- **question_node** — Confidence/max-questions/no-tests termination checks → builds full test pool from ALL differential diseases (not just top-5) → calls `QuestionSelectorAgent` once → saves `pending_question` to state → returns. No `interrupt()` here.
- **answer_node** — Reads `pending_question` from state → `interrupt(question)` surfaces it to the caller → on resume evaluates answer via `EvidenceEvaluatorAgent` → `DifferentialEngine.update()`. Always routes back to `question_node`.
- **finalize_node** — Compiles report. Includes `finalization_reason` ("confidence_gate" | "no_tests" | "max_questions") and `confidence_warning` bool for callers to distinguish forced terminations from genuine high-confidence diagnoses.

### State (`DiagnosticState` TypedDict)

The LangGraph checkpointer serialises the entire `DiagnosticState`. `DifferentialEngine` is **not** stored in state — it is reconstructed each turn from `differential` (list of dicts with `log_prob`) and `evidence_history` via `_restore_engine()`. Float precision across serialisation formats is a known risk for future Redis checkpointers.

`pending_question` is the `question_node → answer_node` handoff: set by `question_node`, read by `answer_node`, cleared to `None` on `answer_node` return.

### Two LLM agents

Both agents share the same retry pattern (3 attempts, exponential backoff 1s/2s) and `_extract_json()` helper that handles clean JSON, markdown fences, and reasoning-prefixed output.

**QuestionSelectorAgent** (`temperature=1`, `reasoning_effort="medium"`)
- Receives: differential probabilities (top 5), available tests, optional `test_lr_map` (numeric LRs from Neo4j for each test), optional `symptom` string, optional `qa_history` list.
- When `test_lr_map` is provided the prompt shows `"ILD LR≈20.0, COPD LR≈3.5"` per test instead of `"relevant for COPD, ILD"` — this is the N1 fix so the LLM can rank high-information tests.

**EvidenceEvaluatorAgent** (`temperature=0`, `reasoning_effort="medium"`)
- The LLM only decides **polarity** (positive/negative). Both evidence branches are pre-computed in Python from the Neo4j edges; the LLM copies the matching branch verbatim.
- Optional `test_threshold` parameter (e.g. `"FEV1/FVC < 0.70 = POSITIVE"`) is injected into Step 1 of the prompt to prevent polarity confusion on quantitative tests.
- `_validate_output()` whitelists disease names against the supplied edges and clamps LRs to `[0.001, 100]` before any value reaches the Bayesian engine.

### Two pure-math engines

**DifferentialEngine** (`engines/differential.py`)
- Works entirely in log-probability space to prevent underflow.
- `initialize()` — `raw_score = specificity × prevalence`, normalized, floored at `RARE_DISEASE_FLOOR = 0.02`.
- `update()` — builds a `log_lr_map` per disease by summing `log(LR)` for all entries in `rules_in + rules_out`, then adds to each disease's `log_prob`, then log-sum-exp normalises. Warns if a `rules_in` entry has `LR < 1` or a `rules_out` entry has `LR > 1` (authoring error).

**ConfidenceJudge** (`engines/confidence_judge.py`)
- All three criteria must hold: `evidence_count >= min_evidence` (default 4), `top >= 0.75`, `runner_up < 0.40`.
- Stability gate: if the current leader dropped >10 pp on the last answer, keep asking regardless of other criteria.
- Returns `confidence_warning: bool` on every code path so forced terminations (`no_tests`, `max_questions`) can be flagged to API consumers.

### Mock clients (`orchestrator/mock_clients.py`)

`MockQdrantClient` does real semantic search using `sentence-transformers/all-MiniLM-L6-v2` (CPU, ~80 MB). The production interface is identical — swap to `text-embedding-3-small` on a real Qdrant collection.

`MockNeo4jClient` provides three methods that the real Neo4j client must also implement:
- `get_initial_differential(symptom_id)` → `List[{name, specificity, prevalence}]`
- `get_available_tests(disease_names)` → `List[{id, name, diseases}]`
- `get_test_edges(test_id)` → `List[{disease, relationship, lr}]` — `relationship` is `"RULES_IN"` or `"RULES_OUT"`

Optionally, `get_test_threshold(test_id) -> str | None` is called via `getattr` in `answer_node` and forwarded to `EvidenceEvaluatorAgent` if the client implements it.

### API (`main.py`, FastAPI)

- `POST /api/v1/sessions` — starts session, returns `{session_id, next_question, initial_differential}`
- `POST /api/v1/sessions/{id}/answer` — resumes graph, returns `{updated_differential, next_question, final_diagnosis}`
- `GET /api/v1/sessions/{id}/diagnosis` — returns final report
- All `/api/v1/*` routes require `Authorization: Bearer <API_KEY>`
- CORS open for `http://localhost:3000`

### Design documents

`reports/` contains deep-dive analyses of every component. `reports/08_known_issues.md` is the master issue register — read it before touching any component logic.

### Test structure

LLM integration tests are marked `@pytest.mark.skip(reason="requires Ollama")` so the full `pytest` run is always LLM-free and fast. Unit tests mock the LLM via `unittest.mock.MagicMock`. The integration test for the full workflow is `orchestrator/tests/test_orchestrator.py::test_full_workflow`.
