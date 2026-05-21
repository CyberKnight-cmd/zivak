# ZIVAK / DOCKTOR — System Overview & Blueprint Map

## 1. What We Are Building

**DOCKTOR** (codenamed ZIVAK internally) is an agentic AI clinical reasoning system.
Given a patient's chief complaint in plain language, it:

1. Maps the complaint to a clinical concept via semantic search
2. Seeds a Bayesian differential diagnosis from a disease knowledge graph
3. Asks the single highest-yield diagnostic question each turn
4. Updates disease probabilities with each answer using Bayes' theorem
5. Converges on a grounded, explainable final diagnosis

The core differentiator vs. symptom checkers: **LLM reasoning is constrained by a
knowledge-base firewall** — no claim exits the system without tracing to a graph node.
The system is **entropy-driven** (max information gain per question) and **Bayesian-transparent**
(every probability is auditable).

---

## 2. Seven-Layer Architecture (from Blueprint)

```
LAYER 1 — USER INTERFACE
  React 18 + Tailwind + Zustand
  - Chat input panel
  - Live differential panel (probability bars update every turn)
  - 60-second urgency mode with countdown timer

LAYER 2 — API GATEWAY
  FastAPI + Pydantic v2
  - JWT auth (currently static Bearer token)
  - Redis-backed session state (currently MemorySaver in dev)
  - Rate limiting
  - Input sanitisation before any agent call

LAYER 3 — AGENT ORCHESTRATION
  LangGraph state machine
  Five sub-agents:
    ├── Differential Engine   (Bayesian probability updater, pure math)
    ├── Q-Selector            (entropy-reduction question picker, LLM)
    ├── Evidence Evaluator    (maps answers to KB edges, LLM)
    ├── Confidence Judge      (gates diagnosis until thresholds met, pure logic)
    └── Explainability Agent  (grounded reasoning chain — NOT YET IMPLEMENTED)

LAYER 4 — KNOWLEDGE BASES
  - Disease Graph     (Neo4j ontology — PRESENTS_WITH, RULES_IN, RULES_OUT edges)
  - Semantic Index    (Qdrant 768-dim embeddings)
  - Evidence Base     (ICD-11, PubMed guidelines — future)

LAYER 5 — DATABASES
  - Neo4j       graph,     disease ontology & relations
  - Qdrant      vector,    semantic symptom/diagnosis search
  - PostgreSQL  relational, sessions, audit, doctor reviews  (NOT YET IMPLEMENTED)
  - Redis       in-memory, hot cache, session state, rate limits (NOT YET IMPLEMENTED)

LAYER 6 — SAFETY GUARDRAILS
  - Hallucination Firewall   (KB node tracing — partial: whitelist validation exists)
  - Evidence Threshold Gate  (min 4 data points — IMPLEMENTED in ConfidenceJudge)
  - Rare Disease Seeder      (floors rare conditions at 2% prior — IMPLEMENTED)

LAYER 7 — FEEDBACK LOOP
  - Doctor reviews → Case Store → nightly KB Updater (NOT YET IMPLEMENTED)
  - Qdrant re-embed + Neo4j weight adjust (NOT YET IMPLEMENTED)
```

---

## 3. Current Implementation State

| Component                  | Status          | Notes                                          |
|----------------------------|-----------------|------------------------------------------------|
| FastAPI gateway            | DONE            | Static Bearer, no JWT, no Redis                |
| LangGraph orchestrator     | DONE            | MemorySaver (dev), Redis-ready via param       |
| DifferentialEngine         | DONE            | Full Bayesian log-prob implementation          |
| QuestionSelectorAgent      | DONE            | Ollama/Groq switchable                         |
| EvidenceEvaluatorAgent     | DONE            | Ollama/Groq switchable                         |
| ConfidenceJudge            | DONE            | All 3 criteria + stability gate                |
| ExplainabilityAgent        | NOT IMPLEMENTED | Blueprint specifies as L3 sub-agent            |
| MockQdrantClient           | DONE            | Real semantic search (all-MiniLM-L6-v2)        |
| MockNeo4jClient            | DONE            | 3 symptom clusters, 16 diseases, 29 test edges |
| Real Neo4j client          | STUB (empty)    | knowledge/neo4j_client.py is 0 bytes           |
| Real Qdrant client         | STUB (empty)    | knowledge/qdrant_client.py is 0 bytes          |
| PostgreSQL (sessions/audit)| NOT IMPLEMENTED |                                                |
| Redis (hot cache)          | NOT IMPLEMENTED |                                                |
| React frontend             | NOT IMPLEMENTED |                                                |
| Doctor review loop         | NOT IMPLEMENTED |                                                |
| Nightly KB updater         | NOT IMPLEMENTED |                                                |

---

## 4. End-to-End Diagnostic Pipeline (12 steps from Blueprint)

```
Step 1  Chief complaint entry
        User types → FastAPI receives with session UUID

Step 2  Sanitisation & embedding
        FastAPI strips unsafe chars (_sanitize())
        Qdrant cosine search maps text → clinical concept (symptom_id)

Step 3  Initial differential seeding
        Neo4j scores diseases by P(specificity × prevalence)
        Rare Disease Seeder: floor at 2% prior
        Written to state + returned to UI

Step 4  Question selection
        Q-Selector calls LLM with current differential + available tests
        Returns single question maximising entropy reduction

Step 5  User answers
        Answer → Evidence Evaluator
        Maps to RULES_IN / RULES_OUT Neo4j edges
        Likelihood ratios applied

Step 6  Bayesian update
        P(D|E) = P(E|D)·P(D) / P(E)
        Log-space arithmetic; normalised; Redis + UI updated

Step 7  Confidence gate
        ≥4 evidence, top ≥75%, runner-up <40%
        Fail → back to Step 4
        Pass → proceed

Step 8  Hallucination firewall
        [PARTIAL] Every claim traced to Neo4j node
        Currently: whitelist validation in EvidenceEvaluator

Step 9  Explainability chain
        [NOT IMPLEMENTED] ExplainabilityAgent generates grounded reasoning

Step 10 Output rendered
        Diagnosis + confidence + reasoning → React
        Session written to PostgreSQL [NOT IMPLEMENTED]

Step 11 Doctor review (async)
        [NOT IMPLEMENTED]

Step 12 KB update (nightly)
        [NOT IMPLEMENTED]
```

---

## 5. Data Flow Diagram

```
Patient Input (plain text)
        │
        ▼
[_sanitize()] ──────────────────────────── strip control chars, cap 1000 chars
        │
        ▼
[MockQdrantClient.search_symptom()]
  all-MiniLM-L6-v2 cosine similarity
  → {clinical_term, symptom_id, score}
        │
        ▼
[MockNeo4jClient.get_initial_differential(symptom_id)]
  → [{name, specificity, prevalence}, ...]
        │
        ▼
[DifferentialEngine.initialize()]
  probability = max(spec × prev / total, 0.02)
  → [{name, probability, log_prob, ...}, ...]  sorted desc
        │
        ▼ ◄──────────────────────────────────────┐
[ConfidenceJudge.should_finalize()]              │
  evidence_count ≥ 4?                            │
  top prob ≥ 0.75?                               │
  runner-up < 0.40?                              │
  NOT unstable (delta > -10%)?                   │
  ────YES──► [finalize_node] ──► final report    │
  ────NO───►                                     │
        │                                        │
        ▼                                        │
[MockNeo4jClient.get_available_tests(top5)]      │
  → [{id, name, diseases}, ...]                  │
  minus already-asked tests                      │
        │                                        │
        ▼                                        │
[QuestionSelectorAgent.select_question()]        │
  LLM picks highest-yield test                   │
  → {question, test_id, reasoning}               │
        │                                        │
        ▼                                        │
   interrupt() ──► surface to user               │
        │                                        │
        ▼  (user answers)                        │
[MockNeo4jClient.get_test_edges(test_id)]        │
  → [{disease, relationship, lr}, ...]           │
        │                                        │
        ▼                                        │
[EvidenceEvaluatorAgent.evaluate()]              │
  LLM decides +/- polarity                       │
  copies matching pre-computed LR set            │
  → {rules_in: [...], rules_out: [...]}          │
        │                                        │
        ▼                                        │
[DifferentialEngine.update()]                    │
  log P += log(LR) per disease                   │
  log-sum-exp normalise                          │
  → updated differential sorted desc             │
        │                                        │
        └────────────────────────────────────────┘
```

---

## 6. LLM Configuration

| Setting          | Production (Groq)            | Local (Ollama)             |
|------------------|------------------------------|----------------------------|
| env flag         | USE_LOCAL_LLM=false          | USE_LOCAL_LLM=true         |
| model            | openai/gpt-oss-120b          | qwen2.5:14b                |
| provider         | langchain_groq.ChatGroq      | langchain_openai.ChatOpenAI|
| base_url         | Groq API                     | http://localhost:11434/v1  |
| Q-Selector temp  | 1                            | 1                          |
| Evidence temp    | 0                            | 0                          |
| max_tokens       | 512                          | 512                        |

---

## 7. Knowledge Base Coverage (Mock)

| Symptom Cluster     | HPO ID     | Diseases                                                                 | Tests |
|---------------------|------------|--------------------------------------------------------------------------|-------|
| Exertional Dyspnoea | HP:0002875 | COPD, Asthma, Heart Failure, Pulmonary Embolism, Pneumonia, Anemia, ILD | 10    |
| Palpitations        | HP:0001962 | Atrial Fibrillation, SVT, Anxiety, Hyperthyroidism, Anemia              | 6     |
| Vertigo             | HP:0002321 | BPPV, Vestibular Neuritis, Meniere's Disease, Central Vertigo            | 5     |

---

## 8. Gap vs. Blueprint — Priority List

| Gap                             | Blueprint Spec                        | Current State              | Priority |
|---------------------------------|---------------------------------------|----------------------------|----------|
| ExplainabilityAgent             | L3 sub-agent, grounded reasoning      | Not implemented            | HIGH     |
| SessionNotFoundError never raised| 404 on invalid session               | Returns 503 instead        | HIGH     |
| Double LLM call per turn        | 1 selector call per Q                 | 2 calls (LangGraph re-exec)| HIGH     |
| PostgreSQL session store        | Sessions, QA history, audit trail     | Not implemented            | MED      |
| Redis hot cache                 | Sub-10ms differential retrieval       | Not implemented            | MED      |
| Hallucination firewall          | Every claim traces to Neo4j node ID   | Whitelist only             | MED      |
| Real Neo4j / Qdrant clients     | Production databases                  | Empty stubs                | MED      |
| JWT auth                        | Per-user auth                         | Static Bearer token        | LOW      |
| Doctor review loop              | Corrections re-train KB               | Not implemented            | LOW      |
| 60-sec urgency mode             | Timer + ruthless Q-prioritisation     | Not implemented            | LOW      |
| Langfuse observability          | LLM call tracing                      | Not implemented            | LOW      |
