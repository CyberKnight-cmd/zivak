# Neo4j Integration — Implementation Tracker

Mark each step `[x]` when done. If Claude context resets, resume from the first `[ ]` step.

## Status
- [x] Step 0 — Write this file
- [ ] Step 1 — Run neo4j-setup pipeline (USER action — requires Docker)
- [x] Step 2 — Create `knowledge/` package
- [x] Step 3 — Write `knowledge/neo4j_client.py`
- [x] Step 4 — Write `knowledge/qdrant_client.py` (Neo4j primary, symptom_index.json fallback)
- [x] Step 5 — Update `orchestrator/orchestrator.py`
- [x] Step 6 — `main.py` already handles real clients (USE_MOCK=false) — no changes needed
- [x] Step 6b — Fixed ONTOLOGIES_DIR in 2_parse_ontologies.py, 4a_tag_hpo_subtrees.py, .env.example
- [x] Step 6c — Added neo4j, pandas, tqdm, pronto to pyproject.toml; installed via uv
- [x] Step 6d — Created docker-compose.yml at project root
- [ ] Step 7 — Add Neo4j vars to `.env` (USER action)
- [x] Step 8 — pytest: 81 passed, 3 skipped — all green
- [ ] Step 9 — Smoke test real pipeline (USER action — requires Neo4j running)

---

## Database schema

```
(Disease {
    id:         "DOID:3083"          ← primary key, used for all Neo4j lookups
    name:       "COPD"               ← display name, used by EvidenceEvaluator (case-insensitive)
    icd:        ["J44.1"]
    omim_id:    "OMIM:606391"
    orpha_id:   null
    prevalence: 0.01                 ← uniform for all diseases (hardcoded in pipeline)
})

(Symptom {
    id:         "HP:0002875"         ← primary key = the test_id in our architecture
    name:       "Exertional dyspnoea"
    synonyms:   ["Dyspnea on exertion", ...]
    category:   "symptom" | "lab_finding" | "sign"
})

(Disease)-[:PRESENTS_WITH {sensitivity: 0.545}]->(Symptom)   ← P(symptom|disease)
(Symptom)-[:RULES_IN      {likelihood_ratio: 5.45}]->(Disease) ← LR+  = sensitivity / background
(Symptom)-[:RULES_OUT     {likelihood_ratio: 0.50}]->(Disease) ← LR-  = (1-Se)/(1-Bg)
```

Key invariant: every PRESENTS_WITH edge has a matching RULES_IN and RULES_OUT edge (same
disease-symptom pair). There are 99,057 edges of each type.

HPO symptom terms ARE the diagnostic questions. `question["test_id"]` is now `"HP:0002875"`.

---

## Folder reference map

| File | Read before writing |
|---|---|
| `knowledge/neo4j_client.py` | `neo4j-setup/neo4j_client.py`, `neo4j-setup/neo4j_queries.py`, `orchestrator/mock_clients.py` |
| `knowledge/qdrant_client.py` | `orchestrator/mock_clients.py` (MockQdrantClient pattern), `knowledge/neo4j_client.py` |
| `orchestrator/orchestrator.py` | current `orchestrator/orchestrator.py`, `knowledge/neo4j_client.py` |

---

## Step 1 — Run the pipeline (USER)

Prerequisites: Docker Desktop running.

```bash
docker compose up -d neo4j          # start Neo4j container

cd neo4j-setup
uv run 2_parse_ontologies.py        # → data/processed/{disease_index,symptom_index,...}.json
uv run 3_compute_lr_table.py        # → data/processed/lr_table.csv  (~100k rows)
uv run 4a_tag_hpo_subtrees.py       # → data/processed/symptom_categories.json
uv run 4_load_neo4j.py              # → loads 12k diseases, 19k symptoms, 99k edges each
uv run 5_verify.py                  # → must print all PASS
```

Data files are already in `data/`: `doid.obo`, `hp.obo`, `phenotype.hpoa`.

---

## Step 7 — .env additions (USER)

Add to the project `.env`:
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=<password from docker-compose>
USE_MOCK=false
```

---

## Step 8 — Run tests

```bash
pytest agents/tests/ engines/tests/ orchestrator/tests/ -v
# Expected: 81 passed, 3 skipped
```

---

## Step 9 — Smoke test

```python
# Run from project root with Neo4j running and USE_MOCK=false
from knowledge.neo4j_client import Neo4jClient
from knowledge.qdrant_client import QdrantClient

neo4j  = Neo4jClient()
qdrant = QdrantClient(neo4j)

# Test Qdrant fallback
match = qdrant.search_symptom("chest feels heavy when walking")
print(match)   # should show HP:0002875 or similar

# Test differential seeding
diff = neo4j.get_initial_differential(match["symptom_ids"])
print(diff[:3])   # top 3 diseases with probabilities

# Test available tests
tests = neo4j.get_available_tests([d["disease_id"] for d in diff[:5]])
print(f"{len(tests)} available tests")   # should be > 5

# Test LR edges
edges = neo4j.get_test_edges(tests[0]["id"])
print(edges[:3])  # should show RULES_IN and RULES_OUT entries
```

---

## Architecture notes

### Why HPO terms are the tests
The mock used hand-crafted `test_fev1`, `test_bnp` etc. with manually set LRs.
The real graph has LRs derived from actual epidemiology (HPOA annotations) for 99k
disease-symptom pairs. Each HP symptom term IS the "test" — asking "do you have this
symptom?" and getting yes/no triggers the RULES_IN or RULES_OUT LR update.

### What does NOT change
`DifferentialEngine`, `ConfidenceJudge`, `EvidenceEvaluatorAgent`, `QuestionSelectorAgent`,
all unit tests, `orchestrator/mock_clients.py` — zero modifications.

### disease_id flow
`get_initial_differential` returns `{name, disease_id, specificity, prevalence}`.
`DifferentialEngine.initialize()` stores ALL input keys (it only ADDS raw_score, probability,
log_prob — doesn't strip other fields). So `disease_id` flows into `state["differential"]`
automatically. `question_node` reads it for Neo4j lookups.
