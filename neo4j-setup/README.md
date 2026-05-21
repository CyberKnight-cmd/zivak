# exclusively-neo4j

A self-contained pipeline that builds the Zivak Neo4j disease-symptom knowledge graph from open medical ontologies. No LLM required. No API keys required. Only Neo4j running locally via Docker.

---

## What gets built

```
12,127  Disease nodes
19,389  Symptom nodes
99,057  PRESENTS_WITH edges  (Disease → Symptom, sensitivity weight)
99,057  RULES_IN edges       (Symptom → Disease, LR+ weight)
99,057  RULES_OUT edges      (Symptom → Disease, LR− weight)
```

---

## Prerequisites

- Python 3.13
- Docker Desktop running
- `uv` installed (`pip install uv`)
- Neo4j running: `docker compose up -d neo4j`
  (use the `docker-compose.yml` in the project root)

---

## Setup steps

```
Step 1: Copy .env.example to .env and fill in NEO4J_PASSWORD
Step 2: Read 1_download_ontologies.md and download the three files
Step 3: uv run 2_parse_ontologies.py
Step 4: uv run 3_compute_lr_table.py
Step 5: uv run 4a_tag_hpo_subtrees.py
Step 6: uv run 4_load_neo4j.py
Step 7: uv run 5_verify.py  — must show 20/20 PASS
```

---

## Verify the graph

Open http://localhost:7474 in your browser.
Log in with credentials from `.env`.
Run: `MATCH (d:Disease) RETURN count(d)` — expect **12,127**

---

## Data files location

All scripts read/write to `../data/processed/` by default (relative to this folder, inside the repo). If running standalone outside the repo, set `DATA_DIR` in `.env`.

---

## What the graph is used for

The Bayesian diagnostic engine queries this graph at runtime. Disease probabilities update using likelihood ratios stored on edges. See the root `README.md` for the full system architecture.
