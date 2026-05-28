# ZIVAK Data Pipeline
**Complete Technical Reference**
**Last updated:** 2026-05-27 — commit `8eeede2`
**Status:** Neo4j reloading on new Aura instance, Qdrant repopulating

---

## 1. The Problem This Pipeline Solves

ZIVAK's diagnostic engine is Bayesian. Every time a patient answers a
question, the probability of each candidate disease shifts up or down
by a factor called the **likelihood ratio (LR)**. The entire diagnostic
quality of the system depends on whether these LR values are clinically
meaningful.

**Option A — Trust the LLM to generate LRs:** Rejected. LLMs hallucinate
medical numbers. There is no audit trail.

**Option B — Build a knowledge graph from published medical ontologies:**
Chosen. Every LR value is derived from three decades of clinical research
encoded in the Human Phenotype Ontology, the Disease Ontology, and the
HPO disease-phenotype annotation corpus. Every probability shift is
traceable to a specific graph edge.

---

## 2. Source Datasets

### 2.1 Human Phenotype Ontology — `hp.obo`
19,389 clinical abnormalities — symptoms, signs, lab findings — each with
a unique `HP:XXXXXXX` identifier, standardized name, and synonyms.

### 2.2 Disease Ontology — `doid.obo`
12,127 disease entries with DOID identifiers and cross-references to
ICD-10/ICD-9, OMIM, ORPHA, MESH, UMLS. Also contains `alt_id` fields
(old IDs merged into current DOID) and `replaced_by` fields (obsolete
terms pointing to their active replacement) — used for the Gap 1 fix.

### 2.3 HPO Disease-Phenotype Annotations — `phenotype.hpoa`
264,245 rows. Each row: "disease X presents with symptom Y with frequency Z."
The only source of `P(symptom | disease)` — sensitivity values for all LRs.

### 2.4 Orphanet Prevalence Data — `en_product9_prev.xml`
Population prevalence classes per ORPHA ID. Used to set `P(disease)` —
the prior probability before any symptoms are observed.

---

## 3. Pipeline Scripts — Execution Order

```
Script                           Output
────────────────────────────────────────────────────────────────────
scripts/1_parse_ontologies.py  → disease_index.json     (12,127)
                                  symptom_index.json     (19,389)
                                  hpoa_annotations.json  (264,245)
                                  omim_to_doid.json
                                  orpha_to_doid.json
                                  obsolete_doid_map.json (Gap 1 fix)

scripts/2_parse_orpha_prev.py  → orpha_prevalence.json

scripts/3_patch_disease_prev.py→ disease_index.json (+ prevalence fields)

scripts/4_compute_lr_table.py  → lr_table.csv           (105,865+ rows)

scripts/5_tag_hpo_subtrees.py  → symptom_categories.json (19,389)

scripts/6_load_neo4j.py        → Neo4j graph (Aura)
                                  edge_reclassifications.json
                                  edge_node_missing.json

scripts/7_verify_neo4j.py      → confirms graph counts

scripts/8_populate_qdrant.py   → Qdrant "symptoms" collection (19,389)
                                  qdrant_id_map.json

scripts/9_verify_qdrant.py     → confirms search quality

scripts/10_generate_report.py  → reports/final_pipeline_report.md
```

---

## 4. Core Mathematical Calculations

### 4.1 Sensitivity — P(symptom | disease)

Parsed from `frequency_raw` in `phenotype.hpoa`:

```
HP:0040280  Obligate      → 1.000
HP:0040281  Very frequent → 0.895
HP:0040282  Frequent      → 0.545
HP:0040283  Occasional    → 0.170
HP:0040284  Very rare     → 0.025
HP:0040285  Excluded      → row dropped
Ratio "6/10"              → 0.600
Percentage "80%"          → 0.800
Empty / unknown           → row dropped
```

Clamped to `[0.005, 0.995]` before LR computation.

### 4.2 Disease Prevalence — P(disease)

Five-tier assignment (priority order):

| Tier | Source | Count | Value range |
|---|---|---|---|
| 1 | Real ORPHA Point prevalence | 1,346 | 0.0000005 – 0.005 |
| 2 | ICD chapter default | 3,954 | 0.005 – 0.05 |
| 3 | ORPHA xref but not in XML | 478 | 0.00001 |
| 4 | OMIM prefix heuristic | ~2,000 est. | 0.0001 or 0.000001 |
| 5 | Ultra-rare default | <4,349 est. | 0.000001 |

ICD chapter defaults (GBD-derived):
```
J (Respiratory) 0.05 · I (Cardiovascular) 0.05 · E (Endocrine) 0.03
K (Digestive) 0.03 · M (Musculoskeletal) 0.03 · F (Mental) 0.02
G (Neurological) 0.02 · N (Genitourinary) 0.02 · L (Skin) 0.02
D (Blood) 0.01 · C (Neoplasms) 0.01 · H (Eye/Ear) 0.01
A,B (Infectious) 0.01 · Q (Congenital) 0.005 · R (Signs) 0.005
```

OMIM prefix heuristic:
```
1xxxxx → 0.0001  (phenotype series, historically common conditions)
2xxxxx-6xxxxx → 0.000001  (rare Mendelian disorders)
```

### 4.3 Background Rate — P(symptom | ¬disease)

**Corrected formula (prevalence-weighted):**
```
background_rate(hp_id) = Σ prevalence_i × sensitivity_i
                         for all diseases i annotated with hp_id
```

Clamped to `[0.001, 0.999]`.

### 4.4 Likelihood Ratios

```
LR+ = sensitivity / background_rate
LR− = (1 − sensitivity) / (1 − background_rate)
clip: both to [0.01, 100]
```

**COPD example:**
```
sensitivity = 0.895, background_rate ≈ 0.08
LR+ = 0.895 / 0.08 = 11.2  (obstruction confirms COPD)
LR− = 0.105 / 0.92 = 0.11  (normal spirometry rules out COPD)

If COPD prior = 40%:
posterior_odds = (0.4/0.6) × 11.2 = 7.47
posterior_prob = 7.47 / 8.47 = 88.2%
```

### 4.5 Bayesian Update (DifferentialEngine)

Operates in log-probability space to prevent numerical underflow.

**Initialization:**
```
raw_score_i = prevalence_i × sensitivity_i
prob_i      = raw_score_i / Σ(raw_score_j)
rare floor  = max(prob_i, 0.02)  → re-normalise
log_prob_i  = log(prob_i)
```

**Per-answer update:**
```
POSITIVE result on test T:
  log_prob_i += log(LR+_i)   if RULES_IN edge exists
  log_prob_i += log(LR-_i)   if RULES_OUT edge exists
  log_prob_i += 0             if no edge

Log-sum-exp normalisation:
  m          = max(log_prob_i)
  log_total  = m + log(Σ exp(log_prob_i − m))
  log_prob_i -= log_total
  prob_i      = exp(log_prob_i)
```

---

## 5. Neo4j Graph Schema and Edge Direction Logic

```
(Disease {id, name, icd, omim_id, orpha_id, prevalence, prevalence_source})
(Symptom {id, name, synonyms, category})

(Disease)-[:PRESENTS_WITH {sensitivity}]→(Symptom)
(Symptom)-[:RULES_IN      {likelihood_ratio}]→(Disease)   LR always > 1.0
(Symptom)-[:RULES_OUT     {likelihood_ratio}]→(Disease)   LR always < 1.0
```

**Edge assignment per lr_table row:**

```
Case 1 — lr_positive >= 1.0 (~90.6% of rows):
  RULES_IN  lr = lr_positive  (presence argues for disease)
  RULES_OUT lr = lr_negative  (absence argues against)

Case 2 — lr_positive < 1.0 (~9.4% of rows — reclassified):
  RULES_IN  lr = lr_negative  (absence argues for disease)
  RULES_OUT lr = lr_positive  (presence argues against)
```

Zero edges dropped. 9,915 reclassified.

---

## 6. Symptom Categorization

```
symptom     — patient-reported (9,209 terms)
lab_finding — objective test results (2,345 terms)
sign        — clinician-observed (7,835 terms)
```

`get_available_tests()` in `knowledge/neo4j_client.py` returns only
`symptom` and `lab_finding` for diagnostic questions. Signs excluded —
patients cannot self-report structural findings.

Qdrant indexes ALL 19,389 terms including signs — a patient describing
"barrel chest" should still map to a HP ID even if that HP ID is
excluded from question selection.

---

## 7. Pipeline Results — All Three Rounds

| Metric | Baseline | Round 1 | Round 2 | Round 3 |
|---|---|---|---|---|
| lr_positive min | 0.0253 | 0.7094 | 0.0100 | 0.0100 |
| lr_positive max | 995.0 | 100.0 | 100.0 | 100.0 |
| lr_positive mean | 135.61 | 94.56 | 31.98 | 31.98 |
| lr_positive median | 37.10 | 100.0 | 10.68 | 10.68 |
| Cap hit rate | 30.8% | 90.4% | 19.8% | 19.8% |
| Edge direction errors | — | 134 | 9,915 | 0 |
| Reclassified edges | — | — | — | 9,915 |
| PRESENTS_WITH count | 99,057 | 99,185 | 99,185 | 99,185 |

Round 1 made things temporarily worse (cap hit 30.8% → 90.4%) because
dropping to ORPHA-only defaults made background_rates near-zero.
Round 2's ICD chapter defaults fixed this. Round 3 recovered 9,915
dropped edges via reclassification.

---

## 8. Known Gaps and Fixes

### Gap 1 — 6,680 Missing Edges (Being Fixed in This Run)

**Root cause:** `phenotype.hpoa` references disease IDs via OMIM/ORPHA
which bridge to DOIDs via `omim_to_doid.json` and `orpha_to_doid.json`.
Some of these target DOIDs exist in the bridge map but NOT in
`disease_index.json` — because `doid.obo` marked them as obsolete or
merged them into a different DOID during ontology curation.

When `6_load_neo4j.py` tries to MERGE an edge referencing a non-existent
Disease node, the MERGE silently produces no edge. The gap was 6,680 rows.

**Fix implemented in this run:**

`scripts/1_parse_ontologies.py` now parses two additional field types from
`doid.obo`:

- `alt_id`: alternate IDs that all resolve to the same current DOID.
  Example: `DOID:9999` was renamed to `DOID:3083` — any annotation
  referencing `DOID:9999` should load under `DOID:3083`.
- `replaced_by`: obsolete term that was superseded by a new DOID.
  Example: `DOID:0001` is obsolete, `replaced_by DOID:0002`. An
  annotation referencing `DOID:0001` should load under `DOID:0002`.

This produces `data/processed/obsolete_doid_map.json`:
```json
{ "DOID:old1": "DOID:current1", "DOID:old2": "DOID:current2", ... }
```

`scripts/6_load_neo4j.py` loads this map and for each lr_table row
whose `disease_doid` is absent from the loaded Disease node set,
attempts resolution through the map before adding to `edge_node_missing`.

**Expected improvement:** The 6,680 gap should shrink significantly.
Some rows may still be unresolvable (DOIDs that were deleted entirely
from the ontology with no replacement). These remain in
`edge_node_missing.json` with `reason: "no_resolution_found"`.

### Gap 2 — 52.4% Ultra-Rare Default (Being Fixed in This Run)

**Root cause:** 6,349 diseases had no ICD code and no ORPHA xref,
all receiving `prevalence = 0.000001`. Most of these have OMIM IDs.

**Fix implemented in this run:**

`scripts/3_patch_disease_prevalence.py` now applies a fifth tier:
OMIM ID prefix heuristic. OMIM numbers in the 1xxxxx range tend to
cover better-characterized (often more common) phenotypes that were
catalogued first. Numbers 2xxxxx and above are predominantly rare
Mendelian disorders.

```
OMIM 1xxxxx → prevalence = 0.0001
OMIM 2xxxxx–6xxxxx → prevalence = 0.000001
```

This is an approximation — OMIM numbering is historical, not strictly
frequency-ordered. But it differentiates the previously flat group
into two tiers, improving background_rate calculation for OMIM-mapped
diseases that were previously treated identically to truly unknown ones.

**Expected improvement:** `ultra_rare_default` count drops from 6,349
to below 4,500. `omim_prefix` becomes a new source tier.

### Gap 3 — Remaining (Post-Qdrant Scope)

| Issue | Status | Planned fix |
|---|---|---|
| Remaining unresolvable edge_node_missing | Monitored | Accept — true ontology gaps |
| Prevalence for OMIM-only diseases | Partially fixed | Full fix: OMIM morbidmap |
| Common disease prevalence accuracy | Partial (ICD chapter) | Full fix: GBD dataset |
| New HPOA annotations | Static | Re-run pipeline on new release |

---

## 9. Qdrant — Semantic Symptom Search Layer

### Model
`NeuML/pubmedbert-base-embeddings` — 768d, PubMed fine-tuned.
Natively understands clinical abbreviations (SOB, FEV1/FVC, BNP).

### What Is Indexed
All 19,389 HPO terms (symptom + lab_finding + sign).
Signs included for maximum query coverage — a patient describing
"barrel chest" maps to a HP ID even though it won't be asked as
a question.

Text per point:
```
text = name + " " + " ".join(synonyms)
```

### Query Preprocessing
Three stages before embedding:
1. Strip leading framing (`"my patient has"`, `"I have"`, age demographics)
2. Sentence split → term split (preserving `FEV1/FVC`, `non-ST-elevation`)
3. Deduplicate terms

STOP_WORDS includes `"i"`, `"me"`, `"my"` to prevent `"I smoke"` from
passing through as a standalone term after framing strip.

### Gibberish Rejection
Minimum score threshold: `QDRANT_MATCH_THRESHOLD=0.45` (tunable).
Queries where all cosine scores fall below threshold return:
`{"matched": False, "symptom_ids": [], "score": 0.0}`

Orchestrator `seed_node` returns a user-facing error for no-match:
`finalization_reason: "no_symptom_match"`

### Thread Safety
`_get_model()` and `_get_qdrant()` use double-checked locking with
`threading.Lock()` — safe under FastAPI/Uvicorn concurrent requests.

---

## 10. How All Components Connect

```
Patient: "I get winded going upstairs, smoked for 30 years"
    │
    ▼ QdrantClient._preprocess()
      → ["winded going upstairs"]
    │
    ▼ embed + search (pubmedbert-base-embeddings)
      → HP:0002875 (Exertional dyspnea) score=0.87 matched=True
    │
    ▼ Neo4jClient: PRESENTS_WITH query on [HP:0002875]
      → diseases × prevalence × sensitivity
    │
    ▼ DifferentialEngine.initialize()
      raw_score = prevalence × sensitivity
      COPD:  0.00005 × 0.895 = 0.0000448
      Asthma: 0.005  × 0.800 = 0.0040
      HF:    0.05    × 0.300 = 0.0150
      → normalise → rare floor → log_prob
    │
    ▼ InformationGainEngine: EIG(T) = H(P) - E[H(P|T)]
      → top 5 tests ranked by entropy reduction
    │
    ▼ QuestionSelectorAgent (Gemini Flash, temp=1)
      → "What was your FEV1/FVC ratio?"
    │
    ▼ Patient answers "0.55 — obstruction"
    │
    ▼ EvidenceEvaluatorAgent (Gemini Flash, temp=0)
      → POSITIVE
      → RULES_IN: COPD LR=11.2, Asthma LR=3.0
      → RULES_OUT: HF LR=0.5
    │
    ▼ DifferentialEngine.update()
      log_prob += log(LR) per disease
      log-sum-exp normalise
      → COPD 22% → 72%, Asthma 21% → 24%, HF 8% → 2%
    │
    ▼ ConfidenceJudge: min_evidence=4, top≥75%, runner_up<40%
      → keep asking or finalize
```

---

## 11. What Happens Next

### This Run
- New Neo4j Aura instance fully loaded (Gap 1 + Gap 2 fixes applied)
- Qdrant repopulated with all 19,389 symptoms
- `reports/final_pipeline_report.md` generated

### Near-Term
- OMIM morbidmap integration — proper prevalence for OMIM-only diseases
- GBD dataset — accurate common disease prevalence
- HPOA re-run when new annual release drops

### None of These Require Architecture Changes
All future improvements are additional inputs to the same scripts,
producing a richer lr_table.csv and disease_index.json.