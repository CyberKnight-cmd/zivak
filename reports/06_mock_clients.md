# Mock Database Clients — Deep Dive Report

**File:** `orchestrator/mock_clients.py`
**Type:** Mock implementations of Qdrant (vector DB) and Neo4j (graph DB)
**Purpose:** Enable full end-to-end testing without running real database servers

---

## 1. Design Principle

Both mock clients implement the same interface as the (not-yet-built) real clients.
The goal is: swap `MockQdrantClient` for `QdrantClient` and `MockNeo4jClient` for
`Neo4jClient` and the entire system should work identically.

Currently the real clients (`knowledge/qdrant_client.py`, `knowledge/neo4j_client.py`)
are empty stubs (0 bytes). The mock clients are the entire knowledge base.

```python
def get_clients(use_mock=True):
    if use_mock:
        return MockQdrantClient(), MockNeo4jClient()
    else:
        from knowledge.qdrant_client import QdrantClient    # empty stub
        from knowledge.neo4j_client  import Neo4jClient     # empty stub
        return QdrantClient(), Neo4jClient()
```

---

## 2. MockQdrantClient — Real Semantic Search on Mock Data

### What makes this "real"

Unlike a simple dict lookup, this mock uses an actual sentence-transformer model
(`all-MiniLM-L6-v2`, 80MB, CPU-only) to compute cosine similarity. It behaves
identically to production Qdrant — colloquial phrases, typos, and non-clinical
language all map correctly.

The blueprint specifies `text-embedding-3-small` (768-dim, OpenAI) for production.
The mock uses `all-MiniLM-L6-v2` (384-dim). Same interface, different model and
embedding dimensionality. When real Qdrant is connected, the corpus needs to be
re-embedded with the production model.

### Class-level caching (important)

```python
class MockQdrantClient:
    _model: SentenceTransformer = None      # shared across all instances
    _embeddings: np.ndarray = None          # shared across all instances

    def __init__(self):
        if MockQdrantClient._model is None:
            MockQdrantClient._model = SentenceTransformer("all-MiniLM-L6-v2")
            descriptions = [row[0] for row in self._CORPUS]
            MockQdrantClient._embeddings = self._model.encode(
                descriptions, normalize_embeddings=True, show_progress_bar=False
            )
```

Model + embeddings load ONCE per process, regardless of how many clients are
instantiated. `run_scenarios.py` instantiates one `MockQdrantClient` and reuses it
for all 15 scenarios — the model loads at startup, not per-scenario.

**Warning:** Not thread-safe for concurrent `encode()` calls. In a multi-threaded
FastAPI server, parallel requests could race. For production, use an actual Qdrant
server which handles concurrency internally.

### Corpus — 30 descriptions across 3 symptom clusters

```python
_CORPUS = [
    # Exertional Dyspnoea (HP:0002875) — 11 descriptions
    ("chest feels heavy when I walk",               "Exertional Dyspnoea", "HP:0002875"),
    ("I get breathless climbing stairs",             "Exertional Dyspnoea", "HP:0002875"),
    ...
    ("exertional dyspnoea",                         "Exertional Dyspnoea", "HP:0002875"),
    ("dyspnea on exertion",                         "Exertional Dyspnoea", "HP:0002875"),

    # Palpitations (HP:0001962) — 10 descriptions
    ("my heart is racing",                          "Palpitations", "HP:0001962"),
    ...

    # Vertigo (HP:0002321) — 9 descriptions
    ("the room is spinning",                        "Vertigo", "HP:0002321"),
    ...
]
```

Each symptom has multiple paraphrases — both clinical ("dyspnea on exertion") and
colloquial ("I get winded very easily"). The production Qdrant collection should
mirror this richness.

### search_symptom() — The only public method

```python
def search_symptom(self, user_text):
    query      = self._model.encode([user_text], normalize_embeddings=True)[0]
    scores     = self._embeddings @ query   # dot product = cosine similarity (L2-normed)
    best_idx   = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    if best_score < 0.30:   # _SIMILARITY_THRESHOLD
        return {"clinical_term": "Unknown symptom", "symptom_id": "HP:0000001", "score": best_score}

    _, clinical_term, symptom_id = self._CORPUS[best_idx]
    return {"clinical_term": clinical_term, "symptom_id": symptom_id, "score": best_score}
```

**Returns the SINGLE best match.** No aggregation, no second-place consideration.
If the query is equidistant between "Exertional Dyspnoea" and "Palpitations", the
first alphabetically (or first by corpus order) wins via `argmax`.

**Threshold = 0.30** is quite permissive. Cosine similarity of 0.30 on 384-dim
embeddings can match loosely related text. In practice, all 15 run_scenarios.py
symptoms are well within their clusters and score > 0.80.

### Corpus coverage gaps

The corpus covers ONLY 3 symptom clusters. Any chief complaint outside these
(e.g., "my knee hurts", "fever for 3 days", "blurry vision") returns
`HP:0000001` (unknown), and the system can't start a session.

---

## 3. MockNeo4jClient — The Entire Medical Knowledge Base

This is the most knowledge-dense file in the codebase. It encodes:
- Disease prevalence and specificity per symptom cluster
- Which tests are relevant for each disease
- Likelihood ratios per test per disease

### 3a. _DISEASES — Initial Differential Seeding

```python
_DISEASES = {
    "HP:0002875": [   # Exertional Dyspnoea
        {"name": "COPD",                    "specificity": 0.85, "prevalence": 0.065},
        {"name": "Asthma",                  "specificity": 0.80, "prevalence": 0.080},
        {"name": "Heart Failure",            "specificity": 0.90, "prevalence": 0.020},
        {"name": "Pulmonary Embolism",       "specificity": 0.75, "prevalence": 0.005},
        {"name": "Pneumonia",                "specificity": 0.70, "prevalence": 0.010},
        {"name": "Anemia",                   "specificity": 0.60, "prevalence": 0.030},
        {"name": "Interstitial Lung Disease","specificity": 0.78, "prevalence": 0.003},
    ],
    "HP:0001962": [   # Palpitations
        {"name": "Atrial Fibrillation",      "specificity": 0.88, "prevalence": 0.020},
        {"name": "SVT",                      "specificity": 0.82, "prevalence": 0.015},
        {"name": "Anxiety",                  "specificity": 0.65, "prevalence": 0.080},
        {"name": "Hyperthyroidism",           "specificity": 0.80, "prevalence": 0.012},
        {"name": "Anemia",                   "specificity": 0.60, "prevalence": 0.030},
    ],
    "HP:0002321": [   # Vertigo
        {"name": "BPPV",                     "specificity": 0.85, "prevalence": 0.060},
        {"name": "Vestibular Neuritis",       "specificity": 0.80, "prevalence": 0.015},
        {"name": "Meniere's Disease",         "specificity": 0.78, "prevalence": 0.008},
        {"name": "Central Vertigo",           "specificity": 0.90, "prevalence": 0.004},
    ],
}
```

**Initial probability calculation** (done by DifferentialEngine.initialize()):
- `raw_score = specificity × prevalence`
- Normalized, then floored at 2%

Initial probabilities for Exertional Dyspnoea (7 diseases):
| Disease       | spec × prev  | Approx start% |
|---------------|--------------|----------------|
| Asthma        | 0.064        | ~28%           |
| COPD          | 0.055        | ~24%           |
| Anemia        | 0.018        | ~10% (floored) |
| Heart Failure | 0.018        | ~10% (floored) |
| Pneumonia     | 0.007        | ~5%            |
| PE            | 0.004        | ~3%            |
| ILD           | 0.002        | ~2% (floored)  |

Note: Asthma starts ABOVE COPD despite COPD being the "canonical" dyspnoea disease,
because Asthma has higher prevalence (8% vs 6.5%). This is clinically correct.

### 3b. _TESTS — Available Tests Per Disease

```python
_TESTS = {
    "COPD":    [test_fev1, test_cxr_hyp, test_peak_flow, test_hrct],
    "Asthma":  [test_fev1, test_bronch, test_peak_flow],
    "Heart Failure": [test_bnp, test_echo, test_cxr_card],
    "Pulmonary Embolism": [test_ddimer, test_ctpa, test_wells],
    "Pneumonia": [test_cxr_inf, test_sputum, test_crp],
    "Anemia":  [test_cbc, test_ferritin],
    "Interstitial Lung Disease": [test_hrct, test_pft],
    "Atrial Fibrillation": [test_ecg, test_holter, test_echo_pal],
    "SVT":     [test_ecg, test_holter, test_electro],
    "Anxiety": [test_gad7, test_ecg],
    "Hyperthyroidism": [test_tsh],
    "BPPV":    [test_dix, test_roll],
    "Vestibular Neuritis": [test_hit, test_vng],
    "Meniere's Disease": [test_audio, test_vng, test_tymp],
    "Central Vertigo": [test_mri, test_hit],
}
```

`get_available_tests(disease_names)` iterates these in order, deduplicating by test ID:

```python
def get_available_tests(self, disease_names):
    tests = []
    seen_ids = set()
    for disease in disease_names:
        for test in self._TESTS.get(disease, []):
            if test["id"] not in seen_ids:
                tests.append(test)
                seen_ids.add(test["id"])
    return tests
```

The `diseases` field on each test object lists ALL diseases that share the test,
not just the one it was filed under. For example, `test_ecg` appears under both
"Atrial Fibrillation" and "SVT", but whichever disease is queried first "owns"
the copy in the list. The `diseases` field still correctly shows both.

**Key issue: called with TOP 5 only.**
```python
top_diseases = [d["name"] for d in state["differential"][:5]]
```
In a 7-disease dyspnoea differential, diseases ranked 6 and 7 (e.g., PE and ILD
at the bottom) never contribute their tests to the pool. If the correct diagnosis
is in position 6 or 7, its discriminating tests (e.g., test_ctpa for PE, test_pft
for ILD) may not be offered early. However, as other diseases are ruled out, the
bottom diseases rise and their tests become available.

**Single-test disease: Hyperthyroidism.**
Only has `test_tsh`. If this is never offered (e.g., not in top 5) and it's the
correct diagnosis, the session exhausts all palpitations tests without confirming
Hyperthyroidism.

### 3c. _LIKELIHOOD_RATIOS — The Core Knowledge

This dict maps test_id → list of {disease, relationship, lr} edges.
This is the encoded medical knowledge — the "what does this test result mean?" data.

#### Full LR table (positive result → RULES_IN = disease more likely)

**Exertional Dyspnoea tests:**

| Test      | Disease      | Rel       | LR+   | Clinical meaning |
|-----------|--------------|-----------|-------|------------------|
| test_fev1 | COPD         | RULES_IN  | 8.5   | FEV1/FVC <0.7 → obstruction |
| test_fev1 | Asthma       | RULES_IN  | 3.0   | FEV1/FVC <0.7 → possible asthma |
| test_fev1 | Heart Failure| RULES_OUT | 0.5   | Obstruction not typical in HF |
| test_fev1 | PE           | RULES_OUT | 0.6   | Obstruction not typical in PE |
| test_bronch| Asthma      | RULES_IN  | 6.0   | Reversibility = Asthma |
| test_bronch| COPD        | RULES_OUT | 0.15  | No reversibility → not Asthma, rules out |
| test_bronch| HF          | RULES_OUT | 0.5   | Bronchodilator doesn't help HF |
| test_bnp  | Heart Failure| RULES_IN  | 9.2   | Elevated BNP → HF |
| test_echo | Heart Failure| RULES_IN  | 12.0  | Reduced EF → HF confirmed |
| test_ctpa | PE           | RULES_IN  | 24.0  | CTPA confirms PE — strongest LR |
| test_cbc  | Anemia       | RULES_IN  | 15.0  | Low Hb → Anemia |
| test_hrct | ILD          | RULES_IN  | 20.0  | Honeycombing → ILD |

**Palpitations tests:**

| Test       | Disease | Rel      | LR+   | Clinical meaning |
|------------|---------|----------|-------|------------------|
| test_ecg   | AF      | RULES_IN | 15.0  | Irregularly irregular → AF |
| test_ecg   | SVT     | RULES_IN | 8.0   | Regular tachycardia → SVT |
| test_ecg   | Anxiety | RULES_OUT| 0.3   | Arrhythmia found → not pure anxiety |
| test_holter| AF      | RULES_IN | 6.0   | Captures paroxysmal AF |
| test_tsh   | Hyper   | RULES_IN | 12.0  | Low TSH, high T4 → Hyperthyroidism |

**Vertigo tests:**

| Test     | Disease         | Rel      | LR+  | Clinical meaning |
|----------|-----------------|----------|------|------------------|
| test_dix | BPPV            | RULES_IN | 12.0 | Positive Dix-Hallpike → BPPV |
| test_dix | Vestibular Neur | RULES_OUT| 0.3  | Positional nystagmus rules out VN |
| test_hit | Vestibular Neur | RULES_IN | 8.0  | Positive HIT → VN |
| test_hit | Central Vertigo | RULES_OUT| 0.2  | Positive HIT almost rules out central |
| test_mri | Central Vertigo | RULES_IN | 8.0  | Lesion found → central |
| test_mri | BPPV            | RULES_OUT| 0.1  | Lesion found almost eliminates BPPV |

### Missing cross-test links

Several diseases lack connections to tests for OTHER diseases in the same cluster:
- Anemia (dyspnoea) has no cross-links to COPD/HF/PE tests
- Hyperthyroidism has only 1 test and no RULES_OUT edges to other palpitation diseases
- This means if these diseases are the correct diagnosis, convergence is fast (their
  own tests have high LRs) but if they're WRONG diagnoses, they may linger because
  other tests don't rule them out

---

## 4. Known Issues & Gaps

### Issue 1: get_available_tests queries only top 5 diseases
Lower-ranked diseases' tests are excluded from consideration until those diseases
rise in the differential. This is by design (clinical efficiency) but can delay
diagnosis for diseases starting below rank 5.

### Issue 2: Anemia appears in TWO symptom clusters
Anemia is in both `HP:0002875` (dyspnoea) and `HP:0001962` (palpitations). The
`_TESTS["Anemia"]` entries are the same (CBC, ferritin) regardless of which cluster
the session started in. This is clinically correct — iron-deficiency anemia causes
both dyspnoea and palpitations, and the same tests confirm it either way.

### Issue 3: No scenario in run_scenarios.py for Anemia-palpitations
Anemia is in the palpitations disease list but there's no palpitations-context
Anemia scenario. If the Anemia disease rises in a palpitations session, the system
would ask CBC/ferritin — but there are no test answers prepared for this in
run_scenarios.py's FALLBACK_ANSWERS for palpitations-context questions.

### Issue 4: Hyperthyroidism has no RULES_OUT edges for other diseases
If the patient has Anxiety and test_tsh comes back normal, the evaluator returns:
- `neg_in` for Hyperthyroidism: lr = 1/12.0 = 0.083 (rules out Hyper strongly)
- `neg_out`: empty

This means a normal TSH rules out Hyperthyroidism but provides ZERO evidence for
or against any other disease. A negative TSH in an Anxiety scenario is a missed
opportunity to reduce the differential faster.

### Issue 5: No run_scenarios.py test for unknown symptom
If `HP:0000001` is returned (unknown symptom), `get_initial_differential` returns
an empty list. `DifferentialEngine.initialize([])` returns `[]`. The orchestrator
would fail or produce an empty differential. This code path is untested.

---

## 5. Interface Contract (for real client implementation)

```python
class QdrantClient:
    def search_symptom(self, user_text: str) -> Dict:
        # Returns: {"clinical_term": str, "symptom_id": str, "score": float}
        # symptom_id must be a valid HPO code matching Neo4j nodes
        # score is cosine similarity [0, 1]

class Neo4jClient:
    def get_initial_differential(self, symptom_id: str) -> List[Dict]:
        # Returns deepcopy of: [{"name": str, "specificity": float, "prevalence": float}]

    def get_available_tests(self, disease_names: List[str]) -> List[Dict]:
        # Returns: [{"id": str, "name": str, "diseases": List[str]}]
        # De-duplicated by "id"

    def get_test_edges(self, test_id: str) -> List[Dict]:
        # Returns: [{"disease": str, "relationship": "RULES_IN"|"RULES_OUT", "lr": float}]
```
