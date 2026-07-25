# ZIVAK Diagnostic Pipeline — Root Cause Analysis

## EXECUTIVE SUMMARY

Six independent failure modes were identified across the retrieval, data, and API layers. They compound each other but have distinct sources. The `"winded going upstairs" → Furuncle` failure originates entirely in the retrieval layer (Problems 1–4). The `RULES_IN / RULES_OUT contradiction` is an independent data-loading bug (Problem 5). Forced finalization transparency is a presentation problem, not a logic problem (Problem 6).

---

## 1. TEXT → HPO TRACE: Exact Execution Path

```
user_input = "I got winded up while going upstairs."

QdrantClient.search(query)

  _preprocess():
    SENT_RE splits by [.!?] → one sentence (no split)
    _strip_framing():
      Tries all 11 FRAMING_PATTERNS.
      None match "I got winded..."
      Patterns cover: "I've been", "I am having", "I'm feeling",
                      "I am a X-year-old" — NOT "I got X"
      → sentence UNCHANGED: "I got winded up while going upstairs."
    SPLIT_RE: no delimiters found, no split
    Filters: passes len>2, not digit-only, not stop-word
    → terms = ["I got winded up while going upstairs."]

  _embed(terms):
    Encodes full sentence with NeuML/pubmedbert-base-embeddings

  _search_one(vector, limit=5):
    Returns from Qdrant "symptoms" collection:
      HP:0020083  Furuncle     score=0.626   ← highest
      HP:0025251  (unknown)    score≈0.624
      HP:0025249  (unknown)    score≈0.620
      HP:0031848  (unknown)    score≈0.61x
      HP:6001050  (unknown)    score≈0.60x

  confident = {hp: h for h in best if score >= 0.60}
    → All 5 accepted (all >= 0.60)
  top = HP:0020083, Furuncle, 0.626

  Returns:
    { "clinical_term": "Furuncle", "symptom_id": "HP:0020083",
      "symptom_ids": [HP:0020083, HP:0025251, HP:0025249, HP:0031848, HP:6001050],
      "score": 0.626, "matched": True }

orchestrator.seed_node():
  neo4j.get_initial_differential(symptom_ids=all_5, term="Furuncle")
  MATCH (d:Disease)-[r]-(s:Symptom) WHERE s.id IN $hp_ids
  → 2 diseases returned

differential = [hypotrichosis 7, keratosis pilaris atrophicans]
```

---

## 2. QDRANT DATA CONSTRUCTION

Confirmed from `scripts/8_populate_qdrant.py` (line 77):

```python
text = name + (" " + " ".join(synonyms) if synonyms else "")
```

**Per Qdrant point:**
| Field | Value |
|---|---|
| `hp_id` | HPO identifier (e.g. HP:0020083) |
| `name` | HPO preferred name (e.g. "Furuncle") |
| `synonyms` | HPO formal synonym list (e.g. ["Boil"]) |
| `category` | Symptom category tag |
| `text` | `name + " " + synonyms` — **this is what is embedded** |

**What is NOT embedded:**
- HPO term definitions or descriptions
- Layperson vocabulary ("winded", "out of puff", "can't catch my breath")
- Contextual phrases ("on exertion", "when climbing stairs")
- Any disease association

**Critical observation from population script (lines 132–148):**

```python
# Test: HP:0002875 must be in top 3 AND score >= 0.70
test_vec = model.encode(["shortness of breath going up stairs"], ...)
assert top_score >= 0.70, ...
```

This test runs at population time and **passed**. Therefore HP:0002875 ("Exertional dyspnea") exists in Qdrant and is retrievable with clinical language. The failure is that `"I got winded up while going upstairs"` is not clinical language — it is layperson language that PubMedBERT has never been trained to associate with dyspnea.

---

## 3. WHY "I GOT WINDED UP WHILE GOING UPSTAIRS" → FURUNCLE

### Root Cause A — PubMedBERT Cannot Map Layperson Vocabulary

`NeuML/pubmedbert-base-embeddings` is fine-tuned on PubMed abstracts. Clinicians write "exertional dyspnea", "dyspnea on exertion", or "shortness of breath on exertion" — never "winded." The word "winded" as a casual layperson synonym for dyspnea is essentially absent from PubMed literature. PubMedBERT's semantic neighbourhood for "winded" is wound/skin/injury-adjacent, not respiratory — hence the similarity to Furuncle.

### Root Cause B — `_preprocess()` Doesn't Strip "I got" Framing

`FRAMING_PATTERNS` (qdrant_client.py lines 25–37) covers 11 patterns but is missing all past-tense conversational openers:
- Missing: `"I got"`, `"I had"`, `"I noticed"`, `"I keep"`, `"I started"`, `"I've started"`

The query embedded is the full sentence `"I got winded up while going upstairs."` instead of the stripped core `"winded going upstairs"`. The grammatical noise shifts the vector centroid away from the respiratory cluster.

### Root Cause C — No Margin Check

All 5 top results scored between 0.60 and 0.626 — a range of only 0.026. This is a flat cluster with no discriminative signal. The system accepted the top result with 0.002 margin over 2nd place as if it were a confident 0.90/0.61 separation. There is no mechanism to detect or handle this ambiguity.

### Where Is HP:0002875 (Exertional Dyspnea)?

Based on the population test, HP:0002875 scores ≥ 0.70 for "shortness of breath going up stairs." For "I got winded up while going upstairs", it likely falls below 0.60 and is **not in the top 5 at all**. The logs confirm it: `symptom_ids=['HP:0020083', 'HP:0025251', 'HP:0025249', 'HP:0031848', 'HP:6001050']` — HP:0002875 is absent.

---

## 4. RETRIEVAL QUALITY SUMMARY

| Issue | Severity | Confidence |
|---|---|---|
| PubMedBERT can't map layperson vocabulary | CRITICAL | Certain |
| `_preprocess()` misses "I got/had/noticed/keep" framing | HIGH | Certain |
| No margin gate — 0.626/0.624 treated same as 0.90/0.61 | HIGH | Certain |
| HPO corpus lacks layperson synonyms | MEDIUM | Certain |
| Single-vector, no multi-hypothesis query expansion | MEDIUM | Likely |

The correct HPO concept EXISTS in the corpus. The failure is: (a) the model can't bridge layperson vocabulary, (b) the preprocessing didn't strip grammatical noise, and (c) the system had no way to know it was uncertain.

---

## 5. ABSTENTION / CONFIDENCE

### Current Behavior

```python
confident = {hp: h for hp, h in best.items() if h["score"] >= self._threshold}
if not confident:
    return self._no_match()
```

Binary gate only. No margin check. A score of 0.626 with 2nd at 0.624 (margin=0.002) is processed identically to 0.90 with 2nd at 0.55 (margin=0.35).

### The Correct Mechanism: Two-Metric Gate

**Metric 1 — Absolute Score**: `top_score >= MATCH_THRESHOLD` (existing — prevents globally weak matches)

**Metric 2 — Relative Margin**: `top_score - 2nd_score >= MARGIN_THRESHOLD` (new — detects ambiguous flat clusters)

When Metric 1 passes but Metric 2 fails → `ambiguous=True` → orchestrator asks for clarification.

When both pass → proceed as normal.

### How to Set Thresholds Empirically

Build a benchmark of `(patient_phrase, expected_HP_id)` pairs (Section 10). For each pair record:
- `top_score`, `2nd_score`, `margin`, `correct` (bool)

Then:
- Plot `top_score` vs `correct` → find score where precision collapses (calibrates MATCH_THRESHOLD)
- Plot `margin` vs `correct` → find margin where precision collapses (calibrates MARGIN_THRESHOLD)

Do NOT guess thresholds. Use the benchmark data. Suggested starting points: MATCH_THRESHOLD=0.60 (keep current), MARGIN_THRESHOLD=0.04 (to be validated).

---

## 6. DOWNSTREAM TRACE: Was Any Downstream Component Wrong?

Given that the retrieval produced Furuncle → 2 diseases:

| Component | Behavior | Verdict |
|---|---|---|
| `DifferentialEngine.initialize()` | Seeded with 2 diseases, uniform priors | **Correct given input** |
| `ConfidenceJudge.should_finalize()` | 2 diseases, runner-up too close → keep asking | **Correct given input** |
| `Neo4j.get_available_tests()` | Returned HPO terms for hypotrichosis/keratosis → hair/skin | **Correct given input** |
| `EIG.rank_by_eig()` | Ranked hair/skin questions by max discriminative information gain | **Correct given input** |
| `QuestionSelectorAgent` | Phrased hair/skin questions naturally | **Correct given input** |
| `EvidenceEvaluatorAgent` | Evaluated "no" answers and applied LRs | **Correct given input** |

**All downstream components behaved correctly given the wrong initial differential.** Do not fix any of them for this failure. The root cause is entirely upstream in retrieval.

---

## 7. EIG / QUESTIONSELECTOR MALFUNCTION?

No. They worked correctly:
- EIG correctly ranked "missing eyelashes", "hair thinning" as maximally discriminating between hypotrichosis 7 and keratosis pilaris atrophicans
- QuestionSelector correctly translated those HPO concepts into natural patient language

These components are not the problem. Rewriting them would fix nothing about this failure mode.

---

## 8. CONTRADICTORY LR BUG — SOURCE IDENTIFIED

### Root Cause: `6_load_neo4j.py` `part_d_edges()` Creates Both Edge Types Unconditionally

```python
# Lines 257–263 of 6_load_neo4j.py
record = {
    "doid": doid, "hp_id": hp_id, "sensitivity": sensitivity,
    "rules_in_lr": rules_in_lr, "rules_out_lr": rules_out_lr,
}
pw_batch.append(record)   # PRESENTS_WITH
ri_batch.append(record)   # RULES_IN
ro_batch.append(record)   # RULES_OUT  ← ALWAYS, for every row
```

Every row in `lr_table.csv` unconditionally creates all three relationships. This means every `(Symptom, Disease)` pair in Neo4j has BOTH a `RULES_IN` and a `RULES_OUT` edge.

### How the Contradiction Propagates

`neo4j_client.get_test_edges()` runs two separate Cypher queries, one for each relationship type:

```python
in_rows  = session.run("MATCH (s:Symptom {id: $hp_id})-[r:RULES_IN]->(d:Disease) ...")
out_rows = session.run("MATCH (s:Symptom {id: $hp_id})-[r:RULES_OUT]->(d:Disease) ...")
```

Since both edges exist for every pair, the same disease appears in both result sets. `DifferentialEngine.update()` then correctly detects:

```python
overlap = rules_in_keys & rules_out_keys
if overlap:
    logger.warning("disease(s) %s appear in both rules_in and rules_out ...")
```

And applies both LRs: net effect = `log(LR_in) + log(LR_out)` = `log(LR_in × LR_out)`. Since `LR_out = rules_out_lr` (a number < 1 for the normal case), the product partially cancels the positive evidence. For the reclassified case, both values are swapped so the cancellation is different but still wrong.

### The Intended Semantics vs Reality

**What the evaluator expects:** One edge per `(symptom, disease)` pair. `_build_prompt()` computes BOTH polarities from that single edge:

```python
# evidence_evaluator.py lines 161–169
for e in edges:
    if e["relationship"] == "RULES_IN":
        pos_in.append(...)           # positive result: boost this disease
        neg_out.append(1.0/lr, ...)  # negative result: penalise this disease
    else:  # RULES_OUT
        pos_out.append(...)          # positive result: penalise this disease
        neg_in.append(1.0/lr, ...)   # negative result: boost this disease
```

The evaluator already handles both polarities from a single edge. The `RULES_OUT` edge is therefore **redundant at evaluation time AND contradictory at evidence time**.

### Correct Fix

In `6_load_neo4j.py`, only populate `ri_batch` — never `ro_batch`. The `RULES_OUT` Cypher query and batch can be removed entirely. Every disease gets exactly one LR edge per symptom.

> **CAUTION:** This requires re-running `6_load_neo4j.py` and deleting existing `RULES_OUT` edges from Neo4j first. The `MERGE` pattern in the load script will not delete stale edges.

---

## 9. FINALIZATION: Is 65% Misleading?

### Current State

The API already returns:
```json
{
  "confidence":          0.654,
  "confidence_warning":  true,
  "finalization_reason": "max_questions"
}
```

All three fields are present. The data is there. The problem is **how the frontend renders it** — if it displays `65.4%` without surfacing `finalization_reason: "max_questions"`, the number appears clinically significant.

### Why 65% Is Not Meaningful Here

`0.654` means "65.4% probability within a closed universe of 2 diseases." In a 2-disease differential:
- The top disease will always be ≥ 50%
- After 10 questions all answered "no", the Bayesian math converges to a winner by attrition

If both diseases are clinically wrong, this number is pure mathematical artifact. It is **not** "65.4% probability this is the actual diagnosis."

### Recommendation

Add one computed boolean to `finalize_node()`:

```python
"diagnostic_confidence_threshold_reached": (
    reason == "confidence_gate" and not warn
)
```

This is `True` only when the session ended because the judge hit the genuine confidence criteria (≥75% top, ≤40% runner-up, ≥4 evidence turns, stable). It is `False` for `max_questions`, `no_tests`, `no_diseases_found`, and any run with `confidence_warning=True`.

The frontend can gate its "DIAGNOSIS CONFIRMED" display on this boolean rather than on the probability number.

---

## 10. RETRIEVAL BENCHMARK

Proposed: `backend/tests/retrieval/hpo_benchmark.py`

| Query | Expected HP ID | Concept |
|---|---|---|
| "I get winded going upstairs" | HP:0002875 | Exertional dyspnea |
| "Out of breath walking to the mailbox" | HP:0002875 | Exertional dyspnea |
| "Can't catch my breath when I walk fast" | HP:0002875 | Exertional dyspnea |
| "I'm out of breath when I walk" | HP:0002875 | Exertional dyspnea |
| "shortness of breath going upstairs" | HP:0002875 | Exertional dyspnea |
| "My heart feels like it's racing" | HP:0001962 | Palpitations |
| "The room keeps spinning" | HP:0002321 | Vertigo |
| "I'm tired all the time" | HP:0012378 | Fatigue |
| "my ankles are swollen" | HP:0001714 | Edema of ankles |
| "I've got purple spots on my legs" | HP:0000979 | Purpura |
| "I keep wheezing at night" | HP:0030828 | Wheezing |
| "I've been coughing for three weeks" | HP:0012735 | Cough |
| "chest pain when I breathe in" | HP:0100749 | Chest pain |
| "my skin turns yellow" | HP:0000952 | Jaundice |
| "feeling faint when I stand up" | HP:0031943 | Orthostatic hypotension |
| "I've got a rash on my arms" | HP:0000988 | Skin rash |
| "I can't sleep" | HP:0002360 | Sleep disturbance |
| "I've been having diarrhea for a week" | HP:0002014 | Diarrhea |
| "I lose my balance and fall over" | HP:0002359 | Frequent falls |
| "My vision goes blurry" | HP:0000505 | Visual impairment |

**Metrics per query:**
```
top1_correct     bool    expected HP == top-1 result
recall_at_5      bool    expected HP in top-5
recall_at_10     bool    expected HP in top-10
top1_score       float   cosine score of top-1 result
expected_score   float   cosine score of expected HP (None if not in top-10)
expected_rank    int     rank of expected HP (None if not found)
margin_1_2       float   score[0] - score[1]
```

**Aggregate targets (CI gates):**
- Top-1 accuracy ≥ 0.75
- Recall@5 ≥ 0.90
- MRR ≥ 0.80

---

## 11. FILE IMPACT

---

### `backend/knowledge/qdrant_client.py`
**FUNCTION:** `FRAMING_PATTERNS` / `_preprocess()`
**CURRENT:** 11 patterns, none covering "I got", "I had", "I noticed", "I keep", "I started"
**PROBLEM:** Past-tense and habitual conversational openers are not stripped; grammatical noise shifts the embedding away from the clinical concept
**PROPOSED CHANGE:**
```python
# Add to FRAMING_PATTERNS:
r"^i (got|get|had|have had)\s+",
r"^i noticed\s+",
r"^i (keep|kept) (getting|having|feeling)?\s*",
r"^i started (getting|having|feeling)?\s*",
r"^i('ve| have) started\s+",
r"^i('ve| have) been getting\s+",
```
**WHY:** Highest-leverage, zero-risk change. Strips noise before embedding. Does not require data rebuild.

---

### `backend/knowledge/qdrant_client.py`
**FUNCTION:** `search()`
**CURRENT:** Single binary threshold gate. No margin check.
**PROBLEM:** Flat score clusters (0.626/0.624) accepted same as decisive matches (0.90/0.61)
**PROPOSED CHANGE:**
```python
MARGIN_THRESHOLD = float(os.getenv("QDRANT_MARGIN_THRESHOLD", "0.04"))

# In search(), after computing ranked/top:
all_ranked = sorted(best.values(), key=lambda h: h["score"], reverse=True)
margin = all_ranked[0]["score"] - all_ranked[1]["score"] if len(all_ranked) > 1 else 1.0
ambiguous = margin < MARGIN_THRESHOLD

return {
    "clinical_term": top["name"],
    "symptom_id":    top["hp_id"],
    "symptom_ids":   [h["hp_id"] for h in all_ranked],
    "score":         top["score"],
    "margin":        margin,
    "ambiguous":     ambiguous,
    "matched":       True,
}
```
**WHY:** Separates "no match" from "uncertain match". Orchestrator can act differently on each.

---

### `backend/orchestrator/orchestrator.py`
**FUNCTION:** `seed_node()`
**CURRENT:** If `matched=True`, proceeds unconditionally to build differential
**PROBLEM:** Does not check `ambiguous=True`; a confused retrieval poisons the entire session
**PROPOSED CHANGE:**
```python
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
```
**WHY:** Prevents wrong-universe differential from being seeded. Prompts clarification instead.

---

### `backend/scripts/6_load_neo4j.py`
**FUNCTION:** `part_d_edges()`
**CURRENT:** Every row populates `pw_batch`, `ri_batch`, AND `ro_batch` unconditionally
**PROBLEM:** Creates both `RULES_IN` and `RULES_OUT` edges for every `(Symptom, Disease)` pair. Causes contradictory evidence on every Q&A turn.
**PROPOSED CHANGE:**
```python
pw_batch.append(record)
ri_batch.append(record)
# ro_batch.append(record)  ← REMOVE THIS LINE ENTIRELY
```
Also remove the `ro_query` execution and the RULES_OUT Cypher block. The evaluator already computes the negative polarity from the single RULES_IN edge via `1.0 / lr` inversion.
**WHY:** Eliminates the data authoring bug that causes the DifferentialEngine warning on every turn and corrupts LR arithmetic. Requires DB edge deletion + re-load.

---

### `backend/knowledge/neo4j_client.py`
**FUNCTION:** `get_initial_differential()`
**CURRENT:** Uses undirected `MATCH (d:Disease)-[r]-(s:Symptom)` (added as a workaround)
**PROBLEM:** After fixing the data loader, PRESENTS_WITH edges will exist reliably. The undirected query also picks up RULES_IN/RULES_OUT relationships in the seeding step, which is incorrect.
**PROPOSED CHANGE:** Restore:
```python
MATCH (d:Disease)-[r:PRESENTS_WITH]->(s:Symptom)
WHERE s.id IN $hp_ids
RETURN d.id AS disease_id, d.name AS name, d.prevalence AS prevalence,
       SUM(r.sensitivity) AS score
```
**WHY:** After Phase C (data fix), PRESENTS_WITH edges are correct. The undirected workaround is no longer needed and is semantically wrong for seeding.

---

### `backend/orchestrator/orchestrator.py`
**FUNCTION:** `finalize_node()`
**CURRENT:** Returns `confidence`, `confidence_warning`, `finalization_reason` separately
**PROBLEM:** Frontend must combine three fields to know whether confidence is real. The field name "confidence" alone implies certainty.
**PROPOSED CHANGE:**
```python
report = {
    ...existing fields...,
    "diagnostic_confidence_threshold_reached": (
        reason == "confidence_gate" and not warn
    ),
}
```
**WHY:** Single clear boolean for the frontend to gate "DIAGNOSIS CONFIRMED" display. No ambiguity in interpretation.

---

### NEW: `backend/tests/retrieval/hpo_benchmark.py`
**CURRENT:** Does not exist
**PROPOSED:** 20-query benchmark with top-1 accuracy, recall@5, recall@10, MRR, margin statistics. Runnable as a standalone script: `python hpo_benchmark.py [--verbose]`. CI-gated on top1_accuracy ≥ 0.75.

---

### NEW: `backend/data/processed/layperson_synonyms.json` + update `8_populate_qdrant.py`
**CURRENT:** Only HPO formal name + synonyms embedded
**PROPOSED (Phase D only):** Add supplementary Qdrant points for layperson phrases. Format:
```json
{
  "HP:0002875": ["winded going upstairs", "out of breath on exertion",
                  "can't catch my breath", "get breathless when I walk"],
  "HP:0001962": ["heart racing", "heart pounding", "heart fluttering"],
  ...
}
```
Each phrase gets its own Qdrant point with the same `hp_id` payload. IDs start at 20000 to avoid collisions.
**WHY:** Augments corpus without replacing the model. Directly fixes the layperson vocabulary gap.

---

## 12. IMPLEMENTATION ORDER

### Phase A — Preprocessing Fix (< 1 hour, no data rebuild, no restart required)
**Files:** `qdrant_client.py` (`FRAMING_PATTERNS`)
**Expected:** Correctly strips "I got/had/noticed/keep" → cleaner embedding input. MAY fix the winded case alone.
**Verify:** Run `9_verify_qdrant.py` + add "I got winded going upstairs" as a test case.

### Phase B — Benchmark (parallel with A)
**Files:** New `tests/retrieval/hpo_benchmark.py`
**Expected:** Scientific baseline. Run before Phase A to get current state, then after A to measure improvement.

### Phase C — Ambiguity / Margin Gate (< 2 hours, no data rebuild)
**Files:** `qdrant_client.py` (`search()`), `orchestrator.py` (`seed_node()`)
**Expected:** System asks for clarification when retrieval is ambiguous instead of committing to a flat cluster.
**Verify:** Inject mock scores 0.626/0.624 → ambiguous=True; 0.90/0.61 → ambiguous=False.

### Phase D — RULES_IN / RULES_OUT Data Bug (medium, requires DB rebuild)
**Files:** `6_load_neo4j.py`, `neo4j_client.py`
**Steps:** Remove `ro_batch`, delete RULES_OUT edges from Neo4j, re-run loader, re-run verifier.
**Verify:** DifferentialEngine warning never fires. `get_test_edges()` returns only RULES_IN edges.

### Phase E — Finalization Transparency (< 30 min)
**Files:** `orchestrator.py` (`finalize_node()`)
**Verify:** `diagnostic_confidence_threshold_reached=False` for max_questions, True only for genuine confidence-gate finalization.

### Phase F — Layperson Synonym Augmentation (larger)
**Files:** `8_populate_qdrant.py`, new `layperson_synonyms.json`
**Requires:** Benchmark data to identify which HPO concepts fail top-1. Only do after benchmark is built.

### Phase G — Dataset / Common Disease Bias (deferred)
Do not start until Phases A–D are complete and the retrieval benchmark passes reliably.

---

## 13. TESTS TO ADD AFTER EACH PHASE

### After Phase A:
```
tests/retrieval/test_framing_patterns.py
  "I got winded going upstairs"      → strips to "winded going upstairs"
  "I had chest pain last night"       → strips to "chest pain last night"
  "I keep wheezing at night"          → strips to "wheezing at night"
  "I started getting headaches"       → strips to "getting headaches"
  "I noticed my ankles were swollen"  → strips to "ankles were swollen"
  All existing patterns still pass
```

### After Phase C:
```
tests/retrieval/test_ambiguity_gate.py
  Mock scores 0.626/0.624 → search() returns ambiguous=True
  Mock scores 0.90/0.61   → search() returns ambiguous=False
  seed_node with ambiguous=True → returns finalization_reason="ambiguous_symptom"
  seed_node with ambiguous=False → proceeds to differential normally
```

### After Phase D:
```
tests/engines/test_no_lr_contradiction.py
  For any Q&A turn using real Neo4j, verify no disease appears in
  both rules_in and rules_out in the same evidence payload.

tests/knowledge/test_neo4j_single_lr_edge.py
  Spot-check 20 (disease, symptom) pairs from lr_table.csv.
  For each: verify RULES_IN exists, RULES_OUT does not exist.
```

### After Phase E:
```
tests/orchestrator/test_finalization_flags.py
  max_questions termination    → diagnostic_confidence_threshold_reached=False
  no_tests termination         → diagnostic_confidence_threshold_reached=False
  confidence_gate + warning    → diagnostic_confidence_threshold_reached=False
  confidence_gate + no warning → diagnostic_confidence_threshold_reached=True
```

### After Phase F (benchmark as CI gate):
```
tests/retrieval/hpo_benchmark.py
  20 queries, CI gates:
    top1_accuracy >= 0.75
    recall@5      >= 0.90
    MRR           >= 0.80
```

---

## SUMMARY TABLE

| Problem | Root Cause | File | Severity | Phase |
|---|---|---|---|---|
| "Winded" → Furuncle | PubMedBERT unaware of layperson vocab | qdrant_client.py | CRITICAL | A+F |
| "I got" not stripped | Missing framing patterns | qdrant_client.py `_preprocess()` | HIGH | A |
| 0.626 ≈ 0.90 in logic | No margin confidence gate | qdrant_client.py `search()` | HIGH | C |
| HP:0002875 rank unknown for this query | Layperson vocab absent from corpus | 8_populate_qdrant.py | HIGH | F |
| RULES_IN + RULES_OUT same disease | Both edge types always created in data loader | 6_load_neo4j.py | HIGH | D |
| 65% displayed as certainty | API lacks `diagnostic_confidence_threshold_reached` | orchestrator.py | MEDIUM | E |
| `get_initial_differential` undirected | Workaround for data bug; should revert post-D | neo4j_client.py | LOW | D |
| Dataset rare-disease bias | HPOA/OMIM source bias | Data pipeline | LOW | G (deferred) |
