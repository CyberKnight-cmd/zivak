# DifferentialEngine — Deep Dive Report

**File:** `engines/differential.py`
**Type:** Pure Python math — zero LLM calls, zero I/O
**Tests:** `engines/tests/test_differential.py`

---

## 1. Purpose

Maintains the live probability distribution over candidate diseases.
Every time the user answers a diagnostic question, the engine:
1. Applies Bayes' theorem to shift probabilities up or down
2. Re-normalises the distribution
3. Returns the updated, sorted list

This is the single source of truth for "what does the system currently believe?"

---

## 2. Constants

```python
RARE_DISEASE_FLOOR = 0.02   # 2% minimum prior — matches blueprint spec exactly
MIN_LR = 1e-10              # guards against log(0) if a malformed LR of 0 arrives
```

The `RARE_DISEASE_FLOOR` directly implements the **Rare Disease Seeder** from L6 of the
blueprint. A disease can never be mathematically eliminated from the differential;
it can only be pushed to ~0.02 / N (its renormalized floor fraction).

---

## 3. initialize() — Seeding the Differential

### Input
```python
diseases = [
    {"name": "COPD",         "specificity": 0.85, "prevalence": 0.065},
    {"name": "Asthma",       "specificity": 0.80, "prevalence": 0.080},
    {"name": "Heart Failure", "specificity": 0.90, "prevalence": 0.020},
    ...
]
```

### Algorithm (step by step)

```
Step 1: raw_score = specificity × prevalence
        COPD:          0.85 × 0.065 = 0.05525
        Asthma:        0.80 × 0.080 = 0.06400
        Heart Failure: 0.90 × 0.020 = 0.01800

Step 2: Normalise → probability = raw_score / sum(raw_scores)
        sum = 0.05525 + 0.06400 + 0.01800 = 0.13725
        COPD:          0.05525 / 0.13725 = 0.4025
        Asthma:        0.06400 / 0.13725 = 0.4664
        Heart Failure: 0.01800 / 0.13725 = 0.1311

Step 3: Apply rare-disease floor
        probability = max(probability, 0.02)
        All above 0.02 already → no change in this example

Step 4: Re-normalise (floor may have increased some values)
        sum = 0.4025 + 0.4664 + 0.1311 = 1.0  (unchanged if no floors triggered)

Step 5: log_prob = log(probability) for each disease
        Stores in log-space for subsequent updates

Step 6: Sort descending by probability
```

### What happens with 7 diseases (exertional dyspnoea)?

All 7 diseases get floor-checked. Interstitial Lung Disease has the lowest score:
`0.78 × 0.003 = 0.00234`. Normalised over all 7 it comes to ~2.6%, just above the floor.
Pulmonary Embolism: `0.75 × 0.005 = 0.00375` → ~4.2%. Both above floor naturally.

For vertigo (4 diseases), Central Vertigo: `0.90 × 0.004 = 0.0036` normalised to ~3.9%.
The floor rarely triggers with the current mock data — it's a safety net for future diseases
with very low prevalence.

### Mutation Warning
`initialize()` adds `raw_score`, `probability`, and `log_prob` keys **directly into the
input dicts**. `MockNeo4jClient.get_initial_differential()` uses `deepcopy()` so this is
safe with mock data. Real clients must also return fresh copies.

---

## 4. update() — Bayesian Evidence Update

### Core math

```
Bayes in log-space:
  log P(D|E) = log P(D) + log LR(E|D)

After all LRs applied:
  log P_normalised = log P_unnorm - log_sum_exp(all log P_unnorm)
```

### Input format

```python
evidence = {
    "rules_in":  [{"disease": "COPD",  "likelihood_ratio": 8.5}],
    "rules_out": [{"disease": "Asthma","likelihood_ratio": 0.15}],
}
```

`rules_in` entries have LR > 1 (evidence supports the disease).
`rules_out` entries have LR < 1 (evidence argues against the disease).

### Algorithm

```
Step 1: Build log_lr_map
        For each rule in rules_in + rules_out:
          key = disease.lower().strip()
          log_lr_map[key] += log(max(LR, 1e-10))

        Multiple rules for the same disease are summed in log space
        (equivalent to multiplying the LRs in probability space).

Step 2: Update each disease's log_prob
        For diseases NOT in log_lr_map: log_prob += 0.0  (no change, log(1.0))
        For diseases IN  log_lr_map:    log_prob += log_lr_map[key]

Step 3: Log-sum-exp normalisation (numerically stable)
        max_log   = max(all log_probs)
        log_total = max_log + log( sum( exp(log_prob - max_log) ) )
        For each disease: log_prob -= log_total
                          probability = exp(log_prob)

Step 4: Save evidence_history.append(deepcopy(evidence))
Step 5: Sort descending by probability
Step 6: Return updated differential
```

### Worked example: FEV1 result for COPD scenario

Starting state after seed (3-disease simplified):
- COPD: 40.2%, Asthma: 46.6%, HF: 13.1%

FEV1 positive (obstruction confirmed):
```
evidence = {
  rules_in:  [COPD LR=8.5, Asthma LR=3.0],
  rules_out: [HF LR=0.5,   PE LR=0.6]
}

log_lr_map:
  copd:          log(8.5)  =  2.140
  asthma:        log(3.0)  =  1.099
  heart failure: log(0.5)  = -0.693

Updated log_probs (before normalise):
  COPD:          log(0.402) + 2.140  = -0.912 + 2.140 = 1.228
  Asthma:        log(0.466) + 1.099  = -0.763 + 1.099 = 0.336
  Heart Failure: log(0.131) + (-0.693) = -2.031 - 0.693 = -2.724

After normalise:
  COPD:          exp(1.228) / total = 3.414 / ...
  COPD ends up ~70-75%, Asthma ~25%, HF ~1-2%
```

### Case-insensitive matching
Disease names are normalised with `.lower().strip()` before matching. This prevents
"COPD" vs "copd" mismatches between the LLM output and the differential list.

---

## 5. State Persistence & Reconstruction

The engine is **not** stored directly in LangGraph state. Instead, the state stores
two serialisable fields:

```python
state["differential"]     = engine.differential     # list of dicts with log_prob
state["evidence_history"] = engine.evidence_history # list of evidence dicts
```

Before each qa_node invocation, `_restore_engine()` reconstructs the engine:

```python
def _restore_engine(state):
    engine = DifferentialEngine()
    engine.differential    = state.get("differential", [])
    engine.evidence_history = state.get("evidence_history", [])
    return engine
```

This works because the differential list dicts contain `log_prob`, which is the engine's
working state. As long as LangGraph serialises the floats correctly (it does with
MemorySaver), the reconstruction is lossless.

**Risk:** If a Redis checkpointer is used and rounds floats during JSON serialisation,
`log_prob` values could drift by tiny amounts. Over 10 turns this is negligible but
worth testing.

---

## 6. Known Issues & Design Notes

### Issue 1: No floor enforcement during update()
`initialize()` applies the 2% floor. `update()` does NOT re-apply it. In theory, after
many negative LR updates, a disease could drop below 2%. This is intentional — the
floor is only a *prior* floor, not a hard floor on posteriors. But it means after
sufficient negative evidence, rare diseases can effectively reach 0.

### Issue 2: Evidence history grows unbounded
`evidence_history` stores a deepcopy of every evidence dict. For 10 turns this is ~10
small dicts — fine for MemorySaver. For a Redis checkpointer serialising thousands of
concurrent sessions, this may be worth compressing (store only the log_lr_map instead).

### Issue 3: No cross-check between rules_in and rules_out for same disease
If the LLM returns the same disease in both `rules_in` and `rules_out` (a bug or
adversarial input), both LRs would be applied. The net effect is `log(lr_in) + log(lr_out)`.
If lr_in = 8.5 and lr_out = 0.15, net = `log(8.5 × 0.15) = log(1.275)` — a slight
increase. This is unlikely in practice but not validated.

### Issue 4: Identical disease names in different symptom clusters
"Anemia" appears in both HP:0002875 and HP:0001962. The engine doesn't know about
symptom clusters — it just works on whatever differential it receives. This is fine
by design.

---

## 7. Test Coverage

| Test                 | What it validates                            | Status |
|----------------------|----------------------------------------------|--------|
| test_initialize      | Probabilities sum to 1.0, sorted correctly   | PASS   |
| test_update          | COPD rises above 80% after FEV1 evidence     | PASS   |

**Missing tests:**
- Floor triggering (disease with very low prevalence)
- Multiple sequential updates (compounding LRs)
- Edge case: single disease in differential
- Edge case: evidence for a disease not in the differential (should be a no-op — it is, but untested)
- Log-sum-exp numerical stability with extreme LRs
