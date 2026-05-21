# ConfidenceJudge — Deep Dive Report

**File:** `engines/confidence_judge.py`
**Type:** Pure Python logic — zero LLM calls, zero I/O
**Tests:** `engines/tests/test_confidence_judge.py`

---

## 1. Purpose

Acts as the **termination gate** for the diagnostic loop.
Called at the START of every `qa_node` invocation to decide:
> "Do we have enough certainty to stop asking questions and issue a final diagnosis?"

If `should_finalize()` returns True, the orchestrator routes to `finalize_node`.
If False, it selects another question and loops.

---

## 2. Configuration (constructor defaults)

```python
ConfidenceJudge(
    min_evidence      = 4,     # minimum Q&A turns before any finalisation
    min_top_confidence= 0.75,  # top disease must be at least 75% probable
    max_runner_up     = 0.40,  # second-place must be below 40%
)
```

These match the blueprint spec exactly:
> "≥4 evidence points, top diagnosis P ≥0.75, runner-up P <0.40"

---

## 3. should_finalize() — Full Decision Logic

### Signature
```python
def should_finalize(
    differential: List[Dict],   # current sorted differential
    evidence_count: int,         # number of Q&A turns completed so far
    previous_top_prob: float | None,  # top prob BEFORE the last answer was applied
) -> Tuple[bool, Dict]
```

### Decision Tree

```
1. Guard: len(differential) < 2?
   → return (False, "Insufficient diseases")

2. Stability gate: previous_top_prob is not None?
   delta = differential[0]["probability"] - previous_top_prob
   delta < -0.10?  (leader dropped >10 percentage points on last answer)
   → return (False, "Unstable — dropped X% on last answer")

3. Three criteria checked in parallel:
   a. min_evidence:   evidence_count >= 4
   b. top_confidence: differential[0]["probability"] >= 0.75
   c. runner_up_low:  differential[1]["probability"] < 0.40

4. ALL three met?
   → return (True,  "All criteria met")

5. At least one unmet?
   → return (False, human-readable reason for failure)
```

---

## 4. The Stability Gate — Detail

### What it does
Prevents premature finalization when the most recent answer has **destabilized**
the leader. If the top disease dropped by more than 10 percentage points from the
previous turn, something significant changed and more evidence is needed.

### Timing (this is subtle)

In `qa_node`, the snapshot is taken **before** the current answer is applied:

```python
pre_update_top = state["differential"][0]["probability"]  # before update
updated = engine.update(evidence)                          # apply this answer
return {
    "differential":      updated,
    "previous_top_prob": pre_update_top,   # saved to state
}
```

On the NEXT turn, `should_finalize()` receives:
- `differential` = post-update (after this answer)
- `previous_top_prob` = pre-update (before this answer)
- `delta` = post - pre = how much THIS answer moved the leader

So the gate says: "The answer we just processed caused a big drop — keep asking."

### Example

Turn 3: COPD at 65%, Asthma at 30%.
User answers "Peak flow variability 28%" — this strongly supports Asthma.
After update: COPD drops to 45%, Asthma rises to 50%.
`delta = 0.45 - 0.65 = -0.20` → -20% drop → stability gate fires.
Even though evidence_count = 3 (below min), the gate fires and provides a specific
reason: "Unstable — 'COPD' dropped 20.0% on last answer".

### Gap: First turn has no stability check
On the very first `submit_answer`, `previous_top_prob` in state is `None` (set in
`DiagnosticOrchestrator.start_session()`). So the gate is skipped for the first turn.
This is correct — there's no "previous" to compare to.

---

## 5. Reason Generation

`_get_reason()` priority order (first unmet criterion wins):

```
1. Not enough evidence → "Need more evidence (N/4)"
2. Runner-up too high  → "Runner-up diagnosis still too likely — need discriminating test"
3. Top too low         → "Top diagnosis confidence too low"
4. All met             → "All criteria met - ready to finalize"
```

Note: criterion 2 is checked before criterion 3 because a high runner-up is more
actionable (ask a discriminating test) than a low top (which might just mean more
questions needed in general).

---

## 6. Interaction with MAX_QUESTIONS Cap

The orchestrator has a hard cap: `MAX_QUESTIONS = 10`.

```python
# In qa_node — checked AFTER ConfidenceJudge
if len(state["questions_asked"]) >= MAX_QUESTIONS:
    return {"should_finalize": True, "judge_details": {"reason": "Max questions reached"}}
```

This means:
- ConfidenceJudge can still return False at turn 10, but the orchestrator overrides it
- The session finalizes with whatever the best probability is, even if thresholds aren't met
- `finalize_node` just takes `differential[0]` regardless of confidence score

**Risk:** A session could end with a diagnosis at only 55% confidence if no test
combination can push any disease above 75% within 10 questions.

---

## 7. Threshold Analysis for Current Mock Data

### Exertional Dyspnoea (7 diseases)

Starting probabilities are relatively flat (Asthma ~28%, COPD ~26%, HF ~14%, etc.)
To reach 75% for COPD: needs ~LR product of ~10-15x over baseline.
FEV1 (LR 8.5) + CXR hyperinflation (LR 4.2) = product ~35.7 → easily above 75%.
COPD scenario typically reaches threshold in 3-4 questions.

### Palpitations (5 diseases)

ECG for AF has LR 15.0. Starting AF prior: ~13%.
After positive ECG: dramatic jump. After positive Holter (LR 6.0): near-certain.
Typically finalizes in 2-3 questions.

### Vertigo (4 diseases)

Smaller differential → probabilities start higher (~33-40% for BPPV at top).
Positive Dix-Hallpike (LR 12.0) on BPPV starting at ~40% → rapid convergence.
Typically finalizes in 2-3 questions.

### Potential Finalization Failure

If the correct disease has few tests and they all have weak LRs, the system may hit
MAX_QUESTIONS without reaching 75%. The Anemia scenario is the most vulnerable —
only 2 tests (CBC and ferritin) with high LRs, but in a 7-disease differential,
the other diseases may not drop fast enough if only Anemia-specific tests are available.

---

## 8. Known Issues

### Issue 1: Runner-up check uses fixed 40% — not relative to top
The criterion `runner_up < 0.40` is an absolute threshold. If top = 76% and runner_up = 39%,
it passes. But if top = 51% and runner_up = 39%, it also passes (51% ≥ 75% fails first,
so this case is blocked by top_confidence). The two thresholds work together but a
relative gap check (e.g., top must be at least 2× runner_up) might be more clinically
meaningful.

### Issue 2: Only checks top 2 diseases
The judge checks `differential[0]` and `differential[1]`. If there are 5 diseases all
clustered at 18-22% each, the runner_up check (one at 20%) would fail, correctly
preventing finalization. This works, but the reason given would be "runner-up too high"
which is slightly misleading when all diseases are ambiguous.

### Issue 3: stability gate direction only
The gate only fires on a DROP of the leader (delta < -0.10). It does NOT fire if the
runner-up suddenly surged (e.g., leader stayed at 60%, runner-up jumped from 10% to 42%).
However, the `runner_up_low` criterion (< 40%) would catch this case — a 42% runner-up
fails that check. So combined, the two mechanisms cover both destabilization scenarios.

---

## 9. Test Coverage

| Test                      | What it validates                              | Status |
|---------------------------|------------------------------------------------|--------|
| test_early_stage          | No finalization with evidence_count=1          | PASS   |
| test_high_confidence      | Finalizes when all 3 criteria met              | PASS   |
| test_ambiguous_differential | No finalization when runner-up = 45%         | PASS   |

**Missing tests:**
- Stability gate (delta < -0.10 prevents finalization)
- MAX_QUESTIONS boundary (10 questions, criteria not met)
- Single disease in differential (guard check)
- Exactly at thresholds (evidence=4, top=0.75, runner_up=0.399 — should finalize)
- previous_top_prob = None (first turn)
