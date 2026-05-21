# Known Issues & Planned Changes — Master Issue Register

**Last updated:** After analysing run_scenarios.py output (11/15 = 73% accuracy)
and verifying all previously reported flaws against the current codebase.

---

## STATUS OF PREVIOUSLY REPORTED FLAWS

The table below audits the 10 flaws from the prior analysis against the code that
is currently committed. Each was verified by reading the actual source files.

| # | Component          | Flaw                                       | Current Status |
|---|--------------------|--------------------------------------------|----------------|
| 1 | EvidenceEvaluator  | Negative LRs applied in wrong direction    | ✅ FIXED        |
| 2 | EvidenceEvaluator  | Incomplete edge application per result     | ✅ FIXED        |
| 3 | run_scenarios.py   | Fallback answers were disease-positive     | ✅ FIXED        |
| 4 | QuestionSelector   | No test_id validation after LLM call       | ✅ FIXED        |
| 5 | DifferentialEngine | rules_in/rules_out distinction unused      | ⚠️ SEE NOTE     |
| 6 | Orchestrator       | Available tests from top-3 only            | ✅ FIXED (→ 5)  |
| 7 | EvidenceEvaluator  | temperature=1 on structured output task    | ✅ FIXED (→ 0)  |
| 8 | EvidenceEvaluator  | reasoning_effort=low on hardest task       | ✅ FIXED (→ med)|
| 9 | ConfidenceJudge    | 4-question minimum too low                 | ⚠️ STILL ACTIVE |
|10 | QuestionSelector   | No symptom/history context in prompt       | ⚠️ STILL ACTIVE |

**Note on #5:** The engine merges `rules_in` and `rules_out` into a single loop and
applies `log(LR)` uniformly — the list membership is not used. This is **not a bug**
in the current design because the pre-computed LRs in `rules_in` are always > 1
(boosting) and in `rules_out` are always < 1 (penalising). The math is correct.
The risk is that there is no validation enforcing this invariant — if the LLM
copies a high LR into `rules_out` by mistake, the engine boosts when it should
penalise. `_validate_output()` checks disease whitelists and LR clamping [0.001, 100]
but NOT the sign/direction invariant. This is a latent gap, not an active bug.

---

## CONFIRMED FROM RUN_SCENARIOS OUTPUT

The following are proven by observing the actual Q&A traces.

### CONFIRMED-1: C1 Double-Execution Causes Repeated Tests (CRITICAL)

The C1 bug (LangGraph re-executes `qa_node` from the top on resume) is not just a
theoretical performance cost — the run output proves it is **corrupting sessions**:

| Scenario        | Repeated test     | Times | Turns       |
|-----------------|-------------------|-------|-------------|
| Asthma          | test_peak_flow    | 2×    | 2 and 4     |
| ILD             | test_crp          | 2×    | 7 and 8     |
| Hyperthyroidism | test_echo_pal     | 3×    | 4, 5, and 6 |

**Why repeats happen (exact mechanism):**

```
Turn N, first execution:
  asked_ids = {tests from turns 1..N-1}
  selector picks test_A → interrupt(test_A) → graph pauses

User submits answer. Re-execution:
  asked_ids = {tests from turns 1..N-1}   ← test_A NOT YET in asked_ids
  selector picks test_B (temp=1, different from test_A)
  interrupt(test_B) → returns the user's answer immediately
  evaluator evaluates the answer in test_B's context  ← WRONG CONTEXT
  questions_asked += [test_B]   ← test_A never recorded

Turn N+1:
  asked_ids = {tests 1..N-1} + {test_B}
  test_A is NOT in asked_ids → offered again → selected again
```

**What this actually causes beyond the repeat:**
1. User sees question for test_A; internally test_B's evidence is applied → wrong Bayesian update
2. test_A is never added to `questions_asked` → re-offered on next turn
3. Wasted question slots: Hyperthyroidism used 3 of 8 slots repeating echo_pal
4. ILD used a wasted crp slot instead of asking test_hrct (LR=20.0)

**Fix:** Split `qa_node` into `question_node` + `answer_node`. Only `question_node`
interrupts; `answer_node` runs exactly once per answer with no re-execution risk.

---

### CONFIRMED-2: Key High-LR Tests Never Asked in 3 Failure Cases (CRITICAL)

All 3 dyspnoea failures (Pneumonia, Anemia, ILD) share the same pattern: the
highest-value discriminating test for the correct disease was **never asked** despite
having 7-9 question turns.

| Scenario  | Questions used | Key tests never asked             | Max LR available |
|-----------|----------------|-----------------------------------|------------------|
| Pneumonia | 8              | test_cxr_inf (LR 5), test_sputum (LR 8) | 8.0         |
| Anemia    | 7              | test_cbc (LR 15), test_ferritin (LR 10) | 15.0        |
| ILD       | 9              | test_hrct (LR 20!), test_pft (LR 8)     | 20.0        |

test_hrct has the **highest LR in the entire mock knowledge base (20.0)** and was
never asked in 9 turns. This is a selector quality problem — the selector keeps
choosing tests for top-ranked diseases (COPD, Asthma, HF) rather than asking the
one test that would definitively confirm or deny the lower-ranked correct disease.

**Root cause:** The selector prompt does not include LR magnitudes. test_hrct (LR=20)
looks identical to test_cxr_hyp (LR=4.2) from the selector's perspective — both are
just listed as "relevant for [disease]". Without knowing that HRCT is 5× more
informative than a CXR, the LLM naturally picks tests for the diseases it currently
believes are most likely (COPD, Asthma at top of differential).

---

### CONFIRMED-3: COPD Rises by Exclusion — Prior Dominance Effect (HIGH)

All 3 dyspnoea failures end with COPD at 83–93% despite no COPD-positive evidence.

**Why COPD accumulates probability without any positive test:**

When cardiac tests come back negative, their LR inversions **boost** COPD:
- Normal BNP (negative test_bnp): `neg_in` includes COPD with LR = 1/0.5 = **2.0** — COPD boosted!
- Normal Echo (negative test_echo): `neg_in` includes COPD with LR = 1/0.2 = **5.0** — COPD boosted!
- Normal D-dimer (negative test_ddimer): `neg_in` includes COPD with LR = 1/0.5 = **2.0** — COPD boosted!

This is medically correct (ruling out HF and PE does make obstructive disease more
likely). But the consequence is: COPD rises to 80%+ purely by elimination, and the
system never has to ask a single COPD-positive test to "confirm" it.

For Anemia and Pneumonia to win, the system must ask their specific confirmatory
tests (CBC/ferritin, CXR-infiltrates/sputum). If those tests are never selected,
COPD wins by default.

**This is a fundamental architectural tension:** entropy-based question selection
naturally favours tests that discriminate the TOP diseases. But confirming a
lower-ranked disease requires proactively asking its specific tests even when they
seem less discriminating given the current differential.

---

### CONFIRMED-4: SVT and Hyperthyroidism Finalize Below Confidence Threshold (MEDIUM)

| Scenario        | Confidence | Runner-up | Reason for stopping    |
|-----------------|------------|-----------|------------------------|
| SVT             | 69.8%      | AF 28.1%  | No tests available     |
| Hyperthyroidism | 50.8%      | SVT 28.9% | No tests available     |

Both sessions exhausted all available tests without meeting the 75% confidence gate.
They terminated via the "No tests available" code path. The correct diagnosis was
still #1, so run_scenarios.py marks them ✓ — but a real system should flag these
as **insufficient evidence** rather than issuing a confident diagnosis.

SVT's AF runner-up at 28.1% is particularly dangerous — ECG showed narrow-complex
SVT but Holter + multiple other tests still left AF at 28%. In clinical practice
this ambiguity would require additional workup.

The Hyperthyroidism case was severely damaged by the echo_pal repeat bug (3 wasted
turns), which consumed slots that could have better discriminated SVT vs Hyper.

---

### CONFIRMED-5: Meniere's Failure is a Knowledge-Base Consistency Bug (MEDIUM)

Meniere's Disease fails because `test_tymp` penalises it:

**LR table says:** Meniere RULES_IN LR=4.0 for test_tymp (positive tympanometry → Meniere).  
**run_scenarios.py answer:** "Type A bilaterally — middle ear normal"  
**LLM correctly classifies:** NEGATIVE (Type A = normal = negative finding)  
**Engine applies:** LR = 1/4.0 = 0.25 → Meniere penalised to 13.3%

**The clinical reality:** Meniere's disease is an **inner ear** problem (endolymphatic
hydrops). Tympanometry tests the **middle ear**. Type A (normal middle ear compliance)
IS the expected finding in Meniere's — the middle ear is unaffected. A positive
tympanogram (Type B = flat, Type C = negative pressure) would indicate middle ear
pathology, which is NOT Meniere's.

The current setup has it backwards: the answer phrasing says "Type A = normal" (which
the LLM correctly reads as NEGATIVE), but the system needs it to be POSITIVE to
confirm Meniere. Either:
- Change the run_scenarios answer to "Abnormal pattern consistent with endolymphatic hydrops"
  (to make it genuinely positive for tympanometry in Meniere)
- OR remove tympanometry from the Meniere test list (it's not a standard discriminating test)
- OR redefine what "positive" means for test_tymp in the LR table

---

## ACTIVE ISSUES (existing + new)

### C1: Double LLM Call — qa_node Re-Executes on Resume

**Status:** CONFIRMED ACTIVE. Proven by repeated tests in run output.  
**Files:** `orchestrator/orchestrator.py` → `qa_node()`  
**Fix:** Split into `question_node` (interrupts) + `answer_node` (evaluates).

---

### C2: EvidenceEvaluator Polarity Confusion

**Status:** ACTIVE. No changes to the prompt guard against it.  
**Files:** `agents/evidence_evaluator.py` → `_build_prompt()`

The LLM determines positive/negative from free text. High-risk answers in the
current scenario set (which may cause wrong polarity classification):
- "FEV1/FVC 0.88 — normal ratio, **reduced volumes (restrictive)**" — the "reduced volumes"
  qualifier describes restriction (ILD's finding) but the test is FEV1/FVC which should be
  classified based on the ratio (0.88 = NEGATIVE for obstruction). The LLM may be confused
  by the "restrictive" hint.
- "Persistent sinus tachycardia; **no AF or SVT**" — contains disease names, risk of
  positive classification for Holter.
- "D-dimer 1.1 — mildly elevated, **inflammatory**" — mildly elevated D-dimer is
  technically POSITIVE (>0.5) but the comment "inflammatory" implies the patient/clinician
  thinks it's not PE-related.

**Fix:** Include a per-test reference threshold in the prompt ("For test_fev1: result is
POSITIVE if FEV1/FVC < 0.70, NEGATIVE if ≥ 0.70"). This eliminates ambiguity for
quantitative tests and reduces the LLM's interpretive burden.

---

### C3: SessionNotFoundError Never Raised

**Status:** ACTIVE.  
**Files:** `orchestrator/orchestrator.py` → `submit_answer()`  
**Impact:** Dead sessions return 503 instead of 404.  
**Fix:** Check `graph.get_state()` before invoking with Command(resume=...).

---

### N1: QuestionSelector Prompt Has No LR Magnitude Information (NEW — HIGH)

**Status:** ACTIVE. Directly causes the CONFIRMED-2 failure pattern.  
**Files:** `agents/question_selector.py` → `_build_prompt()`

The selector prompt lists tests as:
```
- HRCT chest (ID: test_hrct): relevant for COPD, Interstitial Lung Disease
- Chest X-ray hyperinflation (ID: test_cxr_hyp): relevant for COPD
```

The selector cannot distinguish test_hrct (LR=20 for ILD) from test_cxr_hyp (LR=4.2
for COPD). Without knowing LR magnitudes, it makes qualitative judgements that
naturally favour tests for top-ranked diseases.

**Fix:** Add LR magnitude hints to the test list in the prompt:

```
- HRCT chest (ID: test_hrct): relevant for COPD (moderate), ILD (very strong LR)
- CXR hyperinflation (ID: test_cxr_hyp): relevant for COPD (moderate LR)
```

Or include numeric LRs directly:
```
- HRCT chest (ID: test_hrct): ILD LR≈20, COPD LR≈3.5
```

This directly enables the selector to prioritise high-information tests like HRCT.

---

### N2: No Proactive Champion-Test Mechanism for Low-Prior Diseases (NEW — HIGH)

**Status:** ACTIVE. Architectural gap causing 3/4 dyspnoea failures.

The selector is incentivised to ask tests that discriminate between the **top**
diseases (COPD vs Asthma vs HF). This is rational entropy-reduction. But for diseases
like Anemia (starts ~10%) or ILD (starts ~2%), no entropy-reduction argument will
favour their tests until they rise — but they can only rise if their specific tests
are asked. This is a chicken-and-egg problem.

**Consequence:** In 7 turns, neither test_cbc nor test_ferritin was asked for Anemia.
In 9 turns, neither test_hrct nor test_pft was asked for ILD. The selector kept
choosing tests for COPD/Asthma/HF because those diseases dominate the differential.

**Fix options:**

A. **Mandate coverage**: Each disease in the differential must have at least one of its
   highest-LR tests asked within the first N turns. If not, the orchestrator forces it
   into the available list as a priority.

B. **Diversity-weighted selection**: Instead of pure entropy reduction over top diseases,
   add a term that rewards tests which would produce HIGH LR changes for ANY disease in
   the differential, even low-ranked ones.

C. **Coverage tracking in selector prompt**: Include "diseases not yet tested" and
   "highest-LR untested test for each disease" in the selector prompt so the LLM
   can make globally informed decisions.

---

### N3: Meniere's Tympanometry LR Inconsistency (NEW — MEDIUM)

**Status:** ACTIVE. Directly caused the Meniere's failure.  
**Files:** `orchestrator/mock_clients.py` → `_LIKELIHOOD_RATIOS["test_tymp"]`

The `test_tymp` RULES_IN LR=4.0 implies a positive (abnormal) tympanogram confirms
Meniere's. But the run_scenarios.py answer is "Type A bilaterally — middle ear normal"
which is clinically the NORMAL result in Meniere's (inner ear disorder, not middle ear).
The LLM correctly classifies Type A as NEGATIVE, which penalises Meniere via LR inversion.

**Fix:** Remove `test_tymp` from Meniere's test list, or change the positive finding
definition to an abnormal tympanogram type, or provide a more explicit answer like
"Type A normal — middle ear unaffected, consistent with inner ear disorder."

---

### N4: Selector Has No Access to Q&A History or Symptom (Flaw #10 — STILL ACTIVE)

**Status:** ACTIVE.  
**Files:** `agents/question_selector.py` → `_build_prompt()`

The selector prompt only shows the current differential and available tests. It has
no knowledge of:
- What the original symptom was ("smoked for 30 years" → strongly suggests COPD)
- What questions have already been asked and what the answers were
- Whether tests have already produced discriminating evidence

Without this context, the selector asks from scratch each turn. It may select a test
whose clinical purpose was already addressed by a previous answer, or miss a test
that would be obviously important given the history (e.g., after confirming normal
FEV1, the selector doesn't know spirometry already ruled out obstruction).

**Fix:** Add symptom and Q&A history summary to the selector prompt:

```
Patient chief complaint: "dry persistent cough for 6 months, exertional dyspnoea"

Evidence collected so far:
  Turn 1 — FEV1/FVC spirometry: Normal ratio (0.88), reduced volumes → rules out obstruction
  Turn 2 — BNP: Normal → rules out heart failure
  Turn 3 — D-dimer: Normal → rules out PE

Remaining questions: which test now best discriminates ILD vs COPD vs Anemia?
```

---

### M1: Available Tests From Top-5 Only (MEDIUM — partially mitigated)

**Status:** PARTIALLY MITIGATED. Changed from top-3 to top-5.  
**Remaining gap:** In the 7-disease dyspnoea differential, diseases at rank 6-7
(PE and ILD) don't contribute tests until they rise. For ILD starting at 2%, it
may never reach rank 5 naturally because it needs test_hrct to confirm it — but
test_hrct only enters the pool if ILD is in rank 5.

**Fix:** Use probability-based cutoff (all diseases above X% contribute tests) or
include all diseases unconditionally (only 7-16 diseases per cluster).

---

### M2: Anemia Has No Cross-Links to Other Tests (MEDIUM)

**Status:** ACTIVE.  
**Files:** `orchestrator/mock_clients.py` → `_LIKELIHOOD_RATIOS`

test_cbc RULES_IN only Anemia. A positive CBC (confirming Anemia) does nothing to
reduce COPD, Asthma, HF, etc. — they all stay at their current probabilities.
Adding RULES_OUT edges (positive CBC argues against respiratory/cardiac disease)
would make the differential collapse faster.

---

### M3: Hyperthyroidism Has Only 1 Test and Weak Cross-Links (MEDIUM)

**Status:** ACTIVE. Hyperthyroidism reached only 50.8% confidence.  
test_tsh is the only test for Hyperthyroidism. ECG and Holter results (sinus
tachycardia, no AF) don't actively RULES_IN Hyperthyroidism — they only RULES_OUT
AF and Anxiety. Hyperthyroidism rises only by exclusion, like COPD in the dyspnoea
cluster.

---

### M4: Finalization Reason Not Surfaced in API Response (MEDIUM)

**Status:** ACTIVE.  
**Impact confirmed:** SVT (69.8%) and Hyperthyroidism (50.8%) issued diagnoses below
the stated 75% threshold because they terminated via "No tests available", not via
the confidence gate. The API response looks identical to a high-confidence diagnosis.

**Fix:** Add `finalization_reason: "confidence_gate" | "no_tests" | "max_questions"`
and `confidence_warning: bool` to the final diagnosis response.

---

### M5: ConfidenceJudge 4-Question Minimum Still Low (Flaw #9 — STILL ACTIVE)

**Status:** ACTIVE but lower impact than before (fallback answers now fixed).  
With fixed fallbacks (all normal/negative), a correct disease needs its specific
tests asked and answered. 4 questions can still be too few if the C1 bug wastes
slots or if the selector makes poor choices.

---

### H1: ExplainabilityAgent Not Implemented (HIGH)

`finalize_node()` returns question text and selector reasoning. No clinical
interpretation of why the diagnosis was reached. Blueprint requires: "Ruled out X
because [evidence]. Confirmed Y because [evidence]."

---

### H2 / H3 / H4: Real DB, PostgreSQL, Redis (HIGH — deferred)

Empty stubs for real Neo4j and Qdrant. No session persistence. No Redis cache.

---

## PRIORITY MATRIX (updated after run analysis)

| ID          | Issue                                    | Severity  | Effort  | Do First? |
|-------------|------------------------------------------|-----------|---------|-----------|
| C1          | Split qa_node (double execution)         | CRITICAL  | MEDIUM  | YES #1    |
| N1          | Add LR magnitudes to selector prompt     | HIGH      | LOW     | YES #2    |
| N2          | Champion-test mechanism for low priors   | HIGH      | MEDIUM  | YES #3    |
| C2          | Per-test threshold in evaluator prompt   | CRITICAL  | LOW     | YES #4    |
| N4 (#10)    | Add symptom + history to selector prompt | HIGH      | LOW     | YES #5    |
| N3          | Fix Meniere tympanometry LR              | MEDIUM    | TRIVIAL | YES #6    |
| M4          | finalization_reason in API response      | MEDIUM    | LOW     | NEXT      |
| M1          | Test pool: all diseases or prob-cutoff   | MEDIUM    | LOW     | NEXT      |
| M2          | Anemia cross-links                       | MEDIUM    | LOW     | NEXT      |
| M3          | Hyperthyroidism test coverage            | MEDIUM    | LOW     | NEXT      |
| M5 (#9)     | Raise min_evidence above 4              | LOW       | TRIVIAL | NEXT      |
| C3          | SessionNotFoundError in submit_answer    | HIGH      | LOW     | NEXT      |
| H1          | ExplainabilityAgent                      | HIGH      | HIGH    | LATER     |
| H2/H3/H4   | Real DBs, PostgreSQL, Redis              | HIGH      | HIGH    | LATER     |

---

## ACCURACY ESTIMATE AFTER FIXES

| Fix set applied                           | Estimated accuracy |
|-------------------------------------------|--------------------|
| Current (C1 bug active, selector blind)   | 73% (11/15)        |
| Fix C1 only (no more repeated tests)      | ~80% (12/15)       |
| Fix C1 + N1 (LR hints in selector)        | ~87% (13/15)       |
| Fix C1 + N1 + N2 (champion-test)          | ~93% (14/15)       |
| Fix C1 + N1 + N2 + C2 + N3 + N4          | ~100% (15/15)      |

The single highest-leverage fix is **C1 (split qa_node)** because it eliminates
wasted question slots AND ensures correct test/answer pairing for evidence evaluation.
The second-highest leverage is **N1 (LR hints)** because it directly fixes the
"key tests never asked" pattern responsible for 3 of the 4 failures.
