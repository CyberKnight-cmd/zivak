# run_scenarios.py — Deep Dive Report

**File:** `run_scenarios.py`
**Purpose:** Automated end-to-end regression test — runs all 15 disease scenarios
against the full diagnostic pipeline and measures accuracy.

---

## 1. Purpose & Design

`run_scenarios.py` is the primary **accuracy benchmark** for the system.
It drives a full session for each of 15 diseases, feeding pre-written "correct"
answers to every diagnostic question, and reports whether the system converges
on the expected diagnosis.

It uses the SAME `DiagnosticOrchestrator` that the API uses — no mocking of any
component except the DB clients (which are always mock here). Every LLM call
goes to the real model (Ollama or Groq).

---

## 2. Architecture

```
parse_args()
    │
    ▼
MockQdrantClient() + MockNeo4jClient() + DiagnosticOrchestrator()
    │         (initialized once, shared across all scenarios)
    ▼
for each scenario:
    run_scenario(orch, scenario, turn_delay)
        │
        ├── orch.start_session(symptom)
        │       → session_id, result (first question)
        │
        └── while question:
                test_id = question["test_id"]
                answer  = _get_answer(scenario, test_id)  ← ANSWER LOOKUP
                result  = orch.submit_answer(session_id, answer)
                break if not should_continue or final_diagnosis
        │
        ▼
    return summary dict
        │
        ▼
print_report(results, verbose)
```

---

## 3. Answer Resolution — _get_answer()

```python
def _get_answer(scenario, test_id):
    return (
        scenario.get("answers", {}).get(test_id)   # 1st: scenario-specific answer
        or FALLBACK_ANSWERS.get(test_id)             # 2nd: generic normal answer
        or DEFAULT_ANSWER                            # 3rd: "Normal / negative"
    )
```

### Priority 1: Scenario-specific answers

Each scenario defines answers for the key discriminating tests:

```python
{
    "name": "COPD",
    "symptom": "I get winded very easily, especially going upstairs — I smoked for 30 years",
    "expected": "COPD",
    "answers": {
        "test_fev1":      "FEV1/FVC 0.55 post-BD; fixed obstruction",
        "test_bronch":    "FEV1 +3% — no significant reversibility",
        "test_peak_flow": "<10% diurnal variability — not asthma",
        "test_cxr_hyp":   "Hyperinflated, barrel chest, flattened diaphragms",
        "test_hrct":      "Centrilobular emphysema + bullae, upper lobes",
        "test_pft":       "Obstruction, air trapping: RV 175%, DLCO 60%",
    },
}
```

These answers are **disease-targeted** — they contain the specific clinical findings
that the EvidenceEvaluatorAgent should classify as positive results for COPD.

### Priority 2: FALLBACK_ANSWERS — 28 tests, all NORMAL/NEGATIVE

```python
FALLBACK_ANSWERS = {
    "test_fev1":      "FEV1/FVC 0.79 — normal spirometry",
    "test_bronch":    "FEV1 +3% — no significant reversibility",
    "test_peak_flow": "Diurnal variability <8% — not consistent with asthma",
    "test_cxr_hyp":   "Normal CXR — no hyperinflation or cardiomegaly",
    "test_bnp":       "BNP 42 pg/mL — normal",
    ...
}
```

All fallbacks return NORMAL/NEGATIVE results. The design rationale:
> "A test asked in the wrong scenario should not inject false evidence."

For example, if the Asthma scenario is asked `test_bnp` (a Heart Failure test),
returning "BNP normal" correctly rules out Heart Failure without providing false
positive evidence for anything.

### Priority 3: DEFAULT_ANSWER = "Normal / negative"

Any test not in either lookup returns a generic normal answer. This is a safety
net for future tests added to the mock without corresponding fallbacks.

---

## 4. The 15 Scenarios — Complete List

### Exertional Dyspnoea (7 scenarios)

| Scenario   | Key symptom in description              | Discriminating tests         |
|------------|-----------------------------------------|------------------------------|
| COPD       | "smoked for 30 years"                   | FEV1 obstruction, HRCT       |
| Asthma     | "worse at night and in cold air"        | FEV1 reversible, peak flow   |
| Heart Failure | "swollen ankles, waking up gasping" | BNP elevated, ECHO reduced EF|
| Pulmonary Embolism | "left calf red and swollen"    | D-dimer high, CTPA confirms  |
| Pneumonia  | "fever, chills, productive cough"       | CXR consolidation, CRP       |
| Anemia     | "very pale and exhausted"               | CBC low Hb, ferritin low     |
| ILD        | "dry persistent cough for six months"   | HRCT honeycombing, PFT       |

### Palpitations (4 scenarios)

| Scenario         | Key symptom                          | Discriminating tests     |
|------------------|--------------------------------------|--------------------------|
| Atrial Fibrill.  | "beating irregularly"                | ECG irregularly irregular|
| SVT              | "starts and stops abruptly"          | ECG narrow complex, K+   |
| Anxiety          | "very anxious all the time"          | GAD-7 severe, ECG sinus  |
| Hyperthyroidism  | "rapid weight loss, feeling very hot" | TSH suppressed           |

### Vertigo (4 scenarios)

| Scenario          | Key symptom                          | Discriminating tests      |
|-------------------|--------------------------------------|---------------------------|
| BPPV              | "when I roll over in bed"            | Dix-Hallpike positive     |
| Vestibular Neuritis| "started suddenly after a cold"     | HIT positive, VNG         |
| Meniere's Disease | "ringing in my left ear"             | Audiometry low-freq SNHL  |
| Central Vertigo   | "double vision, difficulty walking"  | MRI posterior fossa mass  |

---

## 5. Answer Quality Analysis

### Well-designed answers (clear polarity signal)

These answers are unambiguous — the EvidenceEvaluator should always classify them
correctly:

```
"FEV1/FVC 0.55 post-BD; fixed obstruction"   → POSITIVE for obstruction
"BNP 1200 pg/mL — markedly elevated"          → POSITIVE for HF
"CT PA: Bilateral saddle PE confirmed"         → POSITIVE for PE
"MRI: 1.9 cm posterior fossa mass"            → POSITIVE for Central Vertigo
"GAD-7 19/21 — severe GAD"                    → POSITIVE for Anxiety
"TSH <0.01, fT4 38 — overt hyperthyroidism"   → POSITIVE for Hyperthyroidism
```

### Potentially ambiguous answers (risk of polarity confusion)

```
"FEV1 +3% — no significant reversibility"
```
This is a NEGATIVE bronchodilator response (not reversible). But "no significant
reversibility" is a double negative. The LLM must understand that "+3%" is below
the 12% threshold for positive reversibility, and "no significant" means NEGATIVE.
Risk: LLM focuses on "+3%" and classifies as positive.

```
"BNP 95 — normal, no heart failure"
```
This is clearly negative, but appears in the PE scenario where `test_bnp` is asked
in a different context. Fine.

```
"D-dimer 1.1 — mildly elevated, inflammatory"
```
This appears in the Pneumonia scenario. D-dimer of 1.1 is above normal (<0.5) but
the comment says "inflammatory" not PE-related. The evaluator should classify this
as a POSITIVE D-dimer (> 0.5) which would RULES_IN PE slightly. This is medically
correct but might be unexpected — a Pneumonia scenario providing mild PE evidence.

```
"Positive right Dix-Hallpike: upbeat-torsional nystagmus, fatigues"
```
Clearly POSITIVE. "Fatigues" is a key BPPV feature. This is well-written.

```
"3 self-terminating SVT episodes; longest 4 min"
```
This is the Holter answer for SVT. The evaluator needs to understand that "SVT
episodes" means the Holter was POSITIVE for SVT. Good answer, clear polarity.

### The Hyperthyroidism SVT scenario answers

```python
"test_holter": "Persistent sinus tachycardia; no AF or SVT"
```
This is a NEGATIVE Holter for arrhythmia (no AF, no SVT captured). But the test_holter
RULES_IN both AF and SVT. So a negative result should RULES_OUT AF and SVT via
LR inversion. The evaluator needs to correctly classify "no AF or SVT" as NEGATIVE
for the Holter test.

Risk: LLM might see "AF" and "SVT" in the text and classify as positive.

---

## 6. Execution Flow & Timing

```python
def main():
    args = parse_args()
    # ...
    for i, scenario in enumerate(scenarios, 1):
        if i > 1 and args.delay > 0:
            _pause(args.delay, f"rate-limit gap before scenario {i}")
        run_scenario(orch, scenario, turn_delay=args.turn_delay)
```

### Default delays

| Parameter    | Default | Purpose                        | With Ollama |
|--------------|---------|--------------------------------|-------------|
| `--delay`    | 60s     | Between scenarios (Groq limits)| Use 0       |
| `--turn-delay`| 0s     | Between Q&A turns              | Keep 0      |

**With Ollama, always run: `python run_scenarios.py --delay 0`**

15 scenarios × 60s default delay = 14 minutes of just sleeping.
With `--delay 0` and Ollama: total time ≈ 15 scenarios × ~5-10 LLM calls × ~3s per call
= roughly 3-8 minutes total depending on Qwen2.5:14b speed.

### CLI options

```bash
# All scenarios, no delay
python run_scenarios.py --delay 0

# Single scenario with full trace
python run_scenarios.py --filter COPD --delay 0 --verbose

# Multiple scenarios filtered
python run_scenarios.py --filter COPD Asthma BPPV --delay 0

# Add turn delay (useful if throttling Ollama)
python run_scenarios.py --delay 0 --turn-delay 2
```

Filter matching is **case-insensitive substring**:
```
--filter copd     → matches "COPD"
--filter al       → matches "Atrial Fibrillation", "Central Vertigo"
--filter ver      → matches "Central Vertigo"
```

---

## 7. Output Format

### Non-verbose (default)

```
======================================================================
  ZIVAK — Multi-Disease Session Report
======================================================================

  Scenario                   Expected                   Diagnosed                   Conf    Qs  OK?
  ----------------------------------------------------------------------
  COPD                       COPD                       COPD                       92.3%    4   ✓
  Asthma                     Asthma                     Asthma                     88.1%    3   ✓
  Heart Failure              Heart Failure              Heart Failure              95.2%    3   ✓
  ...

----------------------------------------------------------------------
  Result: 14/15 correct (93%)
======================================================================
```

### Verbose (-v)

Each scenario gets full Q&A trace:
```
[✓] COPD
    Symptom cluster : Exertional Dyspnoea  (score 0.85)
    Expected        : COPD
    Primary Dx      : COPD  (92.3%)
    Questions asked : 4
    Top-3 differential:
      COPD                             [████████████████████] 92.3 %
      Asthma                           [██░░░░░░░░░░░░░░░░░░]  5.1 %
      Heart Failure                    [█░░░░░░░░░░░░░░░░░░░]  1.8 %
    Q&A trace:
      Q [test_fev1         ] What is the result of FEV1/FVC spirometry?
        A: FEV1/FVC 0.55 post-BD; fixed obstruction
      Q [test_cxr_hyp      ] What is the result of Chest X-ray (hyperinflation)?
        A: Hyperinflated, barrel chest, flattened diaphragms
      ...
```

---

## 8. Known Issues

### Issue 1: Double LLM calls inflate test timing
Each `submit_answer()` triggers 2 QuestionSelector LLM calls (the re-execution bug).
For a 4-question scenario: 8 selector calls + 4 evaluator calls = 12 LLM calls total
instead of the expected 8. This significantly inflates runtime.

### Issue 2: No answer provided if QuestionSelector picks a test not in FALLBACK_ANSWERS
If the LLM selector picks a test that's both (a) not in the scenario answers AND
(b) not in FALLBACK_ANSWERS, `_get_answer` returns `DEFAULT_ANSWER = "Normal / negative"`.
The evaluator would classify this as NEGATIVE for everything. Not a crash, but
potentially incorrect evidence injection.

Currently all 29 tests from the mock graph ARE in FALLBACK_ANSWERS, so this is not
an issue right now. Adding new tests to the mock without updating FALLBACK_ANSWERS
would trigger this.

### Issue 3: session.questions_asked vs run_scenario question counting
`run_scenario` increments `q_count` for each `submit_answer()` call.
`summary["questions_asked"]` = `q_count` from run_scenario.
But the orchestrator's `state["questions_asked"]` (used for MAX_QUESTIONS check) is
the list of questions that have been fully answered.
These should be equal, but if there's a re-entry edge case, they could diverge.

### Issue 4: Error handling swallows details
```python
except Exception as exc:
    print(f"ERROR — {exc}")
    results.append({
        "scenario": scenario["name"],
        "primary_dx": f"ERROR: {exc}",
        ...
    })
```
The full traceback is not printed. If an LLM call fails or a Python error occurs,
you see `ERROR — ...` with just the exception message. Run with `--verbose` to
at least see which scenario failed. A `traceback.print_exc()` here would help.

### Issue 5: 60-second default delay
Documented above. Always use `--delay 0` with Ollama.

---

## 9. What Run Scenarios Tests vs. What It Doesn't

### DOES test:
- Full pipeline: symptom search → seed → QA loop → finalize
- All 3 symptom clusters and 15 diseases
- LLM polarity classification on realistic clinical answers
- Bayesian convergence over multiple turns
- Confidence gate triggering correctly
- QuestionSelector choosing discriminating tests
- FALLBACK_ANSWERS correctly representing negative results

### Does NOT test:
- Unknown symptom handling (HP:0000001)
- Adversarial / injection patient answers
- Very short answers (1 word) or very long answers (near 1000 chars)
- Session expiry / invalid session_id
- MAX_QUESTIONS cap being reached
- Multiple concurrent sessions
- Anemia in palpitations context
- Any scenario where the FALLBACK is used for the PRIMARY discriminating test
