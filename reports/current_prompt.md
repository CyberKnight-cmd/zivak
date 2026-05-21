Given below is the output of run_scenarios.py

========================================================================
  ZIVAK — Multi-Disease Session Report
========================================================================

[✓] COPD
    Symptom cluster : Exertional Dyspnoea  (score 0.71)
    Expected        : COPD
    Primary Dx      : COPD  (99.7 %)
    Questions asked : 4
    Top-3 differential:
      COPD                             [███████████████████░] 99.7 %
      Asthma                           [░░░░░░░░░░░░░░░░░░░░] 0.3 %
      Heart Failure                    [░░░░░░░░░░░░░░░░░░░░] 0.0 %
    Q&A trace:
      Q [test_fev1       ] What is the result of FEV1/FVC spirometry?
        A: FEV1/FVC 0.55 post-BD; fixed obstruction
      Q [test_peak_flow  ] What is the result of Peak flow variability?
        A: <10% diurnal variability — not asthma
      Q [test_cxr_hyp    ] What is the result of Chest X-ray (hyperinflation)?
        A: Hyperinflated, barrel chest, flattened diaphragms
      Q [test_cbc        ] What is the result of Full blood count?
        A: Hb 14.4 g/dL, MCV 88 — normal blood count

[✓] Asthma
    Symptom cluster : Exertional Dyspnoea  (score 0.82)
    Expected        : Asthma
    Primary Dx      : Asthma  (99.4 %)
    Questions asked : 4
    Top-3 differential:
      Asthma                           [███████████████████░] 99.4 %
      Anemia                           [░░░░░░░░░░░░░░░░░░░░] 0.2 %
      Pneumonia                        [░░░░░░░░░░░░░░░░░░░░] 0.2 %
    Q&A trace:
      Q [test_fev1       ] What is the result of FEV1/FVC spirometry?
        A: FEV1/FVC 0.68 pre-BD, 0.80 post-BD — reversible obstruction
      Q [test_peak_flow  ] What is the result of Peak flow variability?
        A: 28% diurnal variability — consistent with asthma
      Q [test_bronch     ] What is the result of Bronchodilator response?
        A: FEV1 +22% after salbutamol — reversibility confirms asthma
      Q [test_peak_flow  ] What is the result of Peak flow variability?
        A: 28% diurnal variability — consistent with asthma

[✓] Heart Failure
    Symptom cluster : Exertional Dyspnoea  (score 0.58)
    Expected        : Heart Failure
    Primary Dx      : Heart Failure  (99.5 %)
    Questions asked : 4
    Top-3 differential:
      Heart Failure                    [███████████████████░] 99.5 %
      Pneumonia                        [░░░░░░░░░░░░░░░░░░░░] 0.2 %
      Asthma                           [░░░░░░░░░░░░░░░░░░░░] 0.2 %
    Q&A trace:
      Q [test_fev1       ] What is the result of FEV1/FVC spirometry?
        A: FEV1/FVC 0.76 — normal spirometry
      Q [test_bnp        ] What is the result of BNP / NT-proBNP?
        A: BNP 1200 pg/mL — markedly elevated
      Q [test_echo       ] What is the result of Echocardiogram?
        A: EF 25%, dilated LV, global systolic dysfunction
      Q [test_cbc        ] What is the result of full blood count?
        A: Hb 14.4 g/dL, MCV 88 — normal blood count

[✓] Pulmonary Embolism
    Symptom cluster : Exertional Dyspnoea  (score 0.71)
    Expected        : Pulmonary Embolism
    Primary Dx      : Pulmonary Embolism  (92.6 %)
    Questions asked : 4
    Top-3 differential:
      Pulmonary Embolism               [██████████████████░░] 92.6 %
      Asthma                           [░░░░░░░░░░░░░░░░░░░░] 4.2 %
      Anemia                           [░░░░░░░░░░░░░░░░░░░░] 1.8 %
    Q&A trace:
      Q [test_fev1       ] What is the result of FEV1/FVC spirometry?
        A: FEV1/FVC 0.79 — normal
      Q [test_bnp        ] What is the result of BNP / NT-proBNP?
        A: BNP 95 — normal, no heart failure
      Q [test_ddimer     ] What is the result of D-dimer?
        A: D-dimer 4.2 mg/L — significantly elevated
      Q [test_ctpa       ] What is the result of CT Pulmonary Angiogram?
        A: Bilateral saddle PE confirmed in main pulmonary arteries

[✗] Pneumonia
    Symptom cluster : Exertional Dyspnoea  (score 0.65)
    Expected        : Pneumonia
    Primary Dx      : COPD  (83.3 %)
    Questions asked : 8
    Top-3 differential:
      COPD                             [████████████████░░░░] 83.3 %
      Asthma                           [██░░░░░░░░░░░░░░░░░░] 13.1 %
      Heart Failure                    [░░░░░░░░░░░░░░░░░░░░] 1.4 %
    Q&A trace:
      Q [test_fev1       ] What is the result of FEV1/FVC spirometry?
        A: FEV1/FVC 0.79 — normal spirometry
      Q [test_bnp        ] What is the result of BNP / NT-proBNP?
        A: BNP 80 — normal
      Q [test_ddimer     ] What is the result of D-dimer?
        A: D-dimer 1.1 — mildly elevated, inflammatory
      Q [test_ctpa       ] What is the result of CT Pulmonary Angiogram?
        A: No pulmonary embolism identified
      Q [test_pft        ] What is the result of full pulmonary function tests?
        A: Normal spirometry and lung volumes
      Q [test_crp        ] What is the result of CRP / WBC?
        A: CRP 168 mg/L, WBC 16.8 — bacterial infection
      Q [test_cbc        ] What is the result of Full blood count?
        A: Hb 14.4 g/dL, MCV 88 — normal blood count
      Q [test_cxr_hyp    ] What is the result of Chest X-ray (hyperinflation)?
        A: Normal CXR — no hyperinflation or cardiomegaly

[✗] Anemia
    Symptom cluster : Exertional Dyspnoea  (score 0.88)
    Expected        : Anemia
    Primary Dx      : COPD  (84.4 %)
    Questions asked : 7
    Top-3 differential:
      COPD                             [████████████████░░░░] 84.4 %
      Asthma                           [█░░░░░░░░░░░░░░░░░░░] 8.1 %
      Anemia                           [░░░░░░░░░░░░░░░░░░░░] 3.7 %
    Q&A trace:
      Q [test_fev1       ] What is the result of FEV1/FVC spirometry?
        A: FEV1/FVC 0.82 — normal
      Q [test_bnp        ] What is the result of BNP / NT-proBNP?
        A: BNP 72 — normal
      Q [test_ctpa       ] What is the result of CT Pulmonary Angiogram?
        A: No pulmonary embolism identified
      Q [test_cxr_hyp    ] What is the result of Chest X-ray (hyperinflation)?
        A: Normal CXR
      Q [test_echo       ] What is the result of Echocardiogram?
        A: EF 62%, normal LV size and function
      Q [test_bronch     ] What is the result of Bronchodilator response?
        A: FEV1 +3% — no significant reversibility
      Q [test_crp        ] What is the result of CRP / WBC?
        A: CRP 3 mg/L, WBC 6.8 — normal

[✗] Interstitial Lung Disease
    Symptom cluster : Exertional Dyspnoea  (score 0.67)
    Expected        : Interstitial Lung Disease
    Primary Dx      : COPD  (92.9 %)
    Questions asked : 9
    Top-3 differential:
      COPD                             [██████████████████░░] 92.9 %
      Anemia                           [░░░░░░░░░░░░░░░░░░░░] 2.8 %
      Asthma                           [░░░░░░░░░░░░░░░░░░░░] 1.8 %
    Q&A trace:
      Q [test_fev1       ] What is the result of FEV1/FVC spirometry?
        A: FEV1/FVC 0.88 — normal ratio, reduced volumes (restrictive)
      Q [test_bnp        ] What is the result of BNP / NT-proBNP?
        A: BNP 42 pg/mL — normal
      Q [test_ddimer     ] What is the result of D-dimer?
        A: D-dimer 0.28 mg/L — normal
      Q [test_cxr_hyp    ] What is the result of Chest X-ray (hyperinflation)?
        A: Bilateral basal reticular shadowing — no hyperinflation
      Q [test_echo       ] What is the result of Echocardiogram?
        A: EF 62%, normal LV size and function
      Q [test_bronch     ] What is the result of Bronchodilator response?
        A: No reversibility — not asthma
      Q [test_crp        ] What is the result of CRP / WBC?
        A: CRP 3 mg/L, WBC 6.8 — normal
      Q [test_crp        ] What is the result of CRP / WBC?
        A: CRP 3 mg/L, WBC 6.8 — normal
      Q [test_peak_flow  ] What is the result of Peak flow variability?
        A: Diurnal variability <8% — not consistent with asthma

[✓] Atrial Fibrillation
    Symptom cluster : Palpitations  (score 0.75)
    Expected        : Atrial Fibrillation
    Primary Dx      : Atrial Fibrillation  (97.0 %)
    Questions asked : 4
    Top-3 differential:
      Atrial Fibrillation              [███████████████████░] 97.0 %
      SVT                              [░░░░░░░░░░░░░░░░░░░░] 2.7 %
      Anemia                           [░░░░░░░░░░░░░░░░░░░░] 0.2 %
    Q&A trace:
      Q [test_ecg        ] What is the result of 12-lead ECG?
        A: Irregularly irregular, no P waves — AF confirmed
      Q [test_holter     ] What is the result of 24h Holter monitor?
        A: 8h AF in 24h Holter recording
      Q [test_echo_pal   ] What is the result of Echocardiogram?
        A: LA 5.1 cm dilated, mild MR, EF 55%
      Q [test_cbc        ] What is the result of Full blood count?
        A: Hb 14.4 g/dL, MCV 88 — normal blood count

[✓] SVT
    Symptom cluster : Palpitations  (score 0.89)
    Expected        : SVT
    Primary Dx      : SVT  (69.8 %)
    Questions asked : 8
    Top-3 differential:
      SVT                              [█████████████░░░░░░░] 69.8 %
      Atrial Fibrillation              [█████░░░░░░░░░░░░░░░] 28.1 %
      Anxiety                          [░░░░░░░░░░░░░░░░░░░░] 2.1 %
    Q&A trace:
      Q [test_ecg        ] What is the result of 12-lead ECG?
        A: Regular narrow-complex tachycardia 178 bpm — SVT captured
      Q [test_holter     ] What is the result of 24h Holter monitor?
        A: 3 self-terminating SVT episodes; longest 4 min
      Q [test_echo_pal   ] What is the result of Echocardiogram?
        A: Normal cardiac structure — no structural cause
      Q [test_tsh        ] What is the result of TSH + free T4?
        A: TSH 1.8 — normal
      Q [test_electro    ] What is the result of Electrolytes / Mg / Ca?
        A: K+ 3.0 mmol/L — hypokalemia triggering SVT
      Q [test_cbc        ] What is the result of Full blood count?
        A: Hb 14.4 g/dL, MCV 88 — normal blood count
      Q [test_gad7       ] What is the result of GAD-7 anxiety screen?
        A: GAD-7 5 — mild, not primary
      Q [test_ferritin   ] What is the result of Ferritin / iron studies?
        A: Ferritin 74 ng/mL — normal iron stores

[✓] Anxiety
    Symptom cluster : Palpitations  (score 0.82)
    Expected        : Anxiety
    Primary Dx      : Anxiety  (99.7 %)
    Questions asked : 4
    Top-3 differential:
      Anxiety                          [███████████████████░] 99.7 %
      Hyperthyroidism                  [░░░░░░░░░░░░░░░░░░░░] 0.1 %
      Anemia                           [░░░░░░░░░░░░░░░░░░░░] 0.1 %
    Q&A trace:
      Q [test_ecg        ] What is the result of 12-lead ECG?
        A: Sinus tachycardia 102 bpm, regular P waves — no arrhythmia
      Q [test_tsh        ] What are the results of TSH + free T4?
        A: TSH 2.4 — normal
      Q [test_cbc        ] What is the result of full blood count?
        A: Hb 14.4 g/dL, MCV 88 — normal blood count
      Q [test_holter     ] What is the result of 24h Holter monitor?
        A: Persistent sinus tachycardia; no arrhythmia; rate tracks anxiety

[✓] Hyperthyroidism
    Symptom cluster : Palpitations  (score 0.59)
    Expected        : Hyperthyroidism
    Primary Dx      : Hyperthyroidism  (50.8 %)
    Questions asked : 8
    Top-3 differential:
      Hyperthyroidism                  [██████████░░░░░░░░░░] 50.8 %
      SVT                              [█████░░░░░░░░░░░░░░░] 28.9 %
      Atrial Fibrillation              [██░░░░░░░░░░░░░░░░░░] 11.6 %
    Q&A trace:
      Q [test_ecg        ] What is the result of 12-lead ECG?
        A: Sinus tachycardia 118 bpm — no primary arrhythmia
      Q [test_holter     ] What is the result of 24h Holter monitor?
        A: Persistent sinus tachycardia; no AF or SVT
      Q [test_tsh        ] What is the result of TSH + free T4?
        A: TSH <0.01, fT4 38 — overt hyperthyroidism
      Q [test_echo_pal   ] What is the result of Echocardiogram?
        A: Normal cardiac structure and function
      Q [test_echo_pal   ] What is the result of Echocardiogram?
        A: Normal cardiac structure and function
      Q [test_echo_pal   ] What is the result of Echocardiogram?
        A: Normal cardiac structure and function
      Q [test_cbc        ] What is the result of Full blood count?
        A: Hb 14.4 g/dL, MCV 88 — normal blood count
      Q [test_ferritin   ] What is the result of Ferritin / iron studies?
        A: Ferritin 74 ng/mL — normal iron stores

[✓] BPPV
    Symptom cluster : Vertigo  (score 0.74)
    Expected        : BPPV
    Primary Dx      : BPPV  (100.0 %)
    Questions asked : 4
    Top-3 differential:
      BPPV                             [███████████████████░] 100.0 %
      Meniere's Disease                [░░░░░░░░░░░░░░░░░░░░] 0.0 %
      Vestibular Neuritis              [░░░░░░░░░░░░░░░░░░░░] 0.0 %
    Q&A trace:
      Q [test_dix        ] What is the result of Dix-Hallpike maneuver?
        A: Positive right Dix-Hallpike: upbeat-torsional nystagmus, fatigues
      Q [test_hit        ] What is the result of Head impulse test (HIT)?
        A: Negative HIT — no corrective saccade
      Q [test_mri        ] What is the result of MRI brain?
        A: Normal MRI — no posterior fossa lesion
      Q [test_roll       ] What is the result of Supine roll test?
        A: Positive supine roll — right horizontal-canal BPPV

[✓] Vestibular Neuritis
    Symptom cluster : Vertigo  (score 0.77)
    Expected        : Vestibular Neuritis
    Primary Dx      : Vestibular Neuritis  (99.5 %)
    Questions asked : 4
    Top-3 differential:
      Vestibular Neuritis              [███████████████████░] 99.5 %
      Meniere's Disease                [░░░░░░░░░░░░░░░░░░░░] 0.4 %
      Central Vertigo                  [░░░░░░░░░░░░░░░░░░░░] 0.1 %
    Q&A trace:
      Q [test_dix        ] What is the result of Dix-Hallpike maneuver?
        A: Negative Dix-Hallpike — no positional nystagmus
      Q [test_vng        ] What is the result of Videonystagmography?
        A: Unilateral caloric hypofunction right — canal paresis 68%
      Q [test_audio      ] What is the result of pure tone audiometry?
        A: Hearing normal bilaterally
      Q [test_hit        ] What is the result of Head impulse test (HIT)?
        A: Positive HIT: corrective saccade to the right

[✗] Meniere's Disease
    Symptom cluster : Vertigo  (score 0.55)
    Expected        : Meniere's Disease
    Primary Dx      : Vestibular Neuritis  (85.1 %)
    Questions asked : 5
    Top-3 differential:
      Vestibular Neuritis              [█████████████████░░░] 85.1 %
      Meniere's Disease                [██░░░░░░░░░░░░░░░░░░] 13.3 %
      BPPV                             [░░░░░░░░░░░░░░░░░░░░] 1.5 %
    Q&A trace:
      Q [test_dix        ] What is the result of Dix-Hallpike maneuver?
        A: Negative Dix-Hallpike — no positional nystagmus
      Q [test_vng        ] What is the result of Videonystagmography?
        A: Reduced caloric response left; episodic low-freq nystagmus
      Q [test_audio      ] What is the result of pure tone audiometry?
        A: Low-freq SNHL (250–1000 Hz) left ear — classic Meniere's
      Q [test_mri        ] What is the result of MRI brain?
        A: Normal MRI; endolymphatic hydrops suspected clinically
      Q [test_tymp       ] What is the result of Tympanometry?
        A: Type A bilaterally — middle ear normal

[✓] Central Vertigo
    Symptom cluster : Vertigo  (score 0.66)
    Expected        : Central Vertigo
    Primary Dx      : Central Vertigo  (91.8 %)
    Questions asked : 5
    Top-3 differential:
      Central Vertigo                  [██████████████████░░] 91.8 %
      Meniere's Disease                [█░░░░░░░░░░░░░░░░░░░] 5.1 %
      BPPV                             [░░░░░░░░░░░░░░░░░░░░] 2.9 %
    Q&A trace:
      Q [test_dix        ] What is the result of Dix-Hallpike maneuver?
        A: Direction-changing nystagmus, non-fatiguing — central pattern
      Q [test_hit        ] What is the result of Head impulse test (HIT)?
        A: Negative HIT — intact VOR suggests central pathology
      Q [test_mri        ] What is the result of MRI brain?
        A: 1.9 cm posterior fossa mass, right cerebellum, perilesional oedema
      Q [test_vng        ] What is the result of Videonystagmography?
        A: Gaze-evoked nystagmus + ocular dysmetria — central signs
      Q [test_roll       ] What is the result of Supine roll test?
        A: Negative supine roll test bilaterally

------------------------------------------------------------------------
  Result: 11/15 correct (73 %)
========================================================================



I want you to see if there's some huge problem in this output, or is it only because we are using mock databases right now and the data isnt sufficient, or maybe the tests arem't sufficient, thats why this output has come, do you see any other huge problem architecture wise that could bump up my accuracy (not  forcibly, only a genuine architecture change, like some flaw in asking question) then add it to 08_known_issues.md


Im also giving you the previous flaws below that you gave me, you have to make sure that the code base doesnt contain these errors right now, and theyre not dampening any performance, if you think they are, add it to 08_known_issues.md


Critical Problems (System Breaks)

  1. EvidenceEvaluatorAgent — fundamentally wrong logic for negative results

  This is the deepest bug. The edges in the knowledge graph encode positive-result LRs:
  test_fev1: COPD RULES_IN lr=8.5 | Asthma RULES_IN lr=3.0 | HeartFailure RULES_OUT lr=0.5 | PE RULES_OUT lr=0.6

  The prompt says:

  ▎ "If answer indicates test is negative → use RULES_OUT edges"

  So for a negative FEV1 (normal spirometry), the LLM puts Heart Failure and PE into rules_out with LR 0.5 and
  0.6 — which crushes their probability when it should raise it. A normal FEV1 makes PE and Heart Failure more
  likely, not less. The correct behaviour for a negative test is to invert every LR (1/LR): RULES_IN diseases get
   penalised (1/8.5 ≈ 0.12), RULES_OUT diseases get boosted (1/0.5 = 2.0). The prompt never says this. The LLM
  has no idea.

  This means every negative test result is applied backwards — penalising diseases it should exonerate and vice
  versa. It is the primary reason the differential collapses to wrong answers.

  ---
  2. EvidenceEvaluatorAgent — incomplete edge application

  The prompt says "use RULES_IN edges" for a positive result — implying the LLM should only return the RULES_IN
  diseases. The RULES_OUT diseases (Heart Failure, PE, Asthma) should also be penalised from a positive FEV1, but
   the prompt silently ignores them. A well-written prompt would say: "for a positive result, output ALL edges:
  RULES_IN diseases in rules_in, RULES_OUT diseases in rules_out".

  ---
  3. run_scenarios.py — fallback answers are all disease-positive

  Every fallback defaults to a strongly abnormal result:
  test_cbc      → "Hb 7.9 g/dL, MCV 68 — microcytic anaemia"   (LR 15× boost for Anemia)
  test_bronch   → "FEV1 +18% after salbutamol; reversible"      (LR 6× boost for Asthma)
  test_peak_flow → "22% diurnal variability"                    (LR 7× boost for Asthma)

  When PE, COPD, Hyperthyroidism, or Vestibular Neuritis scenarios hit a test with no scenario override, they
  inherit an Anemia or Asthma lab result. The differential immediately shifts to the wrong disease before the
  discriminating tests are reached. Every fallback should be neutral/negative (normal result).

  ---
  4. QuestionSelectorAgent — no enforcement that returned test_id is in the available list

  The LLM routinely returns a test it knows from training (e.g. test_gad7) even after it has been filtered out of
   available_tests. There is no post-call assertion. This causes the same question to be asked 3× in a row,
  consuming the 4-question minimum budget on one useless repeated test. (Defensive fix added to orchestrator
  already.)

  ---
  High-Impact Problems

  5. DifferentialEngine — rules_in / rules_out distinction is ignored

  for rule in evidence.get('rules_in', []) + evidence.get('rules_out', []):
      lr = rule['likelihood_ratio']
      log_lr_map[key] += math.log(lr)

  The engine chains both lists identically — it only cares about the LR value, not which list it came from. So if
   the LLM puts a disease with LR 8.5 in rules_out by mistake, the engine still boosts that disease. The split
  into two lists is pure theatre — it affects nothing computationally.

  ---
  6. orchestrator.py — available tests only from top-3 diseases

  top_diseases = [d["name"] for d in state["differential"][:3]]
  available_tests = [t for t in neo4j.get_available_tests(top_diseases) ...]

  For Palpitations: initial top-3 is [AF, SVT, Anxiety]. test_tsh (Hyperthyroidism) and test_cbc (Anemia) are
  never in the pool — yet both are in the 5-disease differential and have very high LRs. The selector never gets
  a chance to ask the one test that would confirm Hyperthyroidism (LR 12) or definitively exclude Anemia (LR 15).
   Bump this to top-5.

  ---
  7. EvidenceEvaluatorAgent — temperature=1 on a structured output task

  Both agents use temperature=1. For the selector a little variance is acceptable. For the evaluator, which needs
   to output deterministic JSON that matches specific disease names and numerical LRs from the edges,
  temperature=1 is actively harmful — it introduces needless variation in JSON structure, disease name casing,
  and LR values, increasing parse failures and inconsistency.

  ---
  8. EvidenceEvaluatorAgent — reasoning_effort="low" on the hardest task

  The evaluator does the most cognitively complex job: interpret ambiguous clinical language, map it to a
  positive/negative result, then apply the correct directional LRs for every edge. It has reasoning_effort="low".
   The selector — which just picks the highest-value test from a ranked list — has reasoning_effort="medium".
  These should be swapped.

  ---
  9. ConfidenceJudge — stops at 4 questions, far too early

  min_evidence: int = 4
  min_top_confidence: float = 0.75

  After 4 questions with biased fallback answers, Anemia hits 90% and the session ends. The judge has no concept
  of whether the top disease actually received its own discriminating test — it just counts Q&A turns. A correct
  session might need the CBC result to be normal and the FEV1 to be obstructive and the chest X-ray to show
  hyperinflation. Four questions is not a safety net, it's a premature exit trigger.

  ---
  10. QuestionSelectorAgent — selector has no symptom or history context

  The selector prompt receives only the current differential and the available test list. It has no knowledge of:
  - What the original symptom was
  - What tests have already been asked and what the answers were
  - The clinical reasoning chain so far

  So it cannot make contextually informed decisions. It just picks the highest-value discriminating test in
  isolation each turn, without knowing that three turns ago the patient already demonstrated reversible airflow
  obstruction.

  ---
  Summary Table

  ┌─────┬────────────────────┬──────────┬────────────────────────────────────────────────┐
  │  #  │     Component      │ Severity │                   Root Cause                   │
  ├─────┼────────────────────┼──────────┼────────────────────────────────────────────────┤
  │ 1   │ EvidenceEvaluator  │ Critical │ Negative result LRs applied in wrong direction │
  ├─────┼────────────────────┼──────────┼────────────────────────────────────────────────┤
  │ 2   │ EvidenceEvaluator  │ Critical │ Incomplete edge application per result         │
  ├─────┼────────────────────┼──────────┼────────────────────────────────────────────────┤
  │ 3   │ run_scenarios.py   │ Critical │ Fallback answers are disease-positive          │
  ├─────┼────────────────────┼──────────┼────────────────────────────────────────────────┤
  │ 4   │ QuestionSelector   │ High     │ No test_id validation (patched)                │
  ├─────┼────────────────────┼──────────┼────────────────────────────────────────────────┤
  │ 5   │ DifferentialEngine │ High     │ rules_in/rules_out distinction unused          │
  ├─────┼────────────────────┼──────────┼────────────────────────────────────────────────┤
  │ 6   │ Orchestrator       │ High     │ Available tests from top-3 only                │
  ├─────┼────────────────────┼──────────┼────────────────────────────────────────────────┤
  │ 7   │ EvidenceEvaluator  │ High     │ temperature=1 on structured output             │
  ├─────┼────────────────────┼──────────┼────────────────────────────────────────────────┤
  │ 8   │ EvidenceEvaluator  │ High     │ reasoning_effort=low on hardest task           │
  ├─────┼────────────────────┼──────────┼────────────────────────────────────────────────┤
  │ 9   │ ConfidenceJudge    │ Medium   │ 4-question minimum too low                     │
  ├─────┼────────────────────┼──────────┼────────────────────────────────────────────────┤
  │ 10  │ QuestionSelector   │ Medium   │ No symptom/history context                     │
  └─────┴────────────────────┴──────────┴────────────────────────────────────────────────┘

  The single highest-leverage fix is #1+#2 in the evaluator prompt — correcting how negative results and
  incomplete edge application work. That alone would likely take accuracy from 20% to 70%+. Combined with neutral
   fallback answers (#3) and top-5 test discovery (#6), you should reach 85%+ correct.