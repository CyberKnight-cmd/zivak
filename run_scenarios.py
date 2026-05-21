"""
Multi-scenario diagnostic runner.

Exercises every disease in the mock knowledge graph by running one
automated session per scenario. Each scenario supplies concise,
disease-specific answers so the differential converges toward the
expected diagnosis without wasting tokens.

Diseases covered:
  Exertional Dyspnoea: COPD, Asthma, Heart Failure, Pulmonary Embolism,
                       Pneumonia, Anemia, Interstitial Lung Disease
  Palpitations:        Atrial Fibrillation, SVT, Anxiety, Hyperthyroidism
  Vertigo:             BPPV, Vestibular Neuritis, Meniere's Disease, Central Vertigo

Usage:
  python run_scenarios.py
  python run_scenarios.py --filter COPD Asthma BPPV
  python run_scenarios.py --delay 90 --turn-delay 5   # rate-limit tuning
  python run_scenarios.py --verbose
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from orchestrator.mock_clients import MockNeo4jClient, MockQdrantClient
from orchestrator.orchestrator import DiagnosticOrchestrator

# ------------------------------------------------------------------ #
#  Answers keyed by test_id — concise so the evaluator LLM call is   #
#  token-light. One clear sentence is enough; no repetition.         #
# ------------------------------------------------------------------ #

# Per-scenario overrides come first; these are the disease-targeted answers.
# FALLBACK_ANSWERS covers any test not listed in a scenario.
# DEFAULT_ANSWER is the last resort (treats the test as normal/negative).

FALLBACK_ANSWERS: dict[str, str] = {
    # All fallbacks are NORMAL / NEGATIVE so that a test asked in the wrong scenario
    # does not inject false evidence. Scenario-specific answers override these.
    "test_fev1":      "FEV1/FVC 0.79 — normal spirometry",
    "test_bronch":    "FEV1 +3% — no significant reversibility",
    "test_peak_flow": "Diurnal variability <8% — not consistent with asthma",
    "test_cxr_hyp":   "Normal CXR — no hyperinflation or cardiomegaly",
    "test_bnp":       "BNP 42 pg/mL — normal",
    "test_echo":      "EF 62%, normal LV size and function",
    "test_cxr_card":  "Normal cardiac silhouette — no cardiomegaly",
    "test_ddimer":    "D-dimer 0.28 mg/L — normal",
    "test_ctpa":      "No pulmonary embolism identified",
    "test_wells":     "Wells score 1 — low clinical probability",
    "test_cxr_inf":   "Clear lung fields — no consolidation",
    "test_sputum":    "No significant pathogens isolated",
    "test_crp":       "CRP 3 mg/L, WBC 6.8 — normal",
    "test_cbc":       "Hb 14.4 g/dL, MCV 88 — normal blood count",
    "test_ferritin":  "Ferritin 74 ng/mL — normal iron stores",
    "test_hrct":      "Normal HRCT — no parenchymal abnormality",
    "test_pft":       "Normal spirometry and lung volumes",
    "test_ecg":       "Normal sinus rhythm — no arrhythmia",
    "test_holter":    "Normal sinus rhythm throughout 24h recording",
    "test_echo_pal":  "Normal cardiac structure and function",
    "test_electro":   "Electrolytes all within normal range",
    "test_gad7":      "GAD-7 2 — minimal/no anxiety",
    "test_tsh":       "TSH 1.9, fT4 14.2 — normal thyroid function",
    "test_dix":       "Negative Dix-Hallpike bilaterally — no positional nystagmus",
    "test_roll":      "Negative supine roll test bilaterally",
    "test_hit":       "Negative HIT — intact VOR, no corrective saccade",
    "test_vng":       "Normal VNG — symmetric caloric responses bilaterally",
    "test_audio":     "Normal hearing bilaterally",
    "test_tymp":      "Type A tympanograms bilaterally — middle ear normal",
    "test_mri":       "Normal MRI brain — no lesion identified",
}

DEFAULT_ANSWER = "Normal / negative"

# ------------------------------------------------------------------ #
#  Scenario definitions                                               #
# ------------------------------------------------------------------ #

SCENARIOS = [
    # ===== Exertional Dyspnoea =====
    {
        "name":     "COPD",
        "symptom":  "I get winded very easily, especially going upstairs — I smoked for 30 years",
        "expected": "COPD",
        "answers": {
            "test_fev1":      "FEV1/FVC 0.55 post-BD; fixed obstruction",
            "test_bronch":    "FEV1 +3% — no significant reversibility",
            "test_peak_flow": "<10% diurnal variability — not asthma",
            "test_cxr_hyp":   "Hyperinflated, barrel chest, flattened diaphragms",
            "test_hrct":      "Centrilobular emphysema + bullae, upper lobes",
            "test_pft":       "Obstruction, air trapping: RV 175%, DLCO 60%",
        },
    },
    {
        "name":     "Asthma",
        "symptom":  "I get breathless climbing stairs — worse at night and in cold air",
        "expected": "Asthma",
        "answers": {
            "test_fev1":      "FEV1/FVC 0.68 pre-BD, 0.80 post-BD — reversible obstruction",
            "test_bronch":    "FEV1 +22% after salbutamol — reversibility confirms asthma",
            "test_peak_flow": "28% diurnal variability — consistent with asthma",
            "test_cxr_hyp":   "Normal CXR — no hyperinflation",
            "test_hrct":      "Mild bronchial wall thickening; no emphysema",
        },
    },
    {
        "name":     "Heart Failure",
        "symptom":  "short of breath on exertion, swollen ankles, waking up gasping at night",
        "expected": "Heart Failure",
        "answers": {
            "test_bnp":       "BNP 1200 pg/mL — markedly elevated",
            "test_echo":      "EF 25%, dilated LV, global systolic dysfunction",
            "test_cxr_card":  "Cardiomegaly, bilateral oedema, Kerley B lines",
            "test_fev1":      "FEV1/FVC 0.76 — normal spirometry",
            "test_bronch":    "No reversibility — not asthma",
            "test_ddimer":    "D-dimer 0.6 mg/L — mildly elevated, cardiac cause likely",
        },
    },
    {
        "name":     "Pulmonary Embolism",
        "symptom":  "I feel like I am suffocating during exercise and my left calf is red and swollen",
        "expected": "Pulmonary Embolism",
        "answers": {
            "test_ddimer":    "D-dimer 4.2 mg/L — significantly elevated",
            "test_ctpa":      "Bilateral saddle PE confirmed in main pulmonary arteries",
            "test_wells":     "Wells 7 — high probability",
            "test_fev1":      "FEV1/FVC 0.79 — normal",
            "test_bnp":       "BNP 95 — normal, no heart failure",
            "test_cxr_inf":   "Clear CXR — no consolidation",
        },
    },
    {
        "name":     "Pneumonia",
        "symptom":  "difficulty breathing when active with fever, chills, productive yellow cough",
        "expected": "Pneumonia",
        "answers": {
            "test_cxr_inf":   "RLL consolidation with air bronchograms",
            "test_sputum":    "S. pneumoniae heavy growth, amoxicillin-sensitive",
            "test_crp":       "CRP 168 mg/L, WBC 16.8 — bacterial infection",
            "test_bnp":       "BNP 80 — normal",
            "test_ddimer":    "D-dimer 1.1 — mildly elevated, inflammatory",
        },
    },
    {
        "name":     "Anemia",
        "symptom":  "out of breath after walking a short distance, very pale and exhausted",
        "expected": "Anemia",
        "answers": {
            "test_cbc":       "Hb 7.2, MCV 65, MCH 19 — severe microcytic anaemia",
            "test_ferritin":  "Ferritin 2 ng/mL — critically low",
            "test_fev1":      "FEV1/FVC 0.82 — normal",
            "test_bnp":       "BNP 72 — normal",
            "test_cxr_hyp":   "Normal CXR",
        },
    },
    {
        "name":     "Interstitial Lung Disease",
        "symptom":  "I get winded easily and have had a dry persistent cough for six months",
        "expected": "Interstitial Lung Disease",
        "answers": {
            "test_hrct":      "Basal honeycombing, traction bronchiectasis, bilateral GGO",
            "test_pft":       "Restrictive: TLC 62%, DLCO 44%",
            "test_fev1":      "FEV1/FVC 0.88 — normal ratio, reduced volumes (restrictive)",
            "test_cxr_hyp":   "Bilateral basal reticular shadowing — no hyperinflation",
            "test_bronch":    "No reversibility — not asthma",
        },
    },

    # ===== Palpitations =====
    {
        "name":     "Atrial Fibrillation",
        "symptom":  "my heart is racing and beating irregularly — comes and goes",
        "expected": "Atrial Fibrillation",
        "answers": {
            "test_ecg":       "Irregularly irregular, no P waves — AF confirmed",
            "test_holter":    "8h AF in 24h Holter recording",
            "test_echo_pal":  "LA 5.1 cm dilated, mild MR, EF 55%",
            "test_gad7":      "GAD-7 4 — minimal anxiety",
            "test_tsh":       "TSH 2.1 — normal",
            "test_electro":   "Electrolytes normal",
        },
    },
    {
        "name":     "SVT",
        "symptom":  "sudden rapid heartbeat that starts and stops abruptly",
        "expected": "SVT",
        "answers": {
            "test_ecg":       "Regular narrow-complex tachycardia 178 bpm — SVT captured",
            "test_holter":    "3 self-terminating SVT episodes; longest 4 min",
            "test_electro":   "K+ 3.0 mmol/L — hypokalemia triggering SVT",
            "test_echo_pal":  "Normal cardiac structure — no structural cause",
            "test_gad7":      "GAD-7 5 — mild, not primary",
            "test_tsh":       "TSH 1.8 — normal",
        },
    },
    {
        "name":     "Anxiety",
        "symptom":  "heart pounding in my chest, very anxious all the time and cannot relax",
        "expected": "Anxiety",
        "answers": {
            "test_gad7":      "GAD-7 19/21 — severe GAD",
            "test_ecg":       "Sinus tachycardia 102 bpm, regular P waves — no arrhythmia",
            "test_holter":    "Persistent sinus tachycardia; no arrhythmia; rate tracks anxiety",
            "test_tsh":       "TSH 2.4 — normal",
            "test_electro":   "Electrolytes normal",
        },
    },
    {
        "name":     "Hyperthyroidism",
        "symptom":  "heart fluttering, rapid weight loss, feeling very hot all the time",
        "expected": "Hyperthyroidism",
        "answers": {
            "test_tsh":       "TSH <0.01, fT4 38 — overt hyperthyroidism",
            "test_ecg":       "Sinus tachycardia 118 bpm — no primary arrhythmia",
            "test_holter":    "Persistent sinus tachycardia; no AF or SVT",
            "test_gad7":      "GAD-7 7 — mild, likely secondary to hyperthyroid",
            "test_electro":   "Electrolytes normal",
        },
    },

    # ===== Vertigo =====
    {
        "name":     "BPPV",
        "symptom":  "the room is spinning when I roll over in bed or tilt my head back",
        "expected": "BPPV",
        "answers": {
            "test_dix":       "Positive right Dix-Hallpike: upbeat-torsional nystagmus, fatigues",
            "test_roll":      "Positive supine roll — right horizontal-canal BPPV",
            "test_hit":       "Negative HIT — no corrective saccade",
            "test_vng":       "Positional nystagmus only — no spontaneous; BPPV pattern",
            "test_mri":       "Normal MRI — no posterior fossa lesion",
        },
    },
    {
        "name":     "Vestibular Neuritis",
        "symptom":  "I feel dizzy and lose my balance continuously — started suddenly after a cold",
        "expected": "Vestibular Neuritis",
        "answers": {
            "test_hit":       "Positive HIT: corrective saccade to the right",
            "test_vng":       "Unilateral caloric hypofunction right — canal paresis 68%",
            "test_dix":       "Negative Dix-Hallpike — no positional nystagmus",
            "test_mri":       "Normal MRI — no central lesion",
            "test_audio":     "Hearing normal bilaterally",
        },
    },
    {
        "name":     "Meniere's Disease",
        "symptom":  "spinning with ringing in my left ear and fluctuating hearing loss",
        "expected": "Meniere's Disease",
        "answers": {
            "test_audio":     "Low-freq SNHL (250–1000 Hz) left ear — classic Meniere's",
            "test_vng":       "Reduced caloric response left; episodic low-freq nystagmus",
            "test_tymp":      "Type A bilaterally — middle ear normal",
            "test_dix":       "Negative Dix-Hallpike — no positional nystagmus",
            "test_mri":       "Normal MRI; endolymphatic hydrops suspected clinically",
        },
    },
    {
        "name":     "Central Vertigo",
        "symptom":  "spinning even when still, with double vision and difficulty walking",
        "expected": "Central Vertigo",
        "answers": {
            "test_mri":       "1.9 cm posterior fossa mass, right cerebellum, perilesional oedema",
            "test_hit":       "Negative HIT — intact VOR suggests central pathology",
            "test_dix":       "Direction-changing nystagmus, non-fatiguing — central pattern",
            "test_vng":       "Gaze-evoked nystagmus + ocular dysmetria — central signs",
        },
    },
]

# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def _get_answer(scenario: dict, test_id: str) -> str:
    return (
        scenario.get("answers", {}).get(test_id)
        or FALLBACK_ANSWERS.get(test_id)
        or DEFAULT_ANSWER
    )


def _pause(seconds: float, label: str) -> None:
    if seconds <= 0:
        return
    print(f"  ⏸  {label} — waiting {seconds:.0f}s …", flush=True)
    time.sleep(seconds)


# ------------------------------------------------------------------ #
#  Session runner                                                     #
# ------------------------------------------------------------------ #

def run_scenario(orch: DiagnosticOrchestrator, scenario: dict, turn_delay: float) -> dict:
    session_id, result = orch.start_session(scenario["symptom"])

    symptom_match = result["symptom_match"]
    q_count       = 0
    question      = result.get("next_question")
    questions_log = []

    while question:
        test_id = question.get("test_id", "")
        answer  = _get_answer(scenario, test_id)
        questions_log.append({
            "q":      question.get("question", ""),
            "test":   test_id,
            "answer": answer,
        })

        if turn_delay > 0 and q_count > 0:
            _pause(turn_delay, f"turn {q_count + 1}")

        result  = orch.submit_answer(session_id, answer)
        q_count += 1

        if not result.get("should_continue") or result.get("final_diagnosis"):
            break
        question = result.get("next_question")

    final = result.get("final_diagnosis") or {}
    return {
        "scenario":        scenario["name"],
        "expected":        scenario["expected"],
        "symptom_matched": symptom_match.get("clinical_term", "?"),
        "match_score":     symptom_match.get("score", 0.0),
        "primary_dx":      final.get("primary_diagnosis", "—"),
        "confidence":      final.get("confidence", 0.0),
        "questions_asked": q_count,
        "questions_log":   questions_log,
        "top3":            final.get("differential", [])[:3],
        "correct":         final.get("primary_diagnosis") == scenario["expected"],
    }


# ------------------------------------------------------------------ #
#  Output                                                             #
# ------------------------------------------------------------------ #

BAR = "=" * 72

def _print_detail(r: dict) -> None:
    tick = "✓" if r["correct"] else "✗"
    print(f"\n[{tick}] {r['scenario']}")
    print(f"    Symptom cluster : {r['symptom_matched']}  (score {r['match_score']:.2f})")
    print(f"    Expected        : {r['expected']}")
    print(f"    Primary Dx      : {r['primary_dx']}  ({r['confidence'] * 100:.1f} %)")
    print(f"    Questions asked : {r['questions_asked']}")
    if r["top3"]:
        print("    Top-3 differential:")
        for d in r["top3"]:
            bar_len = int(d["probability"] * 20)
            bar     = "█" * bar_len + "░" * (20 - bar_len)
            print(f"      {d['name']:32s} [{bar}] {d['probability'] * 100:.1f} %")
    if r["questions_log"]:
        print("    Q&A trace:")
        for entry in r["questions_log"]:
            print(f"      Q [{entry['test']:16s}] {entry['q']}")
            print(f"        A: {entry['answer']}")


def print_report(results: list[dict], verbose: bool) -> None:
    print(f"\n{BAR}")
    print("  ZIVAK — Multi-Disease Session Report")
    print(BAR)

    if verbose:
        for r in results:
            _print_detail(r)
    else:
        header = f"  {'Scenario':<26} {'Expected':<26} {'Diagnosed':<26} {'Conf':>6}  {'Qs':>3}  OK?"
        print(f"\n{header}")
        print("  " + "-" * 70)
        for r in results:
            tick = "✓" if r["correct"] else "✗"
            print(
                f"  {r['scenario']:<26} {r['expected']:<26} {r['primary_dx']:<26}"
                f" {r['confidence'] * 100:5.1f}%  {r['questions_asked']:>3}   {tick}"
            )

    correct = sum(1 for r in results if r["correct"])
    print(f"\n{'-' * 72}")
    print(f"  Result: {correct}/{len(results)} correct ({correct / len(results) * 100:.0f} %)")
    print(f"{BAR}\n")


# ------------------------------------------------------------------ #
#  Entry point                                                        #
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(description="Run ZIVAK multi-disease diagnostic scenarios")
    p.add_argument(
        "--filter", nargs="*", metavar="NAME",
        help="Run only named scenario(s) — case-insensitive substring match",
    )
    p.add_argument(
        "--delay", type=float, default=60.0, metavar="SECS",
        help="Seconds to wait between scenarios to avoid Groq rate limits (default: 60)",
    )
    p.add_argument(
        "--turn-delay", type=float, default=0.0, metavar="SECS",
        help="Seconds to wait between Q&A turns within a scenario (default: 0)",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print full Q&A trace for every scenario",
    )
    return p.parse_args()


def main():
    args    = parse_args()
    filter_ = [f.lower() for f in args.filter] if args.filter else None

    scenarios = SCENARIOS
    if filter_:
        scenarios = [s for s in SCENARIOS if any(f in s["name"].lower() for f in filter_)]
        if not scenarios:
            print(f"No scenarios matched filter: {args.filter}", file=sys.stderr)
            sys.exit(1)

    total_calls = len(scenarios) * 10 * 2  # rough upper bound: 10 turns × 2 LLM calls
    print(f"Initialising mock clients (embedding model loads once) …")
    print(f"Running {len(scenarios)} scenario(s)  |  inter-scenario delay: {args.delay:.0f}s  |  turn delay: {args.turn_delay:.0f}s")
    print(f"Estimated LLM calls (upper bound): ~{total_calls}")

    qdrant = MockQdrantClient()
    neo4j  = MockNeo4jClient()
    orch   = DiagnosticOrchestrator(qdrant, neo4j)

    results = []
    for i, scenario in enumerate(scenarios, 1):
        if i > 1 and args.delay > 0:
            _pause(args.delay, f"rate-limit gap before scenario {i}")

        print(f"\n[{i:02d}/{len(scenarios)}] {scenario['name']} …", end=" ", flush=True)
        try:
            summary = run_scenario(orch, scenario, turn_delay=args.turn_delay)
            results.append(summary)
            tick = "✓" if summary["correct"] else "✗"
            print(
                f"{tick}  Dx={summary['primary_dx']} "
                f"({summary['confidence'] * 100:.1f}%)  Qs={summary['questions_asked']}"
            )
        except Exception as exc:
            print(f"ERROR — {exc}")
            results.append({
                "scenario": scenario["name"], "expected": scenario["expected"],
                "symptom_matched": "?", "match_score": 0.0,
                "primary_dx": f"ERROR: {exc}", "confidence": 0.0,
                "questions_asked": 0, "questions_log": [], "top3": [], "correct": False,
            })

    print_report(results, verbose=args.verbose)


if __name__ == "__main__":
    main()
