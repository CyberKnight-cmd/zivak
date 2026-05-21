"""
run_scenarios.py — LLM-powered rare-disease diagnostic scenario runner.

Drives the ZIVAK orchestrator against the real Neo4j graph (HPO/Orphanet corpus).
A Groq-backed simulator answers diagnostic questions creatively so the whole
pipeline can run end-to-end without GPU conflicts.
Every session is recorded to reports/runs/<timestamp>.json for analysis.

Usage:
  uv run run_scenarios.py
  uv run run_scenarios.py --filter Prader-Willi Kabuki CHARGE
  uv run run_scenarios.py --verbose --delay 5 --turn-delay 2
  uv run run_scenarios.py --simulator-model llama-3.1-8b-instant
  uv run run_scenarios.py --no-log

Environment:
  GROQ_API_KEY    Required — patient simulator (keeps GPU free for orchestrator)
  LOCAL_LLM_URL   Ollama endpoint for the orchestrator (default: http://localhost:11434/v1)
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from orchestrator.mock_clients import get_clients
from orchestrator.orchestrator import DiagnosticOrchestrator

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Patient Simulator                                                  #
# ------------------------------------------------------------------ #

_SIM_RETRIES = 3
_SIM_BACKOFF = 1.0

_SIMULATOR_PROMPT = """\
You are simulating a clinical encounter. The patient presents with:
"{symptom}"

A diagnostic test result is being reviewed. Decide creatively whether the \
finding is POSITIVE (present/abnormal) or NEGATIVE (absent/normal), then give \
a brief clinical response.

Test / question: {question}

Reply in 1-2 sentences. State clearly if the finding is present or absent and \
add one plausible clinical detail. No caveats or hedging."""


class PatientSimulatorAgent:
    """
    Groq-backed patient simulator. Answers HPO diagnostic questions with a
    plausible positive or negative result. Runs entirely via the API so it
    does not compete with the orchestrator's local Ollama model for GPU VRAM.
    """

    def __init__(self, model: str | None = None):
        self.model = model or "llama-3.3-70b-versatile"
        groq_key = os.getenv("GROQ_API_KEY", "")
        if not groq_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. The patient simulator requires Groq "
                "so it does not compete with the orchestrator for GPU VRAM."
            )
        from langchain_groq import ChatGroq
        self._llm = ChatGroq(
            model=self.model,
            temperature=1,   # randomness so answers vary across sessions
            max_tokens=120,
            api_key=groq_key,
        )
        logger.info("PatientSimulator: groq/%s", self.model)

    def answer(self, question: str, symptom: str, scenario_name: str) -> str:
        prompt = _SIMULATOR_PROMPT.format(symptom=symptom, question=question)

        for attempt in range(_SIM_RETRIES):
            try:
                response = self._llm.invoke(prompt)
                content  = response.content.strip()
                content  = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                if content:
                    return content
                raise ValueError("empty response")
            except Exception as exc:
                if attempt < _SIM_RETRIES - 1:
                    time.sleep(_SIM_BACKOFF * (2 ** attempt))
                else:
                    logging.warning("PatientSimulator failed (%s) — using fallback", exc)
        return "Negative — no abnormality detected for this finding."


# ------------------------------------------------------------------ #
#  Scenarios                                                          #
#  All diseases are from the HPO/Orphanet rare-disease graph loaded  #
#  in Neo4j. Disease names must EXACTLY match Neo4j node names.      #
# ------------------------------------------------------------------ #

SCENARIOS = [
    {
        "name":     "Prader-Willi",
        "symptom":  "infant born floppy with weak cry and poor feeding, now an obese child with intellectual disability and behavioral problems",
        "expected": "Prader-Willi syndrome",
    },
    {
        "name":     "Kabuki",
        "symptom":  "child with global developmental delay, distinctive wide eyes, broad nasal tip, and persistent fetal fingertip pads",
        "expected": "Kabuki syndrome",
    },
    {
        "name":     "Noonan",
        "symptom":  "short child with widely spaced eyes, drooping eyelids, webbed neck, and a heart murmur since birth",
        "expected": "Noonan syndrome",
    },
    {
        "name":     "Williams-Beuren",
        "symptom":  "toddler with heart disease, learning difficulties, overly friendly personality, and severe anxiety",
        "expected": "Williams-Beuren syndrome",
    },
    {
        "name":     "Neurofibromatosis 1",
        "symptom":  "young adult with multiple flat brown skin patches since childhood and soft nodular lumps developing under the skin",
        "expected": "neurofibromatosis 1",
    },
    {
        "name":     "Fragile X",
        "symptom":  "teenage boy with moderate intellectual disability, large ears, poor eye contact, and hyperactive repetitive behaviors",
        "expected": "fragile X syndrome",
    },
    {
        "name":     "Tuberous Sclerosis 2",
        "symptom":  "infant with seizures from three months of age, white depigmented skin patches, and progressive developmental delay",
        "expected": "tuberous sclerosis 2",
    },
    {
        "name":     "CHARGE",
        "symptom":  "newborn with a unilateral eye coloboma, profound sensorineural deafness, heart defect, and complete anosmia",
        "expected": "CHARGE syndrome",
    },
    {
        "name":     "Huntington-Like 1",
        "symptom":  "middle-aged adult with uncontrollable jerking arm movements, personality changes, and progressive memory loss",
        "expected": "Huntington's disease-like 1",
    },
    {
        "name":     "Cockayne",
        "symptom":  "child with progressively small head, deteriorating hearing and vision, looks much older than their age",
        "expected": "Cockayne syndrome",
    },
    {
        "name":     "Rett Congenital",
        "symptom":  "infant with severe hypotonia from birth, no motor development, cannot sit, and very simplified brain gyral pattern on MRI",
        "expected": "congenital variant of Rett syndrome",
    },
    {
        "name":     "Mito DNA Depletion 6",
        "symptom":  "infant with profound hypotonia, liver failure, lactic acidosis, and rapidly progressive neurological decline",
        "expected": "mitochondrial DNA depletion syndrome 6",
    },
]


# ------------------------------------------------------------------ #
#  Session runner                                                     #
# ------------------------------------------------------------------ #

def run_scenario(
    orch: DiagnosticOrchestrator,
    simulator: PatientSimulatorAgent,
    scenario: dict,
    turn_delay: float,
    verbose: bool,
) -> dict:
    t0 = time.time()

    session_id, result = orch.start_session(scenario["symptom"])

    symptom_match = result.get("symptom_match", {})
    initial_diff  = result.get("initial_differential", [])
    question      = result.get("next_question")
    q_count       = 0
    turns         = []

    while question:
        q_text  = question.get("question", "")
        test_id = question.get("test_id", "")

        if turn_delay > 0:
            time.sleep(turn_delay)

        t_turn = time.time()
        answer = simulator.answer(q_text, scenario["symptom"], scenario["name"])
        sim_elapsed = round(time.time() - t_turn, 2)

        if verbose:
            print(f"\n  Q{q_count+1:02d} [{test_id}]: {q_text}")
            print(f"       A: {answer}")

        turns.append({
            "turn":          q_count + 1,
            "test_id":       test_id,
            "question":      q_text,
            "reasoning":     question.get("reasoning", ""),
            "answer":        answer,
            "sim_elapsed_s": sim_elapsed,
        })

        result  = orch.submit_answer(session_id, answer)
        q_count += 1

        diff_after = [
            {"name": d["name"], "prob": round(d["probability"], 4)}
            for d in result.get("updated_differential", [])[:5]
        ]
        turns[-1]["differential_after"] = diff_after

        if verbose and diff_after:
            top = diff_after[0]
            print(f"       -> {top['name']} {top['prob']*100:.1f}%")

        if not result.get("should_continue") or result.get("final_diagnosis"):
            break
        question = result.get("next_question")

    final   = result.get("final_diagnosis") or {}
    elapsed = round(time.time() - t0, 1)

    primary = final.get("primary_diagnosis", "")
    correct = bool(primary) and (
        scenario["expected"].lower() in primary.lower()
        or primary.lower() in scenario["expected"].lower()
    )

    return {
        "scenario":             scenario["name"],
        "expected":             scenario["expected"],
        "symptom_text":         scenario["symptom"],
        "symptom_matched":      symptom_match.get("clinical_term", "?"),
        "match_score":          round(symptom_match.get("score", 0.0), 3),
        "initial_differential": [
            {"name": d["name"], "prob": round(d["probability"], 4)}
            for d in initial_diff[:5]
        ],
        "primary_dx":           primary,
        "confidence":           round(final.get("confidence", 0.0), 4),
        "confidence_warning":   final.get("confidence_warning", False),
        "finalization_reason":  final.get("finalization_reason", "unknown"),
        "questions_asked":      q_count,
        "turns":                turns,
        "top5_final": [
            {"name": d["name"], "prob": round(d["probability"], 4)}
            for d in final.get("differential", [])[:5]
        ],
        "correct":              correct,
        "elapsed_s":            elapsed,
    }


# ------------------------------------------------------------------ #
#  Output helpers                                                     #
# ------------------------------------------------------------------ #

BAR = "=" * 80

def _print_detail(r: dict) -> None:
    tick = "PASS" if r["correct"] else "FAIL"
    print(f"\n{'─'*80}")
    print(f"[{tick}]  {r['scenario']}")
    print(f"  Symptom   : {r['symptom_text']}")
    print(f"  HPO match : {r['symptom_matched']}  (score {r['match_score']:.3f})")
    print(f"  Expected  : {r['expected']}")
    print(f"  Diagnosed : {r['primary_dx']}  ({r['confidence'] * 100:.1f}%)")
    print(f"  Reason    : {r['finalization_reason']}  | warning={r['confidence_warning']}")
    print(f"  Questions : {r['questions_asked']}  | elapsed {r['elapsed_s']}s")

    if r["initial_differential"]:
        print("  Initial top-5:")
        for d in r["initial_differential"]:
            print(f"    {d['name']:44s} {d['prob']*100:5.1f}%")

    if r["top5_final"]:
        print("  Final top-5:")
        for d in r["top5_final"]:
            bar_len = int(d["prob"] * 28)
            bar     = "█" * bar_len + "░" * (28 - bar_len)
            print(f"    {d['name']:44s} [{bar}] {d['prob']*100:5.1f}%")

    if r["turns"]:
        print("  Q&A trace:")
        for t in r["turns"]:
            print(f"    T{t['turn']:02d}  [{t['test_id']}]  Q: {t['question'][:90]}")
            print(f"          A: {t['answer'][:100]}")
            if t.get("differential_after"):
                top2 = t["differential_after"][:2]
                summary = "  |  ".join(f"{d['name']} {d['prob']*100:.1f}%" for d in top2)
                print(f"          -> {summary}")


def print_summary(results: list) -> None:
    print(f"\n{BAR}")
    print("  ZIVAK — Neo4j Rare-Disease Diagnostic Evaluation")
    print(BAR)
    header = f"  {'Scenario':<22} {'Expected':<32} {'Diagnosed':<32} {'Conf':>5} {'Qs':>3}  OK?"
    print(f"\n{header}")
    print("  " + "─" * 78)
    for r in results:
        tick     = "PASS" if r["correct"] else "FAIL"
        exp_s    = r["expected"][:30]
        dx_s     = (r["primary_dx"] or "—")[:30]
        warn_tag = " !" if r.get("confidence_warning") else "  "
        print(
            f"  {r['scenario']:<22} {exp_s:<32} {dx_s:<32}"
            f" {r['confidence']*100:4.0f}%{warn_tag} {r['questions_asked']:>3}   {tick}"
        )

    correct  = sum(1 for r in results if r["correct"])
    warn     = sum(1 for r in results if r.get("confidence_warning"))
    errors   = sum(1 for r in results if "ERROR" in (r.get("primary_dx") or ""))
    avg_qs   = sum(r["questions_asked"] for r in results) / max(len(results), 1)
    print(f"\n  {'─'*78}")
    print(f"  Accuracy : {correct}/{len(results)}  ({correct/max(len(results),1)*100:.0f}%)")
    print(f"  Avg Qs   : {avg_qs:.1f}")
    print(f"  Warnings : {warn}/{len(results)}  (low-confidence terminations)")
    print(f"  Errors   : {errors}")
    print(f"{BAR}\n")


def save_log(results: list, log_dir: Path, simulator_model: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    path    = log_dir / f"run_{ts}.json"
    correct = sum(1 for r in results if r["correct"])
    avg_qs  = sum(r["questions_asked"] for r in results) / max(len(results), 1)
    avg_conf = sum(r["confidence"] for r in results) / max(len(results), 1)

    payload = {
        "run_at":          ts,
        "simulator_model": simulator_model,
        "n_scenarios":     len(results),
        "n_correct":       correct,
        "accuracy_pct":    round(correct / max(len(results), 1) * 100, 1),
        "avg_questions":   round(avg_qs, 2),
        "avg_confidence":  round(avg_conf, 4),
        "n_warnings":      sum(1 for r in results if r.get("confidence_warning")),
        "results":         results,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# ------------------------------------------------------------------ #
#  Entry point                                                        #
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(description="Run ZIVAK rare-disease scenarios against live Neo4j")
    p.add_argument("--filter", nargs="*", metavar="NAME",
                   help="Run only named scenarios (case-insensitive substring match)")
    p.add_argument("--delay", type=float, default=5.0, metavar="SECS",
                   help="Pause between scenarios in seconds (default: 5)")
    p.add_argument("--turn-delay", type=float, default=1.0, metavar="SECS",
                   help="Pause between Q&A turns in seconds (default: 1)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print Q&A trace for every scenario")
    p.add_argument("--no-log", action="store_true",
                   help="Skip saving JSON log to reports/runs/")
    p.add_argument("--simulator-model", default=None, metavar="MODEL",
                   help="Groq model for patient simulator (default: llama-3.3-70b-versatile)")
    return p.parse_args()


def main():
    args = parse_args()

    scenarios = SCENARIOS
    if args.filter:
        fl        = [f.lower() for f in args.filter]
        scenarios = [s for s in SCENARIOS if any(f in s["name"].lower() for f in fl)]
        if not scenarios:
            print(f"No scenarios matched filter: {args.filter}", file=sys.stderr)
            sys.exit(1)

    print("Initialising orchestrator (Neo4j + sentence-transformer) ...")
    try:
        qdrant, neo4j = get_clients(use_mock=False)
    except Exception as exc:
        print(f"ERROR: could not connect to Neo4j — {exc}", file=sys.stderr)
        print("Check: docker compose up -d neo4j   and   .env NEO4J_PASSWORD", file=sys.stderr)
        sys.exit(1)

    orch      = DiagnosticOrchestrator(qdrant, neo4j)
    simulator = PatientSimulatorAgent(model=args.simulator_model)

    print(f"Patient simulator : groq / {simulator.model}")
    print(f"Running {len(scenarios)} scenario(s)  |  delay={args.delay}s  |  turn-delay={args.turn_delay}s\n")

    results = []
    for i, scenario in enumerate(scenarios, 1):
        if i > 1 and args.delay > 0:
            time.sleep(args.delay)

        print(f"[{i:02d}/{len(scenarios)}] {scenario['name']} ...", end=" ", flush=True)
        try:
            r = run_scenario(orch, simulator, scenario, args.turn_delay, args.verbose)
            results.append(r)
            tick = "PASS" if r["correct"] else "FAIL"
            dx_s = (r["primary_dx"] or "—")[:40]
            print(f"{tick}  dx={dx_s}  conf={r['confidence']*100:.0f}%  qs={r['questions_asked']}")
        except Exception as exc:
            import traceback
            print("ERROR")
            if args.verbose:
                traceback.print_exc()
            else:
                print(f"  {exc}", file=sys.stderr)
            results.append({
                "scenario":             scenario["name"],
                "expected":             scenario["expected"],
                "symptom_text":         scenario["symptom"],
                "symptom_matched":      "?",
                "match_score":          0.0,
                "initial_differential": [],
                "primary_dx":           f"ERROR: {exc}",
                "confidence":           0.0,
                "confidence_warning":   True,
                "finalization_reason":  "error",
                "questions_asked":      0,
                "turns":                [],
                "top5_final":           [],
                "correct":              False,
                "elapsed_s":            0.0,
            })

    if args.verbose:
        for r in results:
            _print_detail(r)

    print_summary(results)

    if not args.no_log:
        log_path = save_log(results, Path(__file__).parent / "reports" / "runs", simulator.model)
        print(f"Full log saved to: {log_path}\n")


if __name__ == "__main__":
    main()
