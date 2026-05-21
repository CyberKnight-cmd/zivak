"""
Question Selector Agent - Uses LLM to pick best diagnostic question
Requires: GROQ_API_KEY in .env
"""

import json
import logging
import os
import re
import time
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

MAX_RETRIES  = 3
BASE_BACKOFF = 1.0  # seconds — doubles each attempt: 1s, 2s, 4s


def _build_llm(groq_model: str, groq_temperature: float, groq_reasoning: str):
    """Return a ChatGroq or local OpenAI-compatible LLM based on USE_LOCAL_LLM env var."""
    if os.getenv("USE_LOCAL_LLM", "").lower() == "true":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("LOCAL_MODEL", "qwen2.5:14b"),
            temperature=groq_temperature,
            max_tokens=512,
            base_url=os.getenv("LOCAL_LLM_URL", "http://localhost:11434/v1"),
            api_key="ollama",
        )
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=groq_model,
        temperature=groq_temperature,
        max_tokens=512,
        reasoning_effort=groq_reasoning,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def _extract_json(content: str) -> dict:
    """Parse JSON from LLM output that may contain reasoning text before/after the object."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON object found in response: {content[:120]!r}")


class QuestionSelectorAgent:
    """
    LLM agent that selects the most informative diagnostic question.

    Retries up to MAX_RETRIES times with exponential backoff on any failure.
    Raises RuntimeError if all attempts are exhausted — never silently falls back,
    because a silently wrong question is worse than a visible error.
    """

    def __init__(self, model: str = "openai/gpt-oss-120b", temperature: float = 1):
        self.llm = _build_llm(model, temperature, groq_reasoning="medium")

    def select_question(
        self,
        differential: List[Dict],
        available_tests: List[Dict],
        test_lr_map: Optional[Dict[str, List[Dict]]] = None,
        symptom: Optional[str] = None,
        qa_history: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Select the best question to ask next.

        Args:
            differential:    Current disease probabilities (sorted descending).
            available_tests: Tests not yet asked; each has id, name, diseases.
            test_lr_map:     Optional mapping of test_id → LR edge list from Neo4j.
                             When provided, LR magnitudes are shown in the prompt so
                             the LLM can prefer high-information tests (N1 fix).
            symptom:         Original patient complaint, shown as context (N4 fix).
            qa_history:      Prior Q&A turns as list of dicts with 'question' and
                             optional 'answer' keys (N4 fix).

        Raises:
            RuntimeError: if all retry attempts fail.
        """
        prompt   = self._build_prompt(differential, available_tests, test_lr_map, symptom, qa_history)
        last_exc: Exception = None

        for attempt in range(MAX_RETRIES):
            try:
                response = self.llm.invoke(prompt)
                result   = _extract_json(response.content)
                assert "question" in result, "missing 'question' field"
                assert "test_id"  in result, "missing 'test_id' field"
                return result

            except Exception as e:
                last_exc = e
                if attempt < MAX_RETRIES - 1:
                    wait = BASE_BACKOFF * (2 ** attempt)
                    logger.warning(
                        "QuestionSelector attempt %d/%d failed (%s) — retrying in %.0fs",
                        attempt + 1, MAX_RETRIES, e, wait,
                    )
                    time.sleep(wait)

        logger.error("QuestionSelector failed after %d attempts: %s", MAX_RETRIES, last_exc)
        raise RuntimeError(
            f"QuestionSelectorAgent failed after {MAX_RETRIES} attempts"
        ) from last_exc

    def _build_prompt(
        self,
        differential: List[Dict],
        available_tests: List[Dict],
        test_lr_map: Optional[Dict[str, List[Dict]]] = None,
        symptom: Optional[str] = None,
        qa_history: Optional[List[Dict]] = None,
    ) -> str:
        # --- Context block (symptom + history) ---
        context_block = ""
        if symptom:
            context_block += f"Patient presenting complaint: {symptom!r}\n\n"
        if qa_history:
            context_block += "Evidence collected so far:\n"
            for i, qa in enumerate(qa_history, 1):
                answer_part = f": {qa['answer']}" if qa.get("answer") else ""
                context_block += f"  Turn {i} — {qa['question']}{answer_part}\n"
            context_block += "\n"

        # --- Differential block ---
        diff_text = "\n".join(
            f"{i+1}. {d['name']}: {d['probability']*100:.1f}%"
            for i, d in enumerate(differential[:5])
        )

        # --- Tests block ---
        # When LR data is available, show numeric LRs so the LLM can distinguish
        # test_hrct (ILD LR≈20.0) from test_cxr_hyp (COPD LR≈4.2).
        # Only RULES_IN LRs are shown — they are the most actionable for selection.
        def _fmt_test(t: Dict) -> str:
            if test_lr_map:
                edges = test_lr_map.get(t["id"], [])
                lr_parts = [
                    f"{e['disease']} LR≈{e['lr']}"
                    for e in edges
                    if e["relationship"] == "RULES_IN"
                ]
                # Fallback to disease list if no RULES_IN edges found for this test
                if lr_parts:
                    return f"- {t['name']} (ID: {t['id']}): {', '.join(lr_parts)}"
            return f"- {t['name']} (ID: {t['id']}): relevant for {', '.join(t['diseases'])}"

        has_lr = bool(test_lr_map)
        tests_header = (
            "Available tests (LR values show how strongly a positive result shifts probability):"
            if has_lr else
            "Available tests:"
        )
        tests_text = "\n".join(_fmt_test(t) for t in available_tests)

        lr_criterion = (
            "2. Prefer tests with higher LR values — they produce larger probability shifts"
            if has_lr else
            "2. Has strong likelihood ratios (changes probability significantly)"
        )

        return f"""You are ZIVAK's Question Selector Agent.

Your task: Pick the SINGLE diagnostic test that will most reduce uncertainty in the differential diagnosis.

{context_block}Current differential diagnosis:
{diff_text}

{tests_header}
{tests_text}

Selection criteria:
1. Discriminates between top candidates (ideally tests that differ between #1 and #2)
{lr_criterion}
3. Is clinically practical to obtain

Output ONLY valid JSON with this exact structure:
{{
  "question": "What is the result of [test name]?",
  "test_id": "test_xxx",
  "reasoning": "Brief explanation of why this test is most informative"
}}

Do NOT include any text outside the JSON object.
"""
