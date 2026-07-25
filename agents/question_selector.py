"""
Question Selector Agent - Uses LLM to pick best diagnostic question
Requires: GROQ_API_KEY_SELECTOR in .env
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


def _build_llm(model: str, temperature: float):
    """Return a ChatGoogleGenerativeAI LLM for the Question Selector."""
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=model,
        temperature=temperature,
        max_tokens=2048,
        reasoning_format="hidden",
        api_key=os.getenv("GROQ_API_KEY_SELECTOR"),
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

    def __init__(self, model: str = "openai/gpt-oss-20b", temperature: float = 1):
        self.llm = _build_llm(model, temperature)

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

        # Tests have already been pre-ranked by Expected Information Gain in Python.
        # The LLM's job here is purely: pick the most clinically practical one and
        # phrase the question naturally for a patient.
        tests_text = "\n".join(
            f"- {t['name']} (ID: {t['id']})"
            for t in available_tests
        )

        return f"""You are ZIVAK's Question Selector Agent.

These tests have been pre-ranked by diagnostic information value (highest first).
Your task: pick the most clinically practical one and phrase it as a clear, natural question for a patient.

{context_block}Current differential diagnosis:
{diff_text}

Pre-ranked candidate tests (ask the most practical one from this list):
{tests_text}

Selection criteria:
1. Prefer tests higher in the list (they discriminate better between the leading diagnoses)
2. Pick the least invasive / most accessible option when information value is similar
3. Phrase the question in plain language the patient will understand

Output ONLY valid JSON with this exact structure:
{{
  "question": "Natural language question to ask the patient",
  "test_id": "HP:xxxxxxx",
  "reasoning": "One sentence: why this test, why this phrasing"
}}

Do NOT include any text outside the JSON object.
"""
