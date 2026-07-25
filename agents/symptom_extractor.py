import json
import logging
import os
import re
import time
from typing import List

from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

def _extract_json(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON object found: {content[:120]}")

class SymptomExtractorAgent:
    """
    LLM agent that translates conversational patient language into standard 
    clinical terminology before vector search.
    """
    def __init__(self, model: str = "llama-3.1-8b-instant", temperature: float = 0.0):
        self.llm = ChatGroq(
            model=model,
            temperature=temperature,
            max_tokens=150,
            api_key=os.getenv("GROQ_API_KEY_SELECTOR"),
        )

    def extract_symptoms(self, user_input: str) -> List[str]:
        prompt = f"""You are a clinical NLP engine. Extract the main medical symptoms from the patient's statement.
Translate colloquial or layperson terms into short, standard clinical terminology suitable for a medical database (e.g. 'winded' -> 'exertional dyspnea', 'heart racing' -> 'palpitations').
If multiple distinct symptoms are mentioned, list them all.

Patient statement: "{user_input}"

Output ONLY a valid JSON object with a single key 'clinical_terms' containing a list of strings. Do not explain.
"""
        for attempt in range(3):
            try:
                resp = self.llm.invoke(prompt)
                data = _extract_json(resp.content)
                terms = data.get("clinical_terms", [])
                if terms and isinstance(terms, list):
                    return terms
            except Exception as e:
                logger.warning("SymptomExtractor attempt %d failed: %s", attempt + 1, e)
                time.sleep(1)
                
        logger.error("SymptomExtractor failed completely, falling back to raw input.")
        return []
