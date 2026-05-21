# QuestionSelectorAgent — Deep Dive Report

**File:** `agents/question_selector.py`
**Type:** LLM agent (Groq or Ollama)
**Tests:** `agents/tests/test_question_selector.py`

---

## 1. Purpose

Given the current probability distribution over diseases and a list of available
diagnostic tests, pick the **single test that will most reduce uncertainty**.

This implements the Q-Selector sub-agent from Layer 3 of the blueprint:
> "Q-Selector calls Claude with current differential; returns the single question
> maximising entropy reduction over the Neo4j subgraph."

The agent uses an LLM rather than computing Shannon entropy directly because:
- Clinical "practicality" is hard to encode mathematically (some tests are invasive)
- The LLM can reason about which tests discriminate between the TOP candidates specifically
- Future-proofing: the LLM can incorporate richer context when it arrives

---

## 2. LLM Configuration

```python
def _build_llm(groq_model, groq_temperature, groq_reasoning):
    if USE_LOCAL_LLM:
        return ChatOpenAI(
            model    = os.getenv("LOCAL_MODEL", "qwen2.5:14b"),
            temperature = groq_temperature,   # 1 for this agent
            max_tokens  = 512,
            base_url    = os.getenv("LOCAL_LLM_URL", "http://localhost:11434/v1"),
            api_key     = "ollama",
        )
    return ChatGroq(
        model            = groq_model,         # "openai/gpt-oss-120b"
        temperature      = groq_temperature,   # 1
        max_tokens       = 512,
        reasoning_effort = groq_reasoning,     # "medium"
        api_key          = os.getenv("GROQ_API_KEY"),
    )

# Called as:
QuestionSelectorAgent(model="openai/gpt-oss-120b", temperature=1)
```

**Why temperature=1?**
The agent is "creative" in the sense that it should explore different question choices.
With temperature=0 it would always pick the same test given the same input — fine for
determinism but may miss better options as context changes.

**Risk with Ollama/Qwen2.5:14b at temperature=1:** Higher variability. The model may
produce different JSON structure, add markdown code fences (` ```json ` ), or output
reasoning text before the JSON. The `_extract_json()` fallback regex handles most of
these, but malformed JSON on all 3 retries would trigger the fallback test selection.

---

## 3. Prompt Structure

```
You are ZIVAK's Question Selector Agent.

Your task: Pick the SINGLE diagnostic test that will most reduce uncertainty
in the differential diagnosis.

Current differential diagnosis:
1. COPD: 40.2%
2. Asthma: 35.1%
3. Heart Failure: 12.4%
4. Pulmonary Embolism: 7.8%
5. Pneumonia: 4.5%

Available tests:
- FEV1/FVC spirometry (ID: test_fev1): relevant for COPD, Asthma
- Peak flow variability (ID: test_peak_flow): relevant for COPD, Asthma
- BNP / NT-proBNP (ID: test_bnp): relevant for Heart Failure
- Echocardiogram (ID: test_echo): relevant for Heart Failure
- ...

Selection criteria:
1. Discriminates between top candidates (ideally tests that differ between #1 and #2)
2. Has strong likelihood ratios (changes probability significantly)
3. Is clinically practical to obtain

Output ONLY valid JSON with this exact structure:
{
  "question": "What is the result of [test name]?",
  "test_id": "test_xxx",
  "reasoning": "Brief explanation of why this test is most informative"
}

Do NOT include any text outside the JSON object.
```

### What the LLM receives
- Top 5 diseases with current probabilities (as percentages)
- ALL available tests (de-duplicated across top-5 diseases)
- Test names and which diseases they're relevant for (but NOT the actual LR values)

### What the LLM does NOT receive
- The actual likelihood ratio values (it doesn't know LR=8.5 for FEV1/COPD)
- The evidence history (it doesn't know what has already been asked)
  - NOTE: Already-asked tests are filtered out BEFORE building the prompt, so the
    LLM only sees tests that haven't been asked yet

---

## 4. select_question() — Execution Flow

```python
def select_question(differential, available_tests):
    prompt = self._build_prompt(differential, available_tests)

    for attempt in range(3):           # MAX_RETRIES = 3
        try:
            response = self.llm.invoke(prompt)
            result   = _extract_json(response.content)
            assert "question" in result
            assert "test_id"  in result
            return result
        except Exception as e:
            if attempt < 2:
                time.sleep(1.0 * (2 ** attempt))   # 1s, 2s — exponential backoff
    raise RuntimeError("QuestionSelectorAgent failed after 3 attempts")
```

### Fallback when selector returns invalid test_id

In `qa_node` (not the agent), after the agent returns:

```python
valid_ids = {t["id"] for t in available_tests}
if question.get("test_id") not in valid_ids:
    fallback = available_tests[0]   # ← FIRST test in list, not most informative
    question = {
        "test_id":   fallback["id"],
        "question":  f"What is the result of {fallback['name']}?",
        "reasoning": "fallback: selector returned a test outside the available list",
    }
```

**Issue:** `available_tests[0]` is whatever happens to be first when iterating
`get_available_tests(top_diseases)`. The order depends on `_TESTS` dict insertion
order (Python 3.7+ dict order is insertion order). For COPD leading the differential,
the first disease is COPD, and its first test is `test_fev1`. So the fallback would
be FEV1 — actually a reasonable choice. But this is coincidental, not guaranteed.

---

## 5. JSON Extraction

```python
def _extract_json(content: str) -> dict:
    try:
        return json.loads(content)            # clean JSON
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*\}', content, re.DOTALL)   # extract first {...} block
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON found: {content[:120]!r}")
```

This handles:
- Clean JSON output ✓
- JSON wrapped in markdown code fences (` ```json {...} ``` `) ✓ — regex skips fences
- Reasoning text before JSON ✓ — regex finds first `{`
- Reasoning text after JSON ✓ — regex is greedy but DOTALL, finds outermost `{...}`

**Risk:** If the LLM produces nested JSON or multiple JSON objects, the greedy `.*`
with DOTALL will match from the first `{` to the LAST `}`. This could include trailing
garbage. For the simple 3-key output expected here, this is acceptable.

---

## 6. CRITICAL BUG: Double LLM Call Per Turn

This is the most significant correctness issue in the current codebase.

### Root cause: LangGraph interrupt() re-execution

When LangGraph's `interrupt()` is called in a node, the graph pauses and saves state.
When `Command(resume=answer)` is received, LangGraph **re-runs the entire node from
the beginning**. The `interrupt()` call then immediately returns the resume value
instead of pausing.

### Consequence in qa_node

```
First execution (during start_session / after previous answer):
  1. ConfidenceJudge.should_finalize()     ← logic runs
  2. selector.select_question()            ← LLM CALL #1 → picks "test_fev1"
  3. interrupt(question)                   ← PAUSES HERE, surfaces question to user

User submits answer via submit_answer():
  → Command(resume="0.62 below 0.7")
  → LangGraph re-runs qa_node from top:

  1. ConfidenceJudge.should_finalize()     ← same result (state unchanged)
  2. selector.select_question()            ← LLM CALL #2 (WASTED) → might pick different test!
  3. interrupt(question)                   ← returns "0.62 below 0.7" immediately
  4. evaluator.evaluate(question, answer)  ← evaluates with question from CALL #2
  5. engine.update(evidence)
```

### Why this matters

**Wasted LLM calls:** Every `submit_answer()` triggers 2 selector calls, not 1.
A 10-question session costs 20 selector calls instead of 10.

**Correctness risk:** With `temperature=1`, LLM Call #2 may select a DIFFERENT test
than LLM Call #1. The user was shown the question from Call #1 ("What is the FEV1/FVC
ratio?") but the evidence evaluator receives the question from Call #2 ("What is the
bronchodilator response?"). The evaluator then tries to map the user's FEV1 answer to
bronchodilator response edges — producing incorrect evidence.

### How to fix

**Option A: Split qa_node into two nodes**
```
question_node → interrupt → answer_node
```
- `question_node`: runs selector, calls interrupt
- `answer_node`: runs evaluator and engine update
- LangGraph only re-runs `answer_node` on resume, no repeated selector call

**Option B: Cache selected question in state**
Store the selected question in the state before interrupting. On re-execution, check
if state already has a pending question and skip the selector call.

**Option C: Use LangGraph's built-in interrupt pattern**
Newer LangGraph versions have cleaner patterns for human-in-the-loop. Worth reviewing
the latest docs.

Option A (split nodes) is the cleanest architectural fix.

---

## 7. Known Issues Summary

| Issue                              | Severity | Impact                                              |
|------------------------------------|----------|-----------------------------------------------------|
| Double LLM call per turn           | CRITICAL | 2× cost, potential correctness bug with temp=1      |
| Fallback to available_tests[0]     | MEDIUM   | Suboptimal question selection on JSON parse failure |
| temperature=1 with Ollama          | MEDIUM   | Higher JSON parse failure rate than with Groq       |
| No LR values in prompt             | LOW      | LLM makes qualitative not quantitative choice       |
| Reasoning not used downstream      | LOW      | `reasoning` field returned but not shown to patient |

---

## 8. Test Coverage

| Test                    | What it validates                          | Status    |
|-------------------------|--------------------------------------------|-----------|
| test_question_selector  | LLM returns valid {question, test_id}      | Requires Ollama |

**Missing tests:**
- Fallback behaviour when test_id is not in available_tests
- All 3 retries fail → RuntimeError raised
- JSON extraction from markdown-wrapped output
- Selector picks a test not in available_tests (triggers fallback)
