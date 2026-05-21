# ZIVAK API — Complete Reference

**Base URL:** `http://localhost:8000`  
**Auth:** All diagnostic endpoints require `Authorization: Bearer <token>`  
**Dev token:** `zivak-dev-key` (set in `.env` as `API_KEY`)  
**Content-Type:** `application/json` on all POST requests

---

## API STATUS

✅ All 10 endpoint tests pass (verified with mock suite)  
⚠️ Live LLM calls require Ollama running without GPU conflicts  
   → If you get 503, restart Ollama: `ollama serve` in a separate terminal  
   → Ollama must be running BEFORE the FastAPI server starts

---

## ENDPOINT 1 — Health Check

```
GET /health
```

No auth required. Use this to check the server is alive before any session work.

**Request**
```
GET /health HTTP/1.1
Host: localhost:8000
```

**Response 200**
```json
{
  "status": "ok"
}
```

---

## ENDPOINT 2 — Start Diagnostic Session

```
POST /api/v1/sessions
Authorization: Bearer zivak-dev-key
Content-Type: application/json
```

Give the patient's symptom in plain language. The system maps it to a clinical
concept, seeds a Bayesian differential, and returns the first diagnostic question.

**Request body**
```json
{
  "symptom": "I get winded very easily, especially going upstairs"
}
```

| Field     | Type   | Required | Constraints          | Notes                               |
|-----------|--------|----------|----------------------|-------------------------------------|
| `symptom` | string | yes      | min 3, max 1000 chars| Plain language, patient's own words |

**Response 200 — Session started**
```json
{
  "session_id": "a3f1c2d4-e5b6-7890-abcd-ef1234567890",
  "symptom_match": {
    "clinical_term": "Exertional Dyspnoea",
    "symptom_id": "HP:0002875",
    "score": 0.85
  },
  "initial_differential": [
    { "name": "Asthma",        "probability": 0.275, "log_prob": -1.291, "specificity": 0.80, "prevalence": 0.080, "raw_score": 0.064 },
    { "name": "COPD",          "probability": 0.241, "log_prob": -1.422, "specificity": 0.85, "prevalence": 0.065, "raw_score": 0.055 },
    { "name": "Anemia",        "probability": 0.105, "log_prob": -2.254 },
    { "name": "Heart Failure", "probability": 0.105, "log_prob": -2.254 },
    { "name": "Pneumonia",     "probability": 0.043, "log_prob": -3.147 },
    { "name": "Pulmonary Embolism", "probability": 0.028, "log_prob": -3.575 },
    { "name": "Interstitial Lung Disease", "probability": 0.020, "log_prob": -3.912 }
  ],
  "next_question": {
    "question": "What is the result of FEV1/FVC spirometry?",
    "test_id": "test_fev1",
    "reasoning": "FEV1/FVC distinguishes obstructive (COPD/Asthma) from non-obstructive causes"
  }
}
```

| Field                | Type         | Notes                                                            |
|----------------------|--------------|------------------------------------------------------------------|
| `session_id`         | UUID string  | **Store this.** Required for all subsequent calls               |
| `symptom_match`      | object       | Which clinical cluster was matched and how confidently           |
| `symptom_match.score`| float 0–1    | Cosine similarity. Below 0.30 = unrecognised symptom            |
| `initial_differential`| array       | All candidate diseases with starting probabilities, sorted desc  |
| `next_question`      | object\|null | The first question to show the patient. Null if finalised immediately |
| `next_question.test_id` | string    | Internal test ID — use to look up answer in your own logic       |

**Errors**
| Status | When |
|--------|------|
| 401    | Missing or wrong Bearer token |
| 422    | `symptom` field missing, too short (<3), or too long (>1000) |
| 503    | LLM/Orchestrator unavailable (Ollama not running) |

---

## ENDPOINT 3 — Submit Answer

```
POST /api/v1/sessions/{session_id}/answer
Authorization: Bearer zivak-dev-key
Content-Type: application/json
```

Submit the patient's answer to the current question. The system evaluates the
evidence, updates the Bayesian differential, and either returns the next question
or issues a final diagnosis.

**Path parameter**
| Param        | Type   | Notes                                          |
|--------------|--------|------------------------------------------------|
| `session_id` | string | The UUID returned by `POST /api/v1/sessions`  |

**Request body**
```json
{
  "answer": "FEV1/FVC 0.55 post-bronchodilator — fixed obstruction pattern"
}
```

| Field    | Type   | Required | Constraints          |
|----------|--------|----------|----------------------|
| `answer` | string | yes      | min 1, max 1000 chars|

**Response 200 — Session continuing (more questions)**
```json
{
  "updated_differential": [
    { "name": "COPD",          "probability": 0.712, "log_prob": -0.340 },
    { "name": "Asthma",        "probability": 0.241, "log_prob": -1.422 },
    { "name": "Heart Failure", "probability": 0.018, "log_prob": -4.017 },
    { "name": "Anemia",        "probability": 0.011, "log_prob": -4.510 },
    { "name": "Pneumonia",     "probability": 0.009, "log_prob": -4.711 },
    { "name": "Pulmonary Embolism", "probability": 0.006, "log_prob": -5.116 },
    { "name": "Interstitial Lung Disease", "probability": 0.003, "log_prob": -5.809 }
  ],
  "should_continue": true,
  "next_question": {
    "question": "What is the result of Peak flow variability?",
    "test_id": "test_peak_flow",
    "reasoning": "Differentiates fixed COPD obstruction from variable asthma obstruction"
  },
  "judge_details": {
    "should_finalize": false,
    "evidence_count": 1,
    "min_required": 4,
    "top_disease": "COPD",
    "top_probability": 0.712,
    "runner_up_disease": "Asthma",
    "runner_up_probability": 0.241,
    "criteria_met": {
      "min_evidence": false,
      "top_confidence": false,
      "runner_up_low": true
    },
    "reason": "Need more evidence (1/4)"
  },
  "final_diagnosis": null
}
```

**Response 200 — Session complete (final diagnosis)**
```json
{
  "updated_differential": [
    { "name": "COPD",    "probability": 0.997, "log_prob": -0.003 },
    { "name": "Asthma",  "probability": 0.003, "log_prob": -5.809 },
    { "name": "Heart Failure", "probability": 0.000 }
  ],
  "should_continue": false,
  "next_question": null,
  "judge_details": {
    "should_finalize": true,
    "reason": "All criteria met - ready to finalize"
  },
  "final_diagnosis": {
    "primary_diagnosis": "COPD",
    "confidence": 0.997,
    "differential": [
      { "name": "COPD",         "probability": 0.997 },
      { "name": "Asthma",       "probability": 0.003 },
      { "name": "Heart Failure","probability": 0.000 }
    ],
    "evidence_chain": [
      { "question": "What is the result of FEV1/FVC spirometry?",    "reasoning": "..." },
      { "question": "What is the result of Peak flow variability?",  "reasoning": "..." },
      { "question": "What is the result of Chest X-ray (hyperinflation)?", "reasoning": "..." },
      { "question": "What is the result of Full blood count?",       "reasoning": "..." }
    ],
    "total_questions": 4
  }
}
```

| Field                | Type          | Notes                                                              |
|----------------------|---------------|--------------------------------------------------------------------|
| `updated_differential`| array        | All diseases re-sorted by probability after this answer            |
| `should_continue`    | boolean       | **Primary loop control.** True = show next_question. False = done |
| `next_question`      | object\|null  | Next question to show. Null when `should_continue` is false        |
| `judge_details`      | object        | Debug info — confidence gate state, criteria met/unmet             |
| `final_diagnosis`    | object\|null  | Populated only when `should_continue` is false                     |
| `final_diagnosis.confidence` | float | 0–1. Values below 0.75 indicate insufficient evidence             |
| `final_diagnosis.evidence_chain` | array | Questions asked and selector reasoning (not clinical explanation yet) |

**Errors**
| Status | When |
|--------|------|
| 401    | Wrong Bearer token |
| 404    | `session_id` not found or expired (session state lost on server restart) |
| 422    | `answer` field missing or empty |
| 503    | LLM/Orchestrator unavailable |

---

## ENDPOINT 4 — Get Final Diagnosis

```
GET /api/v1/sessions/{session_id}/diagnosis
Authorization: Bearer zivak-dev-key
```

Retrieve the completed diagnosis for a finished session. Use this if you need
to re-fetch the final report after `should_continue` became false.

**Path parameter**
| Param        | Type   |
|--------------|--------|
| `session_id` | string |

**Response 200 — Diagnosis available**
```json
{
  "primary_diagnosis": "COPD",
  "confidence": 0.997,
  "differential": [
    { "name": "COPD",         "probability": 0.997 },
    { "name": "Asthma",       "probability": 0.003 },
    { "name": "Heart Failure","probability": 0.000 }
  ],
  "evidence_chain": [
    { "question": "What is the result of FEV1/FVC spirometry?",    "reasoning": "..." },
    { "question": "What is the result of Peak flow variability?",  "reasoning": "..." }
  ],
  "total_questions": 4
}
```

**Errors**
| Status | When |
|--------|------|
| 401    | Wrong Bearer token |
| 404    | Session not found |
| 409    | Session exists but has not finalised yet (still mid-session) |

---

## COMPLETE SESSION FLOW

```
┌─────────────────────────────────────────────────────────────┐
│  1. POST /api/v1/sessions                                    │
│     Body: { "symptom": "..." }                               │
│     → Save session_id                                        │
│     → Show initial_differential (probability bars)           │
│     → Show next_question.question to patient                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  LOOP while should_continue == true                          │
│                                                              │
│  2. Patient answers current question                         │
│                                                              │
│  3. POST /api/v1/sessions/{session_id}/answer                │
│     Body: { "answer": "patient's answer text" }              │
│     → Update probability bars (updated_differential)         │
│     → if should_continue == true:                            │
│         Show next_question.question to patient               │
│     → if should_continue == false:                           │
│         Show final_diagnosis                                 │
│         Break loop                                           │
└──────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  4. (Optional) GET /api/v1/sessions/{session_id}/diagnosis   │
│     Fetch the saved final report any time after completion   │
└─────────────────────────────────────────────────────────────┘
```

---

## KNOWN BEHAVIOURS TO HANDLE ON THE FRONTEND

| Behaviour | When | How to handle |
|-----------|------|---------------|
| `symptom_match.score < 0.30` | Unrecognised symptom | Show "We couldn't match your symptom, try rephrasing" |
| `should_continue == false` AND `confidence < 0.75` | Low-evidence forced stop | Show warning: "Insufficient evidence — results are indicative only" |
| `final_diagnosis == null` AND `should_continue == false` | Edge case / empty differential | Show "Unable to determine diagnosis — please consult a clinician" |
| 503 response | Ollama/LLM down | Show "Diagnostic system is warming up, please wait and retry" |
| 404 on submit answer | Session expired (server restarted) | Prompt to start a new session |
| Max ~10 questions | Hard cap in engine | Session will self-terminate |

---

## OLLAMA / LLM NOTE FOR DEVELOPMENT

The 503 error appears when Ollama's GPU process crashes (CUDA conflict between
two processes using the GPU). To prevent it:

1. Start Ollama in its own terminal first: `ollama serve`  
2. Then start the FastAPI server: `uvicorn main:app --reload`  
3. If you get a 503, stop both, restart Ollama, then restart the server

For production: use `USE_LOCAL_LLM=false` and set `GROQ_API_KEY` in `.env` — the
Groq cloud API is more reliable than local GPU inference for concurrent requests.
