# Google Stitch Design Prompt — ZIVAK / ZIVAK

Paste the following prompt directly into Google Stitch.

---

## STITCH PROMPT

Design a professional medical AI web application called **ZIVAK**. This is an
agentic diagnostic assistant — it asks the patient a series of targeted questions,
updates a live probability chart after each answer, and converges on a diagnosis.
The app has three distinct screens and one persistent sidebar panel.

---

### BRAND & TONE

- Name: **ZIVAK**
- Tagline: *"Zero-Hallucination Intelligent Verification and Knowledge-grounded Assesment."*
- Tone: Clinical, precise, trustworthy — like a hospital dashboard, not a consumer health app
- Colour palette:
  - Primary: Deep navy `#0A1628`
  - Accent: Electric blue `#2563EB`
  - Success: Emerald `#10B981`
  - Warning: Amber `#F59E0B`
  - Danger: Rose `#F43F5E`
  - Background: Near-black `#0F172A`
  - Card surface: `#1E293B`
  - Text primary: `#F1F5F9`
  - Text muted: `#64748B`
- Typography: Inter for UI, JetBrains Mono for confidence scores and percentages
- Visual language: dark mode, glassmorphism cards with subtle blue glow, thin borders `#334155`

---

### SCREEN 1 — LANDING / SYMPTOM ENTRY

**Layout:** Full-screen centred card, 600px max-width

**Elements:**
- Top left: ZIVAK logo (stylised stethoscope icon + wordmark in white)
- Centre headline: "What brings you in today?" in 2.5rem, light font weight
- Subtext below: "Describe your symptoms in your own words. Be as specific as you can." in muted colour
- Large textarea (min-height 120px): placeholder "e.g. I get winded going up stairs, especially in the morning..."
  - Rounded corners, blue focus ring, subtle background
  - Character counter bottom-right (e.g. "47 / 1000")
- Primary CTA button: "Start Diagnosis →" in accent blue, full width, 52px height, rounded
- Below button: small disclaimer in muted text — "⚕ Educational tool only. Not a substitute for medical advice."
- Subtle animated background: slow-moving hexagonal mesh pattern in very dark navy

---

### SCREEN 2 — ACTIVE DIAGNOSTIC SESSION

**Layout:** Two-panel split layout. Left panel 40%, right panel 60%.

**LEFT PANEL — Live Differential**
- Header: "Live Differential" with a small pulsing green dot (live indicator)
- Below: the patient's symptom cluster match, e.g.:
  "Matched: **Exertional Dyspnoea**" with a subtle blue badge
  "Confidence: 0.85" in monospace
- Divider line
- Disease probability list — each row:
  ```
  COPD                  ████████████████░░░░  71.2%
  Asthma                ████░░░░░░░░░░░░░░░░  24.1%
  Heart Failure         █░░░░░░░░░░░░░░░░░░░   1.8%
  Anemia                ░░░░░░░░░░░░░░░░░░░░   1.1%
  ```
  - Progress bars animate smoothly when updated (CSS transition 0.6s ease)
  - Top disease row has accent blue highlight + slightly larger text
  - Percentage in monospace JetBrains Mono
  - Bars use a gradient from accent blue to emerald for the top disease
  - Other diseases use muted slate bars
- Below list: small label "Updated after each answer"
- Bottom of panel: question counter "Question 2 of ~10" with a thin progress bar

**RIGHT PANEL — Q&A Interface**
- Top: breadcrumb "Diagnosis Session > Question 2"
- Question card (glassmorphism, blue glow border):
  - Small label: "DIAGNOSTIC QUESTION" in uppercase muted
  - Question text in 1.25rem: "What is the result of Peak flow variability?"
  - Internal test ID shown very small in muted text: `test_peak_flow`
- Answer area:
  - Large textarea for patient to type their answer, min-height 100px
  - Placeholder: "Type the test result or describe your finding..."
  - Subtle suggestion chips below textarea for quick answers:
    `Normal / Negative`  `Abnormal / Positive`  `Not done`
    (clicking pre-fills the textarea)
- Submit button: "Submit Answer →" in accent blue, full width
- Below: previous Q&A history collapsed into an accordion:
  - "Previous Questions (1)" — expandable
  - Each entry shows: Q text on left, answer text on right, in a small row

---

### SCREEN 3 — FINAL DIAGNOSIS

**Layout:** Centred single column, 720px max-width

**Top section — Diagnosis banner:**
- Large card with gradient border (accent blue → emerald)
- Small label: "PRIMARY DIAGNOSIS" uppercase muted
- Disease name in 3rem bold: "COPD"
- Confidence badge: large pill — "99.7% confidence" — emerald background if >75%, amber if 50–75%, rose if <50%
- Small warning if confidence < 0.75: ⚠ "Insufficient evidence — results are indicative only"
- Horizontal divider

**Middle section — Differential breakdown:**
- Label: "Full Differential Diagnosis"
- Same bar chart as Screen 2 but now with final values, no live indicator
- Top disease bar is emerald/green
- Other bars stay slate

**Bottom section — Evidence chain:**
- Label: "Diagnostic Journey"
- Timeline list — each step:
  ```
  ● Q1  FEV1/FVC spirometry
        "FEV1/FVC 0.55 post-BD — fixed obstruction"

  ● Q2  Peak flow variability
        "<10% diurnal variability — not asthma"

  ● Q3  Chest X-ray (hyperinflation)
        "Hyperinflated, barrel chest, flattened diaphragms"
  ```
  - Left: blue circle with question number
  - Right: test name in medium weight, patient answer in muted italic below

**Footer actions:**
- "Start New Session" button (outline style)
- "Download Report" button (outline style, disabled for now — placeholder)
- Disclaimer: "⚕ ZIVAK is an educational tool. Always consult a qualified clinician."

---

### SHARED COMPONENTS

**Navigation bar (all screens except landing):**
- Left: ZIVAK logo small
- Centre: session ID shown in monospace muted text (truncated UUID)
- Right: small status pill — "● Session Active" green / "● Session Complete" blue

**Loading state:**
When waiting for the AI response (between submitting an answer and receiving the
next question), show:
- The Q&A panel dims to 50% opacity
- A pulsing "Analysing evidence..." text with three animated dots
- The differential panel shows a subtle shimmer/skeleton on the bars
- Timeout: if no response in 30s, show "Taking longer than expected — still processing"

**Error states:**
- 503 / server error: red banner at top — "Diagnostic system is unavailable. Please wait a moment and retry."
- 404 / session expired: amber banner — "Your session has expired. Please start a new diagnosis."
- Symptom unrecognised (score < 0.30): inline warning below textarea — "We couldn't match this symptom. Try using different words."

---

### RESPONSIVE BEHAVIOUR

- Desktop (>1024px): two-panel layout as described
- Tablet (768–1024px): panels stack — differential above, Q&A below
- Mobile (<768px): full-width single column, differential collapsed by default with "Show differential ↓" toggle

---

### MICRO-INTERACTIONS

- Probability bars animate smoothly on every update (ease-in-out, 600ms)
- Top disease card has a subtle blue glow pulse every 2s (CSS keyframe)
- Submit button shows a spinner while waiting for API response
- When `should_continue` flips to false, trigger a brief "confetti" burst of small
  blue/green particles (JS canvas) before sliding to the Final Diagnosis screen
- Transitions between screens: slide-left (entering session), slide-right (going back)

---

### TECH STACK HINT (for Stitch code generation)

- Framework: React 18 + TypeScript
- Styling: Tailwind CSS (dark mode config)
- State: Zustand (simple store: session_id, differential, current_question, history)
- HTTP: fetch or axios, base URL configurable via env var VITE_API_URL
- Animations: Framer Motion for screen transitions and bar chart updates
- Charts: native CSS progress bars (no heavy chart library needed)
