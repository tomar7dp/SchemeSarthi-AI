# AI-Driven Scheme Matching — Working Prototype
### SIH26092 · Ministry of Social Justice and Empowerment

## What's in this folder

| File | Purpose |
|---|---|
| `scheme_eligibility_data.csv` | 19 real government schemes with structured eligibility data |
| `generate_dataset.py` | Generates synthetic, rule-labeled applicant training data |
| `training_data.csv` | 5,000 synthetic applicant profiles, labeled by rule engine |
| `train_model.py` | Trains the multi-label ML eligibility model |
| `model.pkl` | The trained model (already built — no need to retrain) |
| `evaluation_report.txt` | Precision/recall/F1 per scheme (for your report/slides) |
| `retriever.py` | RAG-style retrieval — pulls grounding text from scheme docs |
| `agent.py` | Orchestrator — chains missing-info check → ML model → retriever |
| `app.py` | **The Streamlit demo app you'll show your mentor** |

## How to run the demo (do this tonight, not tomorrow morning)

1. Make sure Python 3.9+ is installed.
2. Open a terminal in this folder and install dependencies:
   ```
   pip install streamlit scikit-learn pandas numpy
   ```
3. Run the app:
   ```
   streamlit run app.py
   ```
4. It opens automatically in your browser (usually `http://localhost:8501`).
5. Fill in an applicant profile on the left, click **"Find Matching Schemes"**, and you'll see ranked scheme matches with confidence scores and source-grounded explanations on the right.

**Test it tonight, not right before your presentation** — if any package fails to install, you'll want time to fix it.

## Suggested demo script for your mentor (2 minutes)

1. Open the app, point to the "How this works" expander — explain the 3-part architecture (ML model, RAG, agent) in one sentence each.
2. Fill in a profile: e.g. Age 28, Category SC, State Bihar, Education Class 10, Income ₹80,000, Sector Manufacturing.
3. Click "Find Matching Schemes" — walk through the top result: point out the confidence score, then expand "Why this match" to show the RAG-grounded citation pulled from the actual scheme text.
4. Change the profile to something very different (e.g. Category General, Sector Tech Startup, tick "registered company") and re-run to show the system adapts — SISFS should now appear since it's the entity-based scheme.
5. Mention the evaluation number: **macro F1 of 0.83** on held-out synthetic test data, and be upfront that labels are rule-derived (no public historical approval data exists) — mentors respect this kind of honesty.

## Known limitations to mention proactively (don't wait to be asked)

- **Synthetic training labels**, not real historical outcomes — derived from published eligibility criteria
- A few state-specific schemes (Maharashtra CMEGP, Tamil Nadu NEEDS) have very few positive examples in the training data because their eligibility windows (state + age + education combined) are narrow by design — the model is less reliable on these specifically
- The RAG layer uses TF-IDF retrieval rather than neural embeddings, for speed and simplicity in this prototype — swapping in sentence-transformers + FAISS is a natural next step
- A couple of scheme statuses (Mahila Udyam Nidhi, the Gujarat state scheme) were flagged during research as needing verification against official portals for current validity

## If Streamlit won't run for some reason

As a fallback, you can demo `agent.py` directly from the terminal:
```
python3 agent.py
```
This prints a sample match result to the console — not as polished, but proves the pipeline works if the UI has issues.
