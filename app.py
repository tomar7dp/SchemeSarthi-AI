"""
app.py

Interactive Streamlit demo for the AI-Driven Scheme Matching prototype.

RUN THIS WITH:
    streamlit run app.py

This is the live demo you show your mentor: fill in an applicant
profile on the left, hit "Find Matching Schemes", and see the ML
model's matches with confidence scores and RAG-grounded citations
on the right.
"""

import streamlit as st
from agent import SchemeMatchingAgent, REQUIRED_FIELDS
from retriever import SchemeRetriever
import pandas as pd

st.set_page_config(page_title="AI Scheme Matcher", page_icon="🇮🇳", layout="wide")


@st.cache_resource
def load_agent():
    return SchemeMatchingAgent()


agent = load_agent()

st.title("🇮🇳 AI-Driven Scheme Matching for Marginalized Entrepreneurs")
st.caption("SIH26092 · Ministry of Social Justice and Empowerment — Prototype Demo")

with st.expander("ℹ️ How this works (for demo narration)"):
    st.markdown("""
    This prototype combines three techniques:
    - **ML Model** — a trained multi-label classifier scores your eligibility across 19 real government schemes
    - **RAG (Retrieval-Augmented Generation)** — every match is grounded in the actual scheme text, not a guess
    - **Agentic orchestration** — the system checks for missing info before scoring, then chains the model and retriever together

    Trained on a synthetic, rule-derived dataset (5,000 profiles) since historical approval data isn't public.
    Macro-average F1 score on held-out test data: **0.83**
    """)

col1, col2 = st.columns([1, 1.4])

schemes_df = pd.read_csv("scheme_eligibility_data.csv")

with col1:
    st.subheader("Applicant Profile")

    age = st.slider("Age", 16, 70, 28)
    category = st.selectbox("Category", [
        "General", "SC", "ST", "OBC", "Women", "Disabled",
        "ExServicemen", "Minority", "Transgender"
    ])
    state = st.selectbox("State", [
        "Andhra Pradesh", "Bihar", "Gujarat", "Karnataka", "Kerala",
        "Madhya Pradesh", "Maharashtra", "Punjab", "Rajasthan",
        "Tamil Nadu", "Uttar Pradesh", "West Bengal", "Delhi", "Odisha",
        "Sikkim", "Arunachal Pradesh", "Assam", "Meghalaya",
        "Nagaland", "Manipur", "Tripura", "Mizoram",
    ])
    education = st.selectbox("Education Level", [
        "Below Class 5", "Class 5", "Class 8", "Class 10",
        "Class 12", "ITI/Vocational", "Diploma", "Degree"
    ], index=3)
    income = st.number_input("Annual Family Income (₹)", min_value=0,
                              max_value=5000000, value=150000, step=10000)
    sector = st.selectbox("Business Sector / Type", [
        "Manufacturing", "Service", "Trading", "Agriculture",
        "Fisheries", "Artisan/Traditional Trade", "Street Vending",
        "Tech Startup"
    ])
    is_company = st.checkbox("This is a registered company/startup (not an individual applicant)")

    find_button = st.button("🔍 Find Matching Schemes", type="primary", use_container_width=True)

with col2:
    st.subheader("Matched Schemes")

    if find_button:
        profile = {
            "age": age,
            "category": category,
            "state": state,
            "education": education,
            "annual_family_income_inr": income,
            "sector": sector,
            "is_registered_company": is_company,
            "family_member_already_availed_scheme": False,
        }

        with st.spinner("Scoring eligibility and retrieving grounding clauses..."):
            result = agent.match(profile, min_confidence=0.35)

        if result["status"] == "incomplete":
            st.warning(f"Missing information: {', '.join(result['missing_fields'])}")
        elif not result["matches"]:
            st.info("No strong scheme matches found for this profile above the confidence threshold. "
                     "Try adjusting the profile, or lower the confidence bar in the code for exploration.")
        else:
            st.success(f"Found {len(result['matches'])} matching scheme(s)")
            for m in result["matches"]:
                with st.container(border=True):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"### {m['scheme_name']}")
                        st.caption(m["ministry"])
                    with c2:
                        st.metric("Match Confidence", f"{m['confidence']*100:.0f}%")

                    st.markdown(f"**💰 Loan/Subsidy:** {m['loan_amount']}")

                    if "verify" in str(m["status_flag"]).lower():
                        st.warning(f"⚠️ Status: {m['status_flag']}")

                    with st.expander("📄 Why this match (source-grounded)"):
                        st.write(m["grounding_clause"])
    else:
        st.info("👈 Fill in the applicant profile and click 'Find Matching Schemes' to see results.")

st.divider()
st.caption(
    "Prototype for academic demonstration. Eligibility rules are derived from public scheme "
    "documentation and may not reflect the most current guidelines — always verify on official "
    "portals (myscheme.gov.in) before applying."
)
