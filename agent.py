"""
agent.py

The agentic orchestration layer. Given an applicant profile, this:
  1. Checks whether enough information was provided (if not, flags
     what's missing -- in the Streamlit app this becomes a follow-up
     prompt).
  2. Calls the trained ML model to score/predict eligibility across
     all schemes.
  3. For each predicted-eligible scheme, calls the retriever to pull
     the grounding clause explaining WHY (RAG step).
  4. Returns a structured result the UI can render.

This is intentionally a simple, transparent state machine rather than
a free-form LLM agent loop -- for a prototype demo, a deterministic
orchestrator is easier to explain, debug, and trust than an LLM
deciding tool calls on its own, and it still demonstrates the core
agentic idea: multiple tools (model, retriever) called in sequence
to reach a goal, with a decision point (missing info) in between.
"""

import pickle
import numpy as np
import pandas as pd
from retriever import SchemeRetriever

REQUIRED_FIELDS = ["age", "category", "state", "education",
                    "annual_family_income_inr", "sector"]


class SchemeMatchingAgent:
    def __init__(self, model_path="model.pkl", schemes_csv="scheme_eligibility_data.csv"):
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        self.model = bundle["model"]
        self.encoder = bundle["encoder"]
        self.scheme_names = bundle["scheme_names"]
        self.num_cols = bundle["feature_cols_numeric"]
        self.cat_cols = bundle["feature_cols_categorical"]
        self.bool_cols = bundle["feature_cols_bool"]
        self.retriever = SchemeRetriever(schemes_csv)
        self.schemes_df = pd.read_csv(schemes_csv)

    def check_missing_fields(self, profile: dict):
        missing = [f for f in REQUIRED_FIELDS if profile.get(f) in (None, "", "Select...")]
        return missing

    def _profile_to_features(self, profile):
        row = {
            "age": profile["age"],
            "annual_family_income_inr": profile["annual_family_income_inr"],
            "category": profile["category"],
            "state": profile["state"],
            "education": profile["education"],
            "sector": profile["sector"],
            "is_registered_company": profile.get("is_registered_company", False),
            "family_member_already_availed_scheme": profile.get(
                "family_member_already_availed_scheme", False),
        }
        df_row = pd.DataFrame([row])

        cat_features = self.encoder.transform(df_row[self.cat_cols])
        num_features = df_row[self.num_cols].values
        bool_features = df_row[self.bool_cols].astype(int).values
        X = np.hstack([num_features, bool_features, cat_features])
        return X

    def match(self, profile: dict, min_confidence=0.4):
        """
        Returns a list of dicts: scheme_name, confidence, grounding_clause,
        sorted by confidence descending. Only schemes above min_confidence
        are returned.
        """
        missing = self.check_missing_fields(profile)
        if missing:
            return {"status": "incomplete", "missing_fields": missing, "matches": []}

        X = self._profile_to_features(profile)

        # predict_proba for MultiOutputClassifier returns a list (one per
        # label) of arrays shaped (n_samples, n_classes). We want P(class=1).
        proba_per_scheme = self.model.predict_proba(X)
        confidences = []
        for i, name in enumerate(self.scheme_names):
            classes = self.model.estimators_[i].classes_
            if 1 in classes:
                idx = list(classes).index(1)
                conf = proba_per_scheme[i][0][idx]
            else:
                conf = 0.0
            confidences.append((name, conf))

        confidences.sort(key=lambda x: x[1], reverse=True)

        matches = []
        query = (f"eligibility for {profile['category']} category, "
                 f"{profile['sector']} sector, age {profile['age']}")
        for name, conf in confidences:
            if conf < min_confidence:
                continue
            clause = self.retriever.retrieve_clause(name, query)
            scheme_row = self.retriever.get_scheme_row(name)
            matches.append({
                "scheme_name": name,
                "confidence": round(float(conf), 3),
                "grounding_clause": clause,
                "ministry": scheme_row["ministry_body"] if scheme_row is not None else "",
                "loan_amount": scheme_row["loan_subsidy_amount"] if scheme_row is not None else "",
                "status_flag": scheme_row["status"] if scheme_row is not None else "",
            })

        return {"status": "ok", "missing_fields": [], "matches": matches}


if __name__ == "__main__":
    agent = SchemeMatchingAgent()

    # Test profile: SC applicant, rural, low income, manufacturing sector
    test_profile = {
        "age": 28,
        "category": "SC",
        "state": "Bihar",
        "education": "Class 10",
        "annual_family_income_inr": 80000,
        "sector": "Manufacturing",
        "is_registered_company": False,
        "family_member_already_availed_scheme": False,
    }

    result = agent.match(test_profile)
    print(f"Status: {result['status']}")
    for m in result["matches"]:
        print(f"\n{m['scheme_name']}  (confidence: {m['confidence']})")
        print(f"  Ministry: {m['ministry']}")
        print(f"  Loan/Subsidy: {m['loan_amount']}")
        print(f"  Grounding: {m['grounding_clause'][:150]}...")
