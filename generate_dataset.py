"""
generate_dataset.py

Builds a synthetic, rule-labeled applicant dataset for the
AI-Driven Scheme Matching project (SIH26092).

WHAT THIS DOES
--------------
1. Loads scheme_eligibility_data.csv (the 19 schemes you collected).
2. Generates N synthetic applicant profiles with randomized but
   realistic attributes (age, category, state, education, income,
   sector, entity type).
3. For each profile, applies EXPLICIT RULE-BASED LOGIC (not ML) to
   decide which schemes the applicant is eligible for. This becomes
   your ground-truth label.
4. Saves the result as training_data.csv — one row per applicant,
   with a binary eligible/not-eligible column per scheme.

WHY RULE-BASED LABELING, NOT REAL DATA
---------------------------------------
Real government approval/rejection data isn't public. So the labels
here are DERIVED from the eligibility criteria you already extracted,
not from historical outcomes. Be explicit about this in your report:
the model you train downstream is learning to approximate written
eligibility rules, not to predict real-world approval behavior.
This is still a legitimate ML task (it's a genuine multi-label
classification problem with real structure), but the distinction
matters for how you frame your evaluation section.

HOW TO USE
----------
    python generate_dataset.py --n 800 --seed 42

Produces training_data.csv in the same folder.
"""

import argparse
import random
import pandas as pd


NER_STATES = ["Sikkim", "Arunachal Pradesh", "Assam", "Meghalaya",
              "Nagaland", "Manipur", "Tripura", "Mizoram"]

ALL_STATES = [
    "Andhra Pradesh", "Bihar", "Gujarat", "Karnataka", "Kerala",
    "Madhya Pradesh", "Maharashtra", "Punjab", "Rajasthan",
    "Tamil Nadu", "Uttar Pradesh", "West Bengal", "Delhi", "Odisha",
] + NER_STATES

CATEGORIES = ["General", "SC", "ST", "OBC", "Women", "Disabled",
              "ExServicemen", "Minority", "Transgender"]

SECTORS = ["Manufacturing", "Service", "Trading", "Agriculture",
           "Fisheries", "Artisan/Traditional Trade", "Street Vending",
           "Tech Startup"]

EDUCATION_LEVELS = ["Below Class 5", "Class 5", "Class 8", "Class 10",
                     "Class 12", "ITI/Vocational", "Diploma", "Degree"]

EDU_RANK = {lvl: i for i, lvl in enumerate(EDUCATION_LEVELS)}


def meets_education(applicant_edu, requirement_text):
    """
    Very rough parser: most requirement_text fields are free text like
    'Class 8 pass required only if project cost >10L...' or
    'Degree;Diploma;ITI or Vocational Training from recognised institution'.
    We treat 'Not specified' / 'No minimum' as always satisfied, and
    otherwise check whether the applicant's education rank meets the
    LOWEST mentioned threshold we can detect. This is a simplification
    -- document it as such in your report.
    """
    if pd.isna(requirement_text):
        return True
    text = str(requirement_text).lower()
    if "not specified" in text or "not applicable" in text or "no minimum" in text:
        return True
    if "degree" in text or "diploma" in text or "iti" in text:
        # NEEDS-style requirement: applicant must have at least ITI/vocational
        return EDU_RANK.get(applicant_edu, 0) >= EDU_RANK["ITI/Vocational"]
    if "class 8" in text:
        return EDU_RANK.get(applicant_edu, 0) >= EDU_RANK["Class 8"]
    if "class 5" in text:
        return EDU_RANK.get(applicant_edu, 0) >= EDU_RANK["Class 5"]
    if "7th" in text:
        return EDU_RANK.get(applicant_edu, 0) >= EDU_RANK["Class 8"]
    if "10th" in text:
        return EDU_RANK.get(applicant_edu, 0) >= EDU_RANK["Class 10"]
    # Fallback: unrecognized text pattern, assume satisfied
    return True


def category_matches(applicant_category, target_category_field, applicant_sector=None):
    """target_category_field is semicolon-separated, e.g. 'SC;ST;Women'.

    Some schemes (PM Vishwakarma, PM SVANidhi) encode a TRADE/occupation
    here instead of a demographic category (e.g. 'Traditional artisans
    and craftspeople', 'Urban street vendors'). If none of the known
    demographic categories appear in this field at all, treat it as an
    occupation descriptor and defer to sector matching instead of
    failing every applicant outright.
    """
    if pd.isna(target_category_field):
        return True
    field_text = str(target_category_field)
    targets = [t.strip().split(" (")[0] for t in field_text.split(";")]

    known_categories_present = any(cat in field_text for cat in CATEGORIES)
    if not known_categories_present:
        # Occupation-based field, not demographic -- let sector_matches decide.
        return True

    if applicant_category in targets:
        return True
    if "General" in targets and applicant_category == "General":
        return True
    return False


def state_matches(applicant_state, scheme_row):
    """Check state-restricted schemes (Gujarat, Maharashtra, Tamil Nadu, NER)."""
    name = scheme_row["scheme_name"]
    ministry = str(scheme_row["ministry_body"])
    if "Gujarat" in name or "Gujarat" in ministry:
        return applicant_state == "Gujarat"
    if "Maharashtra" in name or "Maharashtra" in ministry:
        return applicant_state == "Maharashtra"
    if "Tamil Nadu" in name or "Tamil Nadu" in ministry:
        return applicant_state == "Tamil Nadu"
    if "NEDFi" in name or "North Eastern" in ministry:
        return applicant_state in NER_STATES
    return True  # central scheme, no state restriction


def age_matches(age, min_age, max_age):
    if not pd.isna(min_age) and age < min_age:
        return False
    if not pd.isna(max_age) and age > max_age:
        return False
    return True


SECTOR_ALIASES = {
    "Manufacturing": ["manufactur"],
    "Service": ["service"],
    "Trading": ["trad"],
    "Agriculture": ["agri", "farm"],
    "Fisheries": ["fisher", "aquaculture", "fish"],
    "Artisan/Traditional Trade": ["artisan", "craft", "traditional", "handicraft",
                                  "hand-tool", "carpentry", "blacksmith", "pottery"],
    "Street Vending": ["street vend", "vending", "vendor"],
    "Tech Startup": ["technology", "tech-driven", "startup", "innovat", "scalable"],
}


def sector_matches(applicant_sector, allowed_text, excluded_text):
    aliases = SECTOR_ALIASES.get(applicant_sector, [applicant_sector.lower()])

    if not pd.isna(excluded_text):
        excluded = str(excluded_text).lower()
        for alias in aliases:
            if alias in excluded:
                return False

    if pd.isna(allowed_text):
        return True

    allowed = str(allowed_text).lower()
    if "general" in allowed or "msme" in allowed:
        return True

    # If the allowed-text field doesn't contain ANY recognizable sector
    # keyword at all (from any category), it's likely describing something
    # else (e.g. ownership structure, ""New and existing SC-owned
    # enterprises"") rather than a genuine sector restriction -- so don't
    # penalize the applicant for a field we can't reliably parse.
    all_known_keywords = [kw for kws in SECTOR_ALIASES.values() for kw in kws]
    if not any(kw in allowed for kw in all_known_keywords):
        return True

    return any(alias in allowed for alias in aliases)


def is_entity_scheme(scheme_row):
    """SISFS applies to registered companies, not individuals."""
    return "startups (not individuals)" in str(scheme_row["target_category"]).lower()


def generate_applicant(rng, applicant_id):
    return {
        "applicant_id": applicant_id,
        "age": rng.randint(16, 65),
        "category": rng.choice(CATEGORIES),
        "state": rng.choice(ALL_STATES),
        "education": rng.choice(EDUCATION_LEVELS),
        "annual_family_income_inr": rng.choice(
            [30000, 50000, 80000, 100000, 150000, 250000,
             300000, 500000, 800000, 1200000]
        ),
        "sector": rng.choice(SECTORS),
        "is_registered_company": rng.random() < 0.15,  # most applicants are individuals
        "family_member_already_availed_scheme": rng.random() < 0.1,
    }


def label_applicant(applicant, schemes_df):
    """
    Returns a dict {scheme_name: 0/1} applying the eligibility logic.
    NOTE: This is intentionally a simplified rule engine covering the
    fields your CSV captures well (age, category, state, education,
    sector). Ownership %, project-cost specifics, and other nuanced
    conditions in 'special_conditions' are NOT fully parsed here --
    that free text is exactly what your RAG layer should surface
    instead of forcing it into a numeric feature.
    """
    labels = {}
    for _, scheme in schemes_df.iterrows():
        name = scheme["scheme_name"]

        # Entity-only schemes (SISFS) require a registered company
        if is_entity_scheme(scheme):
            labels[name] = int(applicant["is_registered_company"])
            continue

        eligible = True
        eligible &= category_matches(applicant["category"], scheme["target_category"],
                                       applicant["sector"])
        eligible &= age_matches(applicant["age"], scheme["min_age"], scheme["max_age"])
        eligible &= state_matches(applicant["state"], scheme)
        eligible &= meets_education(applicant["education"], scheme["education_requirement"])
        eligible &= sector_matches(applicant["sector"],
                                    scheme["business_type_allowed"],
                                    scheme["business_type_excluded"])

        # crude income check where a limit is specified
        if not pd.isna(scheme["income_limit_annual_inr"]):
            limit = scheme["income_limit_annual_inr"]
            if scheme["income_limit_area"] == "rural":
                limit = limit  # already rural-specific figure in this dataset
            eligible &= applicant["annual_family_income_inr"] <= limit

        labels[name] = int(eligible)
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schemes_csv", default="scheme_eligibility_data.csv")
    parser.add_argument("--n", type=int, default=800,
                         help="number of synthetic applicants to generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="training_data.csv")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    schemes_df = pd.read_csv(args.schemes_csv)

    rows = []
    for i in range(args.n):
        applicant = generate_applicant(rng, i)
        labels = label_applicant(applicant, schemes_df)
        row = {**applicant, **{f"eligible__{k}": v for k, v in labels.items()}}
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)

    n_schemes = len(schemes_df)
    eligible_cols = [c for c in out_df.columns if c.startswith("eligible__")]
    avg_matches = out_df[eligible_cols].sum(axis=1).mean()

    print(f"Generated {args.n} applicant profiles across {n_schemes} schemes.")
    print(f"Saved to: {args.out}")
    print(f"Average number of eligible schemes per applicant: {avg_matches:.2f}")
    print(f"Applicants with zero matches: {(out_df[eligible_cols].sum(axis=1) == 0).sum()}")


if __name__ == "__main__":
    main()
