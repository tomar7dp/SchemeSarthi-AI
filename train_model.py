"""
train_model.py

Trains a multi-label ML model that predicts which government schemes
an applicant is eligible for, based on training_data.csv.

Uses RandomForest wrapped in MultiOutputClassifier -- interpretable,
fast to train, no GPU needed, good enough for a prototype/demo.

Outputs:
  - model.pkl              (trained model + label encoders)
  - evaluation_report.txt  (precision/recall/F1 per scheme, for your report)
"""

import json
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, precision_recall_fscore_support


FEATURE_COLS_NUMERIC = ["age", "annual_family_income_inr"]
FEATURE_COLS_CATEGORICAL = ["category", "state", "education", "sector"]
FEATURE_COLS_BOOL = ["is_registered_company", "family_member_already_availed_scheme"]


def build_features(df, encoder=None, fit=False):
    if fit:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        cat_features = encoder.fit_transform(df[FEATURE_COLS_CATEGORICAL])
    else:
        cat_features = encoder.transform(df[FEATURE_COLS_CATEGORICAL])

    num_features = df[FEATURE_COLS_NUMERIC].values
    bool_features = df[FEATURE_COLS_BOOL].astype(int).values

    import numpy as np
    X = np.hstack([num_features, bool_features, cat_features])
    return X, encoder


def main():
    df = pd.read_csv("training_data.csv")
    label_cols = [c for c in df.columns if c.startswith("eligible__")]
    scheme_names = [c.replace("eligible__", "") for c in label_cols]

    X_df_train, X_df_test, y_train, y_test = train_test_split(
        df, df[label_cols], test_size=0.2, random_state=42
    )

    X_train, encoder = build_features(X_df_train, fit=True)
    X_test, _ = build_features(X_df_test, encoder=encoder, fit=False)

    model = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Per-scheme evaluation
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )

    report_lines = ["Per-scheme evaluation (test set)\n" + "=" * 60]
    for i, name in enumerate(scheme_names):
        report_lines.append(
            f"{name[:50]:50s}  P={precision[i]:.2f}  R={recall[i]:.2f}  "
            f"F1={f1[i]:.2f}  support={support[i]}"
        )

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    report_lines.append("\n" + "=" * 60)
    report_lines.append(f"Macro-average  P={macro_p:.3f}  R={macro_r:.3f}  F1={macro_f1:.3f}")

    report_text = "\n".join(report_lines)
    print(report_text)

    with open("evaluation_report.txt", "w") as f:
        f.write(report_text)

    with open("model.pkl", "wb") as f:
        pickle.dump({
            "model": model,
            "encoder": encoder,
            "scheme_names": scheme_names,
            "feature_cols_numeric": FEATURE_COLS_NUMERIC,
            "feature_cols_categorical": FEATURE_COLS_CATEGORICAL,
            "feature_cols_bool": FEATURE_COLS_BOOL,
        }, f)

    print("\nSaved model.pkl and evaluation_report.txt")


if __name__ == "__main__":
    main()
