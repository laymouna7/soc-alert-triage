import pandas as pd
import numpy as np
import joblib
import shap
import os
import sys
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from triage_engine import (
    normalize_label, SEVERITY_MAP, SEVERITY_ORDER, SEVERITY_NAMES,
    SELECTED_FEATURES, prepare_data
)


# ─────────────────────────────────────────────
# 1. BUILD EXPLAINER
# ─────────────────────────────────────────────

def build_explainer(model, X_background):
    """
    TreeExplainer is designed specifically for tree-based models.
    It computes exact SHAP values by traversing the tree structure directly.

    X_background is a sample of real data used as a reference point.
    SHAP values measure how much each feature pushes the prediction
    away from the average prediction on this background sample.
    """
    print("Building SHAP explainer...")
    background = X_background.sample(500, random_state=42)
    explainer = shap.TreeExplainer(model, background)
    print("Explainer ready.")
    return explainer


# ─────────────────────────────────────────────
# 2. PARSE SHAP VALUES
# ─────────────────────────────────────────────

def parse_shap_values(shap_values, class_idx, sample_idx=0):
    """
    Handle both shap output formats depending on version:

    Format A (older shap): list of arrays, one per class
        shap_values[class_idx] -> (n_samples, n_features)
        shap_values[class_idx][sample_idx] -> (n_features,)

    Format B (newer shap): single array
        shap_values -> (n_samples, n_features, n_classes)
        shap_values[sample_idx, :, class_idx] -> (n_features,)

    We detect which format we have and handle both.
    """
    if isinstance(shap_values, list):
        # Format A
        return shap_values[class_idx][sample_idx]
    else:
        # Format B
        if shap_values.ndim == 3:
            return shap_values[sample_idx, :, class_idx]
        else:
            return shap_values[sample_idx]


def parse_global_shap(shap_values, n_features):
    """
    Compute mean absolute SHAP across all classes and samples,
    handling both shap output formats.
    """
    if isinstance(shap_values, list):
        # Format A: list of (n_samples, n_features)
        return np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        # Format B: (n_samples, n_features, n_classes) or (n_samples, n_features)
        if shap_values.ndim == 3:
            return np.abs(shap_values).mean(axis=(0, 2))
        else:
            return np.abs(shap_values).mean(axis=0)


# ─────────────────────────────────────────────
# 3. EXPLAIN A SINGLE FLOW
# ─────────────────────────────────────────────

def explain_flow(explainer, model, feature_names, flow: dict, original_label: str = "unknown"):
    """
    Compute SHAP values for one flow and print a human-readable explanation.

    SHAP logic:
    - The model has a baseline prediction (average over the background sample)
    - Each feature either pushes the prediction UP or DOWN from that baseline
    - SHAP value = how much that feature contributed to this specific prediction
    - Positive SHAP = pushed toward the predicted class
    - Negative SHAP = pushed away from the predicted class

    For a 3-class model, each feature gets 3 SHAP values (one per class).
    We show the values for the predicted class only.
    """
    row = pd.DataFrame([flow])[feature_names]
    row.replace([np.inf, -np.inf], np.nan, inplace=True)
    row.fillna(0, inplace=True)

    severity_idx = model.predict(row)[0]
    severity = SEVERITY_NAMES[severity_idx]
    proba = model.predict_proba(row)[0]

    shap_values = explainer.shap_values(row)
    class_shap = parse_shap_values(shap_values, severity_idx, sample_idx=0)

    feature_shap = pd.Series(class_shap, index=feature_names)
    top_features = feature_shap.abs().sort_values(ascending=False).head(8).index

    print("=" * 70)
    print("FLOW EXPLANATION")
    print("=" * 70)
    print(f"  Original label : {original_label}")
    print(f"  Predicted      : {severity}")
    print(f"  Confidence     : LOW={proba[0]*100:.1f}%  "
          f"MEDIUM={proba[1]*100:.1f}%  "
          f"HIGH={proba[2]*100:.1f}%")

    print(f"\n  Why {severity}? Top contributing features:")
    print(f"  (+ pushed toward {severity}, - pushed away from {severity})\n")

    for feat in top_features:
        shap_val = feature_shap[feat]
        feat_val = flow.get(feat, 0)
        direction = "+" if shap_val > 0 else "-"
        bar = "|" * int(abs(shap_val) * 500)
        print(f"  {direction}  {feat:<38}  val={feat_val:>12.2f}   shap={shap_val:>+.4f}  {bar}")

    print()


# ─────────────────────────────────────────────
# 4. EXPLAIN ONE REAL FLOW PER SEVERITY LEVEL
# ─────────────────────────────────────────────

def explain_test_samples(explainer, model, feature_names, X_test, y_test, df_original):
    """
    Pull one real flow per severity level from the test set and explain it.
    Uses real captured network flows so explanations reflect actual
    attack patterns, not invented values.
    """
    label_lookup = df_original['Label'].apply(normalize_label)

    for target_idx, target_name in enumerate(SEVERITY_NAMES):
        mask = (y_test == target_idx)
        subset = X_test[mask]

        if len(subset) == 0:
            print(f"No {target_name} samples found in test set.")
            continue

        sample_idx = subset.index[0]
        flow = subset.loc[sample_idx].to_dict()
        original_label = (
            label_lookup.loc[sample_idx]
            if sample_idx in label_lookup.index
            else "unknown"
        )

        explain_flow(explainer, model, feature_names, flow, original_label)


# ─────────────────────────────────────────────
# 5. GLOBAL SHAP IMPORTANCE
# ─────────────────────────────────────────────

def global_shap_importance(explainer, X_sample, feature_names, top_n=15):
    """
    Compute mean absolute SHAP values across a sample of flows.

    More reliable than Random Forest's built-in feature_importances_,
    which is biased toward high-cardinality features.
    SHAP importance reflects actual contribution to predictions,
    averaged across all samples and all classes.
    """
    print("=" * 70)
    print(f"GLOBAL SHAP IMPORTANCE (top {top_n} features)")
    print("=" * 70)
    print("Computing SHAP values across 300 sample flows...\n")

    sample = X_sample.sample(300, random_state=42)
    shap_values = explainer.shap_values(sample)

    mean_abs_shap = parse_global_shap(shap_values, len(feature_names))
    importance = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=False)

    for feat, val in importance.head(top_n).items():
        bar = "|" * int(val * 1000)
        print(f"  {feat:<40} {val:.4f}  {bar}")

    print()
    return importance


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    try:
        print("Loading model...")
        model = joblib.load('models/triage_model.pkl')
        feature_names = joblib.load('models/feature_names.pkl')

        print("Loading dataset...")
        df = pd.read_csv('data/combined_dataset.csv', low_memory=False)
        print(f"Loaded {len(df):,} rows\n")

        print("Preparing data...")
        X, y, _ = prepare_data(df)
        y_encoded = y.map(SEVERITY_ORDER)

        _, X_test, _, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        print(f"Test set: {len(X_test):,} flows\n")

        explainer = build_explainer(model, X_test)
        print()

        global_shap_importance(explainer, X_test, feature_names)

        print("=" * 70)
        print("PER-FLOW EXPLANATIONS (real flows from test set)")
        print("=" * 70)
        print()
        explain_test_samples(explainer, model, feature_names, X_test, y_test, df)

    except Exception:
        import traceback
        traceback.print_exc()
