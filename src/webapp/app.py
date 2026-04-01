import os
import sys
import io
import json
import pandas as pd
import numpy as np
import joblib
import shap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triage_engine import (
    normalize_label, SEVERITY_ORDER, SEVERITY_NAMES,
    SELECTED_FEATURES, prepare_data
)
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# ─────────────────────────────────────────────
# LOAD MODEL ON STARTUP
# ─────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'triage_model.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'models', 'feature_names.pkl')

print("Loading model...")
model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)

print("Building SHAP explainer...")
# Use a small dummy background — real background not needed at startup
# Will be rebuilt per-request on first upload using actual data
_explainer = None

def get_explainer(X_background):
    global _explainer
    if _explainer is None:
        sample = X_background.sample(min(200, len(X_background)), random_state=42)
        _explainer = shap.TreeExplainer(model, sample)
    return _explainer


# ─────────────────────────────────────────────
# CORE TRIAGE LOGIC
# ─────────────────────────────────────────────

def triage_dataframe(df_raw):
    """
    Takes a raw uploaded dataframe.
    Returns a list of alert dicts with severity, confidence, and SHAP reasons.
    """
    df = df_raw.copy()

    # Normalize labels if present (for evaluation), otherwise add dummy
    if 'Label' not in df.columns:
        df['Label'] = 'UNKNOWN'
    df['Label'] = df['Label'].apply(normalize_label)

    # Select and clean features
    available = [f for f in feature_names if f in df.columns]
    missing = set(feature_names) - set(available)

    X = df[available].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)

    # Fill any features missing from upload with 0
    for feat in feature_names:
        if feat not in X.columns:
            X[feat] = 0
    X = X[feature_names]

    # Predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    # SHAP
    explainer = get_explainer(X)
    shap_values = explainer.shap_values(X)

    alerts = []
    for i in range(len(X)):
        severity_idx = y_pred[i]
        severity = SEVERITY_NAMES[severity_idx]
        proba = y_proba[i]

        # Parse SHAP for predicted class
        if isinstance(shap_values, list):
            class_shap = shap_values[severity_idx][i]
        elif shap_values.ndim == 3:
            class_shap = shap_values[i, :, severity_idx]
        else:
            class_shap = shap_values[i]

        feature_shap = pd.Series(class_shap, index=feature_names)
        top = feature_shap.abs().sort_values(ascending=False).head(5)

        reasons = []
        for feat in top.index:
            val = X.iloc[i][feat]
            shap_val = feature_shap[feat]
            direction = "+" if shap_val > 0 else "-"
            reasons.append({
                'feature': feat,
                'value': round(float(val), 3),
                'shap': round(float(shap_val), 4),
                'direction': direction
            })

        alerts.append({
            'index': i,
            'original_label': df['Label'].iloc[i] if 'Label' in df.columns else 'N/A',
            'severity': severity,
            'confidence_low': round(float(proba[0]) * 100, 1),
            'confidence_medium': round(float(proba[1]) * 100, 1),
            'confidence_high': round(float(proba[2]) * 100, 1),
            'reasons': reasons,
            'destination_port': int(X.iloc[i].get('Destination Port', 0)),
            'flow_packets_s': round(float(X.iloc[i].get('Flow Packets/s', 0)), 1),
        })

    # Sort by severity descending
    severity_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    alerts.sort(key=lambda x: severity_order[x['severity']])

    return alerts, list(missing)


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/triage', methods=['POST'])
def triage():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        content = file.read().decode('utf-8', errors='replace')
        df = pd.read_csv(io.StringIO(content), low_memory=False)
    except Exception as e:
        return jsonify({'error': f'Could not parse CSV: {str(e)}'}), 400

    if len(df) > 5000:
        df = df.sample(5000, random_state=42)

    try:
        alerts, missing_features = triage_dataframe(df)
    except Exception as e:
        import traceback
        return jsonify({'error': traceback.format_exc()}), 500

    summary = {
        'total': len(alerts),
        'high': sum(1 for a in alerts if a['severity'] == 'HIGH'),
        'medium': sum(1 for a in alerts if a['severity'] == 'MEDIUM'),
        'low': sum(1 for a in alerts if a['severity'] == 'LOW'),
        'missing_features': list(missing_features),
    }

    return jsonify({'summary': summary, 'alerts': alerts})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
