import os
import sys
import io
import pandas as pd
import numpy as np
import joblib
import shap

# Ensure both src/ and src/webapp/ are on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triage_engine import (
    normalize_label, SEVERITY_ORDER, SEVERITY_NAMES,
    SELECTED_FEATURES, prepare_data
)
from pdf_exporter import generate_report
from flask import Flask, request, render_template, jsonify, send_file

app = Flask(__name__)

# ─────────────────────────────────────────────
# LOAD MODEL ON STARTUP
# ─────────────────────────────────────────────

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH    = os.path.join(BASE_DIR, 'models', 'triage_model.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'models', 'feature_names.pkl')

print("Loading model...")
model         = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)
print("Model loaded.")

_explainer = None

def get_explainer(X_background):
    global _explainer
    if _explainer is None:
        sample     = X_background.sample(min(200, len(X_background)), random_state=42)
        _explainer = shap.TreeExplainer(model, sample)
    return _explainer


# ─────────────────────────────────────────────
# CORE TRIAGE LOGIC
# ─────────────────────────────────────────────

def triage_dataframe(df_raw):
    df = df_raw.copy()

    if 'Label' not in df.columns:
        df['Label'] = 'UNKNOWN'
    df['Label'] = df['Label'].apply(normalize_label)

    available = [f for f in feature_names if f in df.columns]
    missing   = set(feature_names) - set(available)

    X = df[available].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)

    # Fill any features missing from the upload with 0
    for feat in feature_names:
        if feat not in X.columns:
            X[feat] = 0
    X = X[feature_names]

    y_pred   = model.predict(X)
    y_proba  = model.predict_proba(X)

    explainer   = get_explainer(X)
    shap_values = explainer.shap_values(X)

    alerts = []
    for i in range(len(X)):
        severity_idx = y_pred[i]
        severity     = SEVERITY_NAMES[severity_idx]
        proba        = y_proba[i]

        # Handle both shap output formats
        if isinstance(shap_values, list):
            class_shap = shap_values[severity_idx][i]
        elif shap_values.ndim == 3:
            class_shap = shap_values[i, :, severity_idx]
        else:
            class_shap = shap_values[i]

        feature_shap = pd.Series(class_shap, index=feature_names)
        top          = feature_shap.abs().sort_values(ascending=False).head(5)

        reasons = []
        for feat in top.index:
            val      = X.iloc[i][feat]
            shap_val = feature_shap[feat]
            reasons.append({
                'feature':   feat,
                'value':     round(float(val), 3),
                'shap':      round(float(shap_val), 4),
                'direction': '+' if shap_val > 0 else '-'
            })

        alerts.append({
            'index':             i,
            'original_label':    df['Label'].iloc[i],
            'severity':          severity,
            'confidence_low':    round(float(proba[0]) * 100, 1),
            'confidence_medium': round(float(proba[1]) * 100, 1),
            'confidence_high':   round(float(proba[2]) * 100, 1),
            'reasons':           reasons,
            'destination_port':  int(X.iloc[i].get('Destination Port', 0)),
            'flow_packets_s':    round(float(X.iloc[i].get('Flow Packets/s', 0)), 1),
        })

    severity_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    alerts.sort(key=lambda x: severity_order[x['severity']])

    return alerts, list(missing)


# In-memory store for last triage result
_last_result = {'summary': None, 'alerts': None}


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
        df      = pd.read_csv(io.StringIO(content), low_memory=False)
    except Exception as e:
        return jsonify({'error': f'Could not parse CSV: {str(e)}'}), 400

    if len(df) > 5000:
        df = df.sample(5000, random_state=42)

    try:
        alerts, missing_features = triage_dataframe(df)
    except Exception:
        import traceback
        return jsonify({'error': traceback.format_exc()}), 500

    summary = {
        'total':            len(alerts),
        'high':             sum(1 for a in alerts if a['severity'] == 'HIGH'),
        'medium':           sum(1 for a in alerts if a['severity'] == 'MEDIUM'),
        'low':              sum(1 for a in alerts if a['severity'] == 'LOW'),
        'missing_features': list(missing_features),
    }

    # Store for PDF export
    _last_result['summary'] = summary
    _last_result['alerts']  = alerts

    return jsonify({'summary': summary, 'alerts': alerts})


@app.route('/export/pdf', methods=['GET'])
def export_pdf():
    if _last_result['summary'] is None:
        return jsonify({'error': 'No triage results available. Upload a file first.'}), 400

    try:
        pdf_bytes = generate_report(_last_result['summary'], _last_result['alerts'])
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename  = f'soc_triage_report_{timestamp}.pdf'

        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )
    except Exception:
        import traceback
        error_text = traceback.format_exc()
        print(error_text)
        return jsonify({'error': error_text}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
