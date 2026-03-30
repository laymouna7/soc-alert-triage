import pandas as pd
import numpy as np
import joblib
import os
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# ─────────────────────────────────────────────
# 1. LABEL NORMALIZATION
# ─────────────────────────────────────────────

def normalize_label(label: str) -> str:
    """
    The dataset has encoding garbage in Web Attack labels.
    e.g. 'Web Attack â Brute Force' -> 'Web Attack - Brute Force'
    We normalize with a regex so the severity map can match cleanly.
    """
    label = label.strip()
    label = re.sub(r'Web Attack\s+.\s+', 'Web Attack - ', label)
    return label


# ─────────────────────────────────────────────
# 2. SEVERITY MAPPING
# ─────────────────────────────────────────────

SEVERITY_MAP = {
    'BENIGN':                     'LOW',
    'PortScan':                   'MEDIUM',
    'Bot':                        'MEDIUM',
    'FTP-Patator':                'MEDIUM',
    'SSH-Patator':                'MEDIUM',
    'DoS Hulk':                   'HIGH',
    'DoS GoldenEye':              'HIGH',
    'DoS slowloris':              'HIGH',
    'DoS Slowhttptest':           'HIGH',
    'DDoS':                       'HIGH',
    'Infiltration':               'HIGH',
    'Heartbleed':                 'HIGH',
    'Web Attack - Brute Force':   'HIGH',
    'Web Attack - XSS':           'HIGH',
    'Web Attack - Sql Injection': 'HIGH',
}

# Numeric encoding for the classifier
SEVERITY_ORDER = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
SEVERITY_NAMES = ['LOW', 'MEDIUM', 'HIGH']


# ─────────────────────────────────────────────
# 3. FEATURE SELECTION
# ─────────────────────────────────────────────

# 38 features selected from 78 available.
# Dropped: bulk/subflow (mostly zeros), duplicate columns,
#          variance (redundant with std), URG/CWE/ECE flags (near-zero everywhere)
SELECTED_FEATURES = [
    'Destination Port', 'Flow Duration',
    'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Flow Bytes/s', 'Flow Packets/s',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'SYN Flag Count', 'ACK Flag Count', 'RST Flag Count',
    'FIN Flag Count', 'PSH Flag Count',
    'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Mean', 'Bwd Packet Length Std',
    'Packet Length Mean', 'Packet Length Std',
    'Average Packet Size', 'Fwd Packets/s', 'Bwd Packets/s',
    'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
    'Active Mean', 'Idle Mean',
]


# ─────────────────────────────────────────────
# 4. DATA PREPARATION
# ─────────────────────────────────────────────

def prepare_data(df):
    """
    Steps:
    1. Normalize label strings (fix encoding)
    2. Map each label to LOW / MEDIUM / HIGH
    3. Select the 38 chosen features
    4. Replace inf/NaN with column medians
    """
    df = df.copy()
    df['Label'] = df['Label'].apply(normalize_label)
    df['severity'] = df['Label'].map(SEVERITY_MAP)

    unknown = df['severity'].isna().sum()
    if unknown > 0:
        print(f"WARNING: Dropping {unknown} rows with unmapped labels:")
        print(df[df['severity'].isna()]['Label'].value_counts().to_string())
        df = df.dropna(subset=['severity'])

    available = [f for f in SELECTED_FEATURES if f in df.columns]
    missing = set(SELECTED_FEATURES) - set(available)
    if missing:
        print(f"WARNING: Missing features (skipped): {missing}")

    X = df[available].copy()
    y = df['severity'].copy()

    # inf values appear when flow duration is 0 (division by zero in rate calcs)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)

    return X, y, available


# ─────────────────────────────────────────────
# 5. TRAIN THE MODEL
# ─────────────────────────────────────────────

def train_model(X, y):
    print("\n" + "=" * 70)
    print("TRAINING TRIAGE ENGINE")
    print("=" * 70)

    y_encoded = y.map(SEVERITY_ORDER)

    print("\nSeverity Distribution:")
    for sev, count in y.value_counts().items():
        print(f"  {sev:<10} {count:>8,}  ({count / len(y) * 100:.1f}%)")

    # stratify=y_encoded ensures each severity class appears in train/test
    # in the same proportion as the full dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"\nTrain size: {len(X_train):,}  |  Test size: {len(X_test):,}")

    # class_weight='balanced' tells the model to penalize mistakes on rare classes more.
    # Without this, the model could ignore LOW (6.7%) and still get 93% accuracy.
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        max_depth=20,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42
    )

    print("\nTraining... (this may take 1-2 minutes)")
    model.fit(X_train, y_train)
    print("Training complete.")

    y_pred = model.predict(X_test)

    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=SEVERITY_NAMES))

    print("Confusion Matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=SEVERITY_NAMES, columns=SEVERITY_NAMES)
    print(cm_df)

    return model, X_test, y_test


# ─────────────────────────────────────────────
# 6. FEATURE IMPORTANCE
# ─────────────────────────────────────────────

def show_feature_importance(model, feature_names, top_n=15):
    importance = pd.Series(model.feature_importances_, index=feature_names)
    importance = importance.sort_values(ascending=False)

    print(f"\n{'=' * 70}")
    print(f"TOP {top_n} MOST IMPORTANT FEATURES")
    print(f"{'=' * 70}")
    for feat, imp in importance.head(top_n).items():
        bar = "|" * int(imp * 300)
        print(f"  {feat:<40} {imp:.4f}  {bar}")

    return importance


# ─────────────────────────────────────────────
# 7. TRIAGE A SINGLE ALERT
# ─────────────────────────────────────────────

def triage_alert(model, feature_names, flow: dict) -> dict:
    """
    Takes a flow (dict of feature name -> value).
    Returns severity, confidence per class, and human-readable reasons.

    Two layers:
    - ML model: assigns the severity label
    - Rule engine: explains WHY in human-readable terms
    Both run independently. The ML decides, the rules explain.
    """
    row = pd.DataFrame([flow])[feature_names]
    row.replace([np.inf, -np.inf], np.nan, inplace=True)
    row.fillna(0, inplace=True)

    severity_idx = model.predict(row)[0]
    severity = SEVERITY_NAMES[severity_idx]
    proba = model.predict_proba(row)[0]

    reasons = []

    if flow.get('SYN Flag Count', 0) > 5 and flow.get('ACK Flag Count', 0) == 0:
        reasons.append(
            f"SYN={flow['SYN Flag Count']} ACK=0 — "
            f"incomplete handshakes, port scan pattern"
        )
    if flow.get('Flow Packets/s', 0) > 10000:
        reasons.append(
            f"Flow rate = {flow['Flow Packets/s']:,.0f} pkt/s — "
            f"exceeds flood threshold"
        )
    if flow.get('Flow IAT Min', 99999) < 100:
        reasons.append(
            f"Min inter-arrival time = {flow['Flow IAT Min']:.1f} us — "
            f"near-zero gap, consistent with automated/scripted traffic"
        )
    if flow.get('Total Backward Packets', 99) < 2 and flow.get('Total Fwd Packets', 0) > 5:
        reasons.append(
            f"One-sided: {flow['Total Fwd Packets']:.0f} fwd packets, "
            f"{flow['Total Backward Packets']:.0f} backward — target not responding"
        )
    if flow.get('Flow Bytes/s', 0) > 500000:
        reasons.append(
            f"Throughput = {flow['Flow Bytes/s'] / 1e6:.2f} MB/s — "
            f"volumetric attack signature"
        )
    if flow.get('Bwd Packet Length Mean', 0) > 2000:
        reasons.append(
            f"Mean backward packet = {flow['Bwd Packet Length Mean']:.0f} bytes — "
            f"abnormally large server responses, possible data exfiltration"
        )
    if not reasons:
        reasons.append(
            "No explicit rule matched — severity assigned purely by ML pattern recognition"
        )

    return {
        'severity': severity,
        'confidence': {
            'LOW':    f"{proba[0] * 100:.1f}%",
            'MEDIUM': f"{proba[1] * 100:.1f}%",
            'HIGH':   f"{proba[2] * 100:.1f}%",
        },
        'reasons': reasons,
    }


# ─────────────────────────────────────────────
# 8. DEMO WITH REAL DATA
# ─────────────────────────────────────────────

def demo_from_real_data(model, feature_names, X_test, y_test, df_original):
    """
    Instead of inventing fake flows (which fail because they don't match
    real traffic statistics), we pull one actual flow per severity level
    from the test set. These are guaranteed to look like real attacks
    because they came from the dataset.

    We also look up the original Label (e.g. 'DoS Hulk', 'PortScan')
    so you can see what specific attack type each flow represents.
    """
    print("\n" + "=" * 70)
    print("DEMO: TRIAGE REAL FLOWS FROM TEST SET")
    print("=" * 70)
    print("(Each flow is a real network capture from the dataset)\n")

    # Rebuild a mapping from test set index to original label
    # X_test has the same index as the original df rows
    label_lookup = df_original['Label'].apply(normalize_label)

    for target_idx, target_name in enumerate(SEVERITY_NAMES):
        mask = (y_test == target_idx)
        subset = X_test[mask]

        if len(subset) == 0:
            print(f"  No {target_name} samples in test set.")
            continue

        # Take the first matching row
        sample_idx = subset.index[0]
        flow = subset.loc[sample_idx].to_dict()
        original_label = label_lookup.loc[sample_idx] if sample_idx in label_lookup.index else "unknown"

        result = triage_alert(model, feature_names, flow)

        print(f"  Actual severity  : {target_name}  (original label: {original_label})")
        print(f"  Predicted        : {result['severity']}")
        print(f"  Confidence       : LOW={result['confidence']['LOW']}  "
              f"MEDIUM={result['confidence']['MEDIUM']}  "
              f"HIGH={result['confidence']['HIGH']}")
        print(f"  Key feature values:")
        for feat in ['Bwd Packets/s', 'Average Packet Size', 'Flow Packets/s',
                     'Flow IAT Min', 'SYN Flag Count', 'Destination Port',
                     'Total Backward Packets', 'Bwd Packet Length Mean']:
            if feat in flow:
                print(f"    {feat:<35} = {flow[feat]:.2f}")
        print(f"  Reasons:")
        for r in result['reasons']:
            print(f"    - {r}")
        print()


# ─────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading dataset...")
    df = pd.read_csv('data/combined_dataset.csv', low_memory=False)
    print(f"Loaded {len(df):,} rows")

    X, y, feature_names = prepare_data(df)
    model, X_test, y_test = train_model(X, y)
    show_feature_importance(model, feature_names)

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/triage_model.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')
    print(f"\nModel saved to: models/triage_model.pkl")

    demo_from_real_data(model, feature_names, X_test, y_test, df)
