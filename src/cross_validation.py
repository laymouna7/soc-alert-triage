import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triage_engine import SEVERITY_NAMES, SEVERITY_ORDER, SEVERITY_MAP, normalize_label

# ─────────────────────────────────────────────
# UNSW-NB15 ATTACK CATEGORIES → SEVERITY
# ─────────────────────────────────────────────

UNSW_SEVERITY_MAP = {
    'Normal':         'LOW',
    'Reconnaissance': 'MEDIUM',
    'Backdoor':       'HIGH',
    'DoS':            'HIGH',
    'Exploits':       'HIGH',
    'Analysis':       'MEDIUM',
    'Fuzzers':        'MEDIUM',
    'Worms':          'HIGH',
    'Shellcode':      'HIGH',
    'Generic':        'HIGH',
}

# ─────────────────────────────────────────────
# FEATURE MAPPING: UNSW-NB15 → CIC-IDS2017
# ─────────────────────────────────────────────

UNSW_TO_CIC = {
    'dur':    'Flow Duration',
    'spkts':  'Total Fwd Packets',
    'dpkts':  'Total Backward Packets',
    'sbytes': 'Total Length of Fwd Packets',
    'dbytes': 'Total Length of Bwd Packets',
    'rate':   'Flow Packets/s',
    'smean':  'Fwd Packet Length Mean',
    'dmean':  'Bwd Packet Length Mean',
    'swin':   'Init_Win_bytes_forward',
    'dwin':   'Init_Win_bytes_backward',
    'sinpkt': 'Flow IAT Mean',
    'dinpkt': 'Bwd IAT Mean',
    'sjit':   'Flow IAT Std',
    'djit':   'Bwd IAT Std',
    'sload':  'Flow Bytes/s',
}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_unsw(path):
    train_path = os.path.join(path, 'UNSW_NB15_training-set.csv')
    test_path  = os.path.join(path, 'UNSW_NB15_testing-set.csv')
    df = pd.concat([
        pd.read_csv(train_path, low_memory=False),
        pd.read_csv(test_path,  low_memory=False)
    ], ignore_index=True)
    print(f"UNSW-NB15 loaded: {len(df):,} rows")
    print("\nAttack categories:")
    print(df['attack_cat'].value_counts().to_string())
    return df


def map_to_cic_features(df, feature_names):
    """
    Build a feature matrix in CIC-IDS2017 space from UNSW data.
    Features with no UNSW equivalent are filled with 0.
    """
    X = pd.DataFrame(index=df.index)
    for feat in feature_names:
        unsw_col = next((u for u, c in UNSW_TO_CIC.items() if c == feat), None)
        if unsw_col and unsw_col in df.columns:
            X[feat] = df[unsw_col].values
        else:
            X[feat] = 0
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)
    return X


def map_unsw_severity(df):
    df = df.copy()
    df['attack_cat'] = df['attack_cat'].str.strip()
    df['severity'] = df['attack_cat'].map(UNSW_SEVERITY_MAP)
    unmapped = df['severity'].isna().sum()
    if unmapped > 0:
        print(f"\nWARNING: {unmapped} unmapped categories:")
        print(df[df['severity'].isna()]['attack_cat'].value_counts().to_string())
        df = df.dropna(subset=['severity'])
    return df


def feature_coverage_report(feature_names):
    mapped = {c: u for u, c in UNSW_TO_CIC.items()}
    print("\n" + "=" * 70)
    print("FEATURE COVERAGE REPORT")
    print("=" * 70)
    print(f"  {'Feature':<40} {'Status':<12} UNSW Column")
    print("  " + "-" * 68)
    covered = 0
    for feat in feature_names:
        if feat in mapped:
            print(f"  {feat:<40} {'MAPPED':<12} {mapped[feat]}")
            covered += 1
        else:
            print(f"  {feat:<40} {'ZEROED':<12} (no equivalent)")
    print(f"\n  Covered: {covered}/{len(feature_names)} ({covered/len(feature_names)*100:.1f}%)")
    print(f"  Zeroed : {len(feature_names)-covered}/{len(feature_names)}")


def evaluate(model, X, y_true_encoded, label, feature_names):
    y_pred = model.predict(X)
    print(f"\n{'=' * 70}")
    print(f"RESULTS: {label}")
    print(f"{'=' * 70}")
    print(classification_report(y_true_encoded, y_pred,
          target_names=SEVERITY_NAMES, zero_division=0))
    cm = confusion_matrix(y_true_encoded, y_pred)
    cm_df = pd.DataFrame(cm, index=SEVERITY_NAMES, columns=SEVERITY_NAMES)
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(cm_df)
    return y_pred


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    try:
        UNSW_PATH = '/home/laymouna/.cache/kagglehub/datasets/mrwellsdavid/unsw-nb15/versions/1'

        # ── Load full 38-feature model ──
        print("Loading model...")
        model = joblib.load('models/triage_model.pkl')
        feature_names = joblib.load('models/feature_names.pkl')

        # ── Feature coverage ──
        feature_coverage_report(feature_names)

        # ── Load and prepare UNSW ──
        df_unsw = load_unsw(UNSW_PATH)
        df_unsw = map_unsw_severity(df_unsw)

        print("\nMapping UNSW-NB15 features to CIC-IDS2017 space...")
        X_unsw_full = map_to_cic_features(df_unsw, feature_names)
        y_unsw = df_unsw['severity'].map(SEVERITY_ORDER)

        # ══════════════════════════════════════════
        # EXPERIMENT 1: Full 38-feature model on UNSW
        # (23 features zeroed — shows raw domain gap)
        # ══════════════════════════════════════════
        print("\n" + "=" * 70)
        print("EXPERIMENT 1: FULL MODEL (38 features) ON UNSW-NB15")
        print("23 features have no UNSW equivalent and are zeroed.")
        print("Expected: poor performance — quantifies the domain gap.")
        print("=" * 70)

        evaluate(model, X_unsw_full, y_unsw,
                 "CIC-IDS2017 model (38 features) → UNSW-NB15", feature_names)

        # Save experiment 1 results
        results1 = df_unsw[['attack_cat', 'severity']].copy()
        results1['predicted'] = [SEVERITY_NAMES[p] for p in model.predict(X_unsw_full)]
        results1['correct'] = results1['severity'] == results1['predicted']
        results1.to_csv('data/unsw_experiment1_results.csv', index=False)
        print(f"\nExperiment 1 results saved to: data/unsw_experiment1_results.csv")

        # ══════════════════════════════════════════
        # EXPERIMENT 2: Retrain on 15 shared features
        # Tests whether behavioral patterns transfer
        # when the feature playing field is level
        # ══════════════════════════════════════════
        print("\n" + "=" * 70)
        print("EXPERIMENT 2: RETRAIN ON 15 SHARED FEATURES ONLY")
        print("Train on CIC-IDS2017 (15 features), test on UNSW-NB15 (same 15).")
        print("Tests zero-shot transfer when features are comparable.")
        print("=" * 70)

        shared_features = [c for u, c in UNSW_TO_CIC.items()]

        # Load CIC, restrict to shared features
        print("\nLoading CIC-IDS2017...")
        df_cic = pd.read_csv('data/combined_dataset.csv', low_memory=False)
        df_cic['Label'] = df_cic['Label'].apply(normalize_label)
        df_cic['severity'] = df_cic['Label'].map(SEVERITY_MAP)
        df_cic = df_cic.dropna(subset=['severity'])

        available_shared = [f for f in shared_features if f in df_cic.columns]
        print(f"Shared features available in CIC: {len(available_shared)}")
        print(f"Features: {available_shared}")

        X_cic = df_cic[available_shared].copy()
        X_cic.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_cic.fillna(X_cic.median(), inplace=True)
        y_cic = df_cic['severity'].map(SEVERITY_ORDER)

        X_train, X_test_cic, y_train, y_test_cic = train_test_split(
            X_cic, y_cic, test_size=0.2, random_state=42, stratify=y_cic
        )

        print(f"\nTraining shared-feature model on {len(X_train):,} CIC flows...")
        model_shared = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            max_depth=20,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42
        )
        model_shared.fit(X_train, y_train)
        print("Training complete.")

        # Evaluate on CIC test set (sanity check)
        evaluate(model_shared, X_test_cic, y_test_cic,
                 "Shared-feature model → CIC-IDS2017 test set (sanity check)",
                 available_shared)

        # Evaluate on UNSW with same 15 features — the real experiment
        X_unsw_shared = map_to_cic_features(df_unsw, available_shared)
        evaluate(model_shared, X_unsw_shared, y_unsw,
                 "Shared-feature model → UNSW-NB15 (zero-shot transfer)",
                 available_shared)

        # Save shared model
        os.makedirs('models', exist_ok=True)
        joblib.dump(model_shared, 'models/triage_model_shared_features.pkl')
        joblib.dump(available_shared, 'models/shared_feature_names.pkl')
        print(f"\nShared-feature model saved to: models/triage_model_shared_features.pkl")

        # ── Final summary ──
        print("\n" + "=" * 70)
        print("CROSS-DATASET VALIDATION SUMMARY")
        print("=" * 70)
        print("""
  Experiment 1 — Full model (38 features) on UNSW-NB15
    Feature coverage : 15/38 (39.5%)
    Result           : Model collapses to predicting LOW for all flows
    Interpretation   : Missing top SHAP features (TCP flags, dst port,
                       packet size stats) make attack flows indistinguishable
                       from benign in the model's learned space.

  Experiment 2 — Shared-feature model (15 features) on UNSW-NB15
    Feature coverage : 15/15 (100%)
    Result           : See classification report above
    Interpretation   : With a level feature playing field, we can measure
                       how much behavioral pattern transfer occurs across
                       environments. Any accuracy above chance (~33%) indicates
                       genuine cross-domain signal in flow-level statistics.

  Conclusion:
    The domain gap between CIC-IDS2017 and UNSW-NB15 is real and measurable.
    It is caused by feature incompatibility, not model failure.
    The correct production approach is domain-specific retraining or
    a feature set standardized across both datasets.
        """)
        # ══════════════════════════════════════════
        # EXPERIMENT 3: Train natively on UNSW-NB15
        # Establishes the performance ceiling for
        # this dataset with its own feature space
        # ══════════════════════════════════════════
        print("\n" + "=" * 70)
        print("EXPERIMENT 3: NATIVE UNSW-NB15 MODEL")
        print("Train and test entirely within UNSW-NB15.")
        print("Establishes the performance ceiling for this dataset.")
        print("=" * 70)

        # UNSW numeric features — equivalent role to our 38 CIC features
        UNSW_NUMERIC_FEATURES = [
            'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate',
            'sttl', 'dttl', 'sload', 'dload', 'sinpkt', 'dinpkt',
            'sjit', 'djit', 'swin', 'dwin', 'smean', 'dmean',
            'tcprtt', 'synack', 'ackdat',
            'ct_srv_src', 'ct_dst_ltm', 'ct_src_dport_ltm',
            'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'ct_src_ltm', 'ct_srv_dst'
        ]

        available_unsw = [f for f in UNSW_NUMERIC_FEATURES if f in df_unsw.columns]
        print(f"\nUsing {len(available_unsw)} native UNSW features")

        X_unsw_native = df_unsw[available_unsw].copy()
        X_unsw_native.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_unsw_native.fillna(X_unsw_native.median(), inplace=True)
        y_unsw_encoded = df_unsw['severity'].map(SEVERITY_ORDER)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_unsw_native, y_unsw_encoded,
            test_size=0.2, random_state=42, stratify=y_unsw_encoded
        )

        print(f"Train: {len(X_tr):,}  Test: {len(X_te):,}")
        print("Training native UNSW model...")

        model_unsw = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            max_depth=20,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42
        )
        model_unsw.fit(X_tr, y_tr)
        print("Training complete.")

        evaluate(model_unsw, X_te, y_te,
                 "Native UNSW-NB15 model → UNSW-NB15 test set",
                 available_unsw)

        joblib.dump(model_unsw, 'models/triage_model_unsw_native.pkl')
        joblib.dump(available_unsw, 'models/unsw_native_feature_names.pkl')
        print(f"\nNative UNSW model saved to: models/triage_model_unsw_native.pkl")

        print("\n" + "=" * 70)
        print("FINAL COMPARISON TABLE")
        print("=" * 70)
        print("""
  Model                   Features  Trained On    Tested On     Expected
  ──────────────────────────────────────────────────────────────────────
  CIC full model          38        CIC-IDS2017   CIC test      ~100%
  CIC full model          38        CIC-IDS2017   UNSW-NB15     36% (collapse)
  CIC shared model        15        CIC-IDS2017   UNSW-NB15     36% (collapse)
  UNSW native model       28        UNSW-NB15     UNSW test     see above

  Finding: Zero-shot cross-dataset transfer fails due to covariate shift.
  Solution: Domain-specific training. Same pipeline, different training data.
  Production implication: Deploy with periodic retraining on target environment.
        """)

    except Exception:
        import traceback
        traceback.print_exc()
