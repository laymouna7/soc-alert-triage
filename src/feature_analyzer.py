import pandas as pd
import numpy as np

SEVERITY_MAP = {
    'BENIGN': 'LOW',
    'PortScan': 'MEDIUM',
    'Bot': 'MEDIUM',
    'FTP-Patator': 'MEDIUM',
    'SSH-Patator': 'MEDIUM',
    'DoS Hulk': 'HIGH',
    'DoS GoldenEye': 'HIGH',
    'DoS slowloris': 'HIGH',
    'DoS Slowhttptest': 'HIGH',
    'DDoS': 'HIGH',
    'Infiltration': 'HIGH',
    'Heartbleed': 'HIGH',
    'Web Attack \xef\xbf\xbd Brute Force': 'HIGH',
    'Web Attack \xef\xbf\xbd XSS': 'HIGH',
    'Web Attack \xef\xbf\xbd Sql Injection': 'HIGH',
}

FEATURE_CATEGORIES = {
    'basic_flow': [
        'Destination Port', 'Flow Duration', 'Total Fwd Packets',
        'Total Backward Packets', 'Total Length of Fwd Packets',
        'Total Length of Bwd Packets'
    ],
    'packet_size': [
        'Fwd Packet Length Max', 'Fwd Packet Length Min',
        'Fwd Packet Length Mean', 'Fwd Packet Length Std',
        'Bwd Packet Length Max', 'Bwd Packet Length Min',
        'Bwd Packet Length Mean', 'Bwd Packet Length Std',
        'Min Packet Length', 'Max Packet Length',
        'Packet Length Mean', 'Packet Length Std'
    ],
    'rate_metrics': [
        'Flow Bytes/s', 'Flow Packets/s',
        'Fwd Packets/s', 'Bwd Packets/s'
    ],
    'timing_iat': [
        'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
        'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
        'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min'
    ],
    'flag_counts': [
        'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
        'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
        'CWE Flag Count', 'ECE Flag Count'
    ],
    'active_idle': [
        'Active Mean', 'Active Std', 'Active Max', 'Active Min',
        'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
    ],
    'advanced': [
        'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size',
        'Avg Bwd Segment Size', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
        'act_data_pkt_fwd', 'min_seg_size_forward'
    ]
}


def analyze_dataset_overview(df):
    benign = df[df['Label'] == 'BENIGN']
    attacks = df[df['Label'] != 'BENIGN']

    print("=" * 80)
    print("DATASET OVERVIEW")
    print("=" * 80)
    print(f"Total samples  : {len(df):,}")
    print(f"Benign         : {len(benign):,} ({len(benign)/len(df)*100:.2f}%)")
    print(f"Attacks        : {len(attacks):,} ({len(attacks)/len(df)*100:.2f}%)")
    print()
    print("Attack breakdown:")
    print(attacks['Label'].value_counts().to_string())

    return benign, attacks


def analyze_features_by_category(df, benign, attacks):
    """
    For every feature in every category, compute benign vs attack means
    and flag features with >10% difference.
    """
    numeric_cols = set(df.select_dtypes(include=[np.number]).columns)

    print()
    print("=" * 80)
    print("FEATURE ANALYSIS BY CATEGORY (Benign vs Attack mean)")
    print("=" * 80)

    for category, features in FEATURE_CATEGORIES.items():
        available = [f for f in features if f in numeric_cols]
        if not available:
            continue

        print(f"\n--- {category.upper()} ---")

        for col in available:
            b_mean = benign[col].replace([np.inf, -np.inf], np.nan).mean()
            a_mean = attacks[col].replace([np.inf, -np.inf], np.nan).mean()

            if pd.isna(b_mean) or pd.isna(a_mean):
                continue

            diff_pct = abs((a_mean - b_mean) / (abs(b_mean) + 1e-10)) * 100

            if diff_pct > 10:
                flag = "  <<< HIGH DIFF" if diff_pct > 100 else ""
                print(f"  {col:<40}  benign={b_mean:>15.2f}  attack={a_mean:>15.2f}  diff={diff_pct:>8.1f}%{flag}")


def analyze_per_attack_type(df):
    """
    Show mean of key features per attack type so we understand
    what makes each attack distinct.
    """
    key_features = [
        'Flow Packets/s', 'Flow IAT Min', 'Flow IAT Mean',
        'SYN Flag Count', 'ACK Flag Count', 'RST Flag Count',
        'Total Backward Packets', 'Fwd Packet Length Mean',
        'Flow Bytes/s'
    ]
    available = [f for f in key_features if f in df.columns]

    print()
    print("=" * 80)
    print("KEY FEATURES PER ATTACK TYPE")
    print("=" * 80)

    df_clean = df.copy()
    df_clean[available] = df_clean[available].replace([np.inf, -np.inf], np.nan)

    grouped = df_clean.groupby('Label')[available].mean()
    pd.set_option('display.float_format', '{:.2f}'.format)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(grouped.to_string())


def create_severity_labels(df):
    df = df.copy()
    df['Label'] = df['Label'].str.strip()
    df['severity'] = df['Label'].map(SEVERITY_MAP)

    unmapped = df['severity'].isna().sum()
    if unmapped > 0:
        print(f"\nWARNING: {unmapped} rows could not be mapped to a severity level:")
        print(df[df['severity'].isna()]['Label'].value_counts().to_string())

    print()
    print("=" * 80)
    print("SEVERITY MAPPING RESULT")
    print("=" * 80)
    print(df['severity'].value_counts().to_string())

    return df


if __name__ == "__main__":
    print("Loading dataset...")
    df = pd.read_csv('data/combined_dataset.csv', low_memory=False)
    print(f"Loaded {len(df):,} rows\n")

    benign, attacks = analyze_dataset_overview(df)
    analyze_features_by_category(df, benign, attacks)
    analyze_per_attack_type(df)
    df = create_severity_labels(df)

    df.to_csv('data/combined_with_priority.csv', index=False)
    print(f"\nSaved to: data/combined_with_priority.csv")
