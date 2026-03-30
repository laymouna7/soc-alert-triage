import os
import pandas as pd
import kagglehub


def download_dataset():
    print("Downloading dataset...")
    path = kagglehub.dataset_download("chethuhn/network-intrusion-dataset")
    print(f"Dataset path: {path}")
    return path


def list_csv_files(path):
    csv_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))

    if not csv_files:
        raise Exception("No CSV files found!")

    print("\nAvailable CSV files:")
    for i, file in enumerate(csv_files):
        print(f"[{i}] {file}")

    return csv_files


def clean_column_names(df):
    df.columns = df.columns.str.strip()
    return df


def load_balanced_sample(file_path, max_benign=5000, max_attacks=None):
    """
    Load a file with a balanced strategy:
    - Cap benign rows to avoid drowning out attacks
    - Keep ALL attack rows (or up to max_attacks)
    """
    print(f"\nLoading: {os.path.basename(file_path)}")
    
    # Load full file
    df = pd.read_csv(file_path, low_memory=False)
    df = clean_column_names(df)
    
    if 'Label' not in df.columns:
        print("  ⚠️ No Label column, skipping")
        return None

    # Replace inf values
    df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)

    benign = df[df['Label'] == 'BENIGN']
    attacks = df[df['Label'] != 'BENIGN']

    print(f"  Full file — Benign: {len(benign)}, Attacks: {len(attacks)}")
    print(f"  Attack types: {attacks['Label'].value_counts().to_dict()}")

    # Sample benign rows to keep dataset manageable
    benign_sample = benign.sample(min(max_benign, len(benign)), random_state=42)

    # Keep all attacks (or cap if specified)
    attack_sample = attacks if max_attacks is None else attacks.sample(
        min(max_attacks, len(attacks)), random_state=42
    )

    combined = pd.concat([benign_sample, attack_sample], ignore_index=True)
    print(f"  Sampled — Benign: {len(benign_sample)}, Attacks: {len(attack_sample)}")

    return combined


def load_all_datasets(csv_files):
    all_dfs = []

    for i, file in enumerate(csv_files):
        print(f"\n{'='*60}")
        print(f"Processing file {i+1}/{len(csv_files)}")
        df = load_balanced_sample(file, max_benign=5000)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        return None

    combined_df = pd.concat(all_dfs, ignore_index=True)

    print(f"\n{'='*60}")
    print("COMBINED DATASET SUMMARY")
    print(f"Total rows: {len(combined_df)}")
    print(f"\nLabel Distribution:")
    print(combined_df['Label'].value_counts())
    print(f"\nLabel Percentages:")
    print(combined_df['Label'].value_counts(normalize=True).mul(100).round(2))

    return combined_df


def main():
    path = download_dataset()
    csv_files = list_csv_files(path)
    combined_df = load_all_datasets(csv_files)

    if combined_df is not None:
        os.makedirs("data", exist_ok=True)
        output_path = "data/combined_dataset.csv"
        combined_df.to_csv(output_path, index=False)
        print(f"\n✅ Combined dataset saved to: {output_path}")

        label_summary = combined_df['Label'].value_counts().to_frame(name='count')
        label_summary['percentage'] = (label_summary['count'] / len(combined_df) * 100).round(4)
        label_summary.to_csv("data/label_distribution.csv")
        print(f"✅ Label distribution saved to: data/label_distribution.csv")


if __name__ == "__main__":
    main()
