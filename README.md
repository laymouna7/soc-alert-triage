# SOC Alert Triage System
### Explainable AI-Based Network Intrusion Detection & Prioritization

---

## Overview

Security Operations Centers (SOCs) are flooded with thousands of network alerts daily. Analysts waste hours triaging false positives while real threats slip through. Alert fatigue is one of the leading causes of security breaches that should have been caught.

This project builds an intelligent, explainable triage system that automatically classifies network flows into priority levels — **LOW**, **MEDIUM**, or **HIGH** — and provides a human-readable justification for every decision. The analyst stays in control. The system tells them where to look first and why.

Core design principles:

- **Human-in-the-loop** — the system augments analysts, never replaces them
- **Explainability first** — every alert comes with feature-level reasoning via SHAP
- **Reproducible** — data, models, and pipelines are versioned and documented
- **Production-aware** — design decisions account for real deployment constraints

---

## Architecture

```
[1] Data Layer
    Raw network captures (PCAP) → CICFlowMeter → CSV flow records
    Download and balanced sampling via kagglehub

[2] Preprocessing & Feature Engineering
    79 raw features → 38 selected SOC-relevant features
    Encoding normalization, inf/NaN handling, severity label mapping

[3] Triage Engine
    Random Forest (100 trees, class_weight='balanced')
    3-class output: LOW / MEDIUM / HIGH
    Trained on 478,116 flows, evaluated on 119,530

[4] Explainability Layer
    SHAP TreeExplainer — exact Shapley values per flow
    Global feature importance across all classes
    Per-alert top-5 contributing features with direction and magnitude

[5] Cross-Dataset Validation
    3-experiment generalization study: CIC-IDS2017 → UNSW-NB15
    Feature coverage analysis, covariate shift quantification
    Native domain model for performance ceiling comparison

[6] Web Interface
    Flask backend — CSV upload, triage, SHAP computation
    Dark-theme SOC-style dashboard
    Expandable per-alert SHAP explanations
    PDF report export with full triage results
```

---

## Project Structure

```
soc-alert-triage/
├── data/
│   ├── combined_dataset.csv               # 597,646 flows, 15 attack types
│   ├── combined_with_priority.csv         # With severity labels
│   ├── label_distribution.csv            # Class distribution summary
│   ├── unsw_experiment1_results.csv       # Cross-validation experiment 1
│   └── unsw_cross_validation_results.csv  # Cross-validation full results
│
├── models/
│   ├── triage_model.pkl                   # Trained Random Forest (38 features)
│   ├── feature_names.pkl                  # Ordered feature list
│   ├── triage_model_shared_features.pkl   # Model trained on 15 shared features
│   ├── shared_feature_names.pkl           # 15 shared feature names
│   ├── triage_model_unsw_native.pkl       # Native UNSW-NB15 model
│   └── unsw_native_feature_names.pkl      # UNSW native feature names
│
├── src/
│   ├── data_loader.py         # Dataset download and balanced sampling
│   ├── feature_analyzer.py    # Feature analysis by category and attack type
│   ├── triage_engine.py       # Model training, evaluation, single-flow triage
│   ├── shap_explainer.py      # SHAP explainer, global and per-flow explanations
│   ├── cross_validation.py    # 3-experiment cross-dataset generalization study
│   └── webapp/
│       ├── app.py             # Flask application
│       ├── pdf_exporter.py    # PDF report generation via ReportLab
│       └── templates/
│           └── index.html     # SOC dashboard UI
│
├── notebooks/
├── docs/
├── requirements.txt
└── README.md
```

---

## Dataset

**CIC-IDS2017** — Canadian Institute for Cybersecurity Intrusion Detection Dataset 2017

Generated in a controlled lab environment with real network infrastructure. Normal traffic was captured over multiple days, then attacks were injected using real offensive tools. Traffic was processed with CICFlowMeter to produce per-flow statistical features — the same format used by production network monitoring tools like Zeek and NetFlow.

Each row represents one network flow: a complete conversation between two endpoints summarized as 79 behavioral statistics. No payload content is used — only metadata.

### Attack Types & Severity Mapping

| Attack | Category | Severity | Key Signature |
|---|---|---|---|
| BENIGN | Normal | LOW | Baseline traffic |
| PortScan | Reconnaissance | MEDIUM | Avg packet size = 5 bytes, no ACK |
| Bot | Persistence | MEDIUM | C2 beacon pattern |
| FTP-Patator | Brute Force | MEDIUM | IAT Min = 79µs, automated |
| SSH-Patator | Brute Force | MEDIUM | IAT Min = 92µs, automated |
| DoS Hulk | Denial of Service | HIGH | 182,511 packets/second |
| DoS GoldenEye | Denial of Service | HIGH | IAT Min = 11,318,405µs — deliberate slowness |
| DoS slowloris | Denial of Service | HIGH | Partial HTTP headers, starves threads |
| DoS Slowhttptest | Denial of Service | HIGH | Slow body, connection exhaustion |
| DDoS | Denial of Service | HIGH | Distributed, target stops responding |
| Web Attack - Brute Force | Web Exploitation | HIGH | Rapid application-level credential attempts |
| Web Attack - XSS | Web Exploitation | HIGH | Injection via HTTP |
| Web Attack - Sql Injection | Web Exploitation | HIGH | Database injection via HTTP |
| Infiltration | Post-Compromise | HIGH | Internal lateral movement |
| Heartbleed | Exploitation | HIGH | 1,897 backward packets — memory dump |

### Dataset Statistics

```
Total flows (balanced sample) : 597,646
Benign                        :  40,000  (6.69%)
Attacks                       : 557,646 (93.31%)

Severity distribution:
  HIGH    : 382,915  (64.1%)
  MEDIUM  : 174,731  (29.2%)
  LOW     :  40,000   (6.7%)
```

### Sampling Strategy

The original dataset contains over 2.8 million flows. A naive 10,000-row-per-file approach captured only 8 attacks in 80,000 rows — attacks appear later in the captures. The loader now reads each file in full, keeps all attack rows, and caps benign rows at 5,000 per file.

---

## Feature Engineering

79 raw features reduced to 38 selected features. Selection criteria: statistical discriminability between benign and attack classes, security domain relevance, elimination of redundant and near-zero columns.

### Feature Categories

| Category | Features | Security Relevance |
|---|---|---|
| Basic Flow | Destination Port, Flow Duration, Total Fwd/Bwd Packets, Total Length Fwd/Bwd | Volume and direction |
| Rate Metrics | Flow Bytes/s, Flow Packets/s, Fwd/Bwd Packets/s | Flood detection |
| IAT (timing) | Flow/Fwd/Bwd IAT Mean, Std, Max, Min | DoS detection — flood vs slow attack |
| TCP Flags | SYN, ACK, RST, FIN, PSH counts | Scan detection, handshake analysis |
| Packet Size | Fwd/Bwd Packet Length Mean/Std, Average Packet Size | Probe vs data traffic |
| Connection | Init_Win_bytes_forward/backward, Active Mean, Idle Mean | TCP negotiation behavior |

### Attack Signatures in Feature Space

```
DoS Hulk       Flow Packets/s = 182,511       (benign baseline: 62)
DoS GoldenEye  Flow IAT Min   = 11,318,405 µs (deliberately slow)
PortScan       Avg Packet Size = 5 bytes       (empty SYN probes)
PortScan       Total Bwd Pkts = 1              (target ignores)
DDoS           Total Bwd Pkts = 0              (target overwhelmed)
Heartbleed     Total Bwd Pkts = 1,897          (memory dump responses)
Infiltration   ACK Flag Count = 0.83           (established internal connections)
FTP-Patator    Flow IAT Min   = 79 µs          (automated brute force)
```

### Dropped Features

| Category | Reason |
|---|---|
| Bulk features (Fwd/Bwd Avg Bytes/Bulk etc.) | Near-zero across entire dataset |
| Subflow features | Redundant with total packet counts |
| Fwd Header Length.1 | Exact duplicate of Fwd Header Length |
| Packet Length Variance | Redundant — variance = std² |
| URG / CWE / ECE flags | Near-zero everywhere, no discriminative value |

---

## Severity Mapping

```
BENIGN                     → LOW     no action required
PortScan                   → MEDIUM  reconnaissance, early warning
Bot                        → MEDIUM  suspicious, may escalate
FTP-Patator                → MEDIUM  active brute force, monitor
SSH-Patator                → MEDIUM  active brute force, monitor
DoS Hulk                   → HIGH    active service disruption
DoS GoldenEye              → HIGH    active service disruption
DoS slowloris              → HIGH    active service disruption
DoS Slowhttptest           → HIGH    active service disruption
DDoS                       → HIGH    distributed, harder to mitigate
Web Attack - Brute Force   → HIGH    active exploitation
Web Attack - XSS           → HIGH    active exploitation
Web Attack - Sql Injection → HIGH    active exploitation
Infiltration               → HIGH    attacker inside the network
Heartbleed                 → HIGH    data exfiltration risk
```

MEDIUM attacks are active but detectable early without immediate damage. HIGH attacks require immediate analyst response.

---

## Model

**Random Forest Classifier** — 100 trees, `class_weight='balanced'`, `max_depth=20`, `min_samples_leaf=5`, `n_jobs=-1`

### Why Random Forest

On pre-engineered tabular flow features, Random Forest consistently matches or outperforms neural networks in published IDS literature. Additional reasons: no normalization required, native feature importance output, exact SHAP computation via TreeExplainer, class imbalance handling without oversampling artifacts.

### Class Imbalance Handling

Without correction, predicting HIGH for every flow yields 64.1% accuracy. `class_weight='balanced'` weights each class inversely proportional to its frequency. Mistakes on LOW (6.7% of data) are penalized approximately 9x more heavily than mistakes on HIGH during training.

### Results on CIC-IDS2017

Evaluated on 119,530 held-out flows — 20% stratified split, random_state=42.

```
              precision    recall  f1-score   support

         LOW       0.99      1.00      0.99      8,000
      MEDIUM       1.00      1.00      1.00     34,946
        HIGH       1.00      1.00      1.00     76,584

    accuracy                           1.00    119,530

Confusion Matrix (rows=actual, cols=predicted):

         LOW   MEDIUM    HIGH
LOW     7,977      10      13
MEDIUM     31  34,909       6
HIGH       62       0  76,522

106 misclassifications out of 119,530 flows.
```

Near-perfect accuracy reflects the statistical separability of CIC-IDS2017 attack signatures — not overfitting. DoS Hulk generates 182,000 packets/second against a benign baseline of 62. Real-world deployment would face noisier data, adversarial evasion, and concept drift. The explainability layer allows analysts to verify model reasoning on any individual prediction.

---

## Explainability (SHAP)

SHAP (SHapley Additive exPlanations) is grounded in cooperative game theory. Each feature is treated as a player in a coalition, and its contribution to a prediction is computed as its average marginal contribution across all possible feature orderings. `shap.TreeExplainer` computes exact Shapley values by traversing the Random Forest tree structure directly — not an approximation.

For each flow prediction:
- The model has a baseline prediction (average over a background sample of 500 flows)
- Each feature pushes the prediction up or down from that baseline
- The sum of all SHAP values equals the difference between the prediction and the baseline
- Positive SHAP = pushed toward the predicted class
- Negative SHAP = pushed away from the predicted class

### Example Explanations

**BENIGN → LOW (100% confidence)**
```
+  Destination Port = 443        shap=+0.1551   HTTPS — overwhelmingly benign in training data
+  Bwd Packet Length Mean = 95   shap=+0.0749   Normal server response size
+  Average Packet Size = 79      shap=+0.0689   Typical TLS handshake traffic
+  Init_Win_bytes_backward = 972 shap=+0.0407   Normal TCP window negotiation
```

**PortScan → MEDIUM (100% confidence)**
```
+  PSH Flag Count = 1            shap=+0.0795   Scanner forcing data through without buffering
+  Packet Length Mean = 3.33     shap=+0.0607   Near-empty probe packets
+  Total Length of Fwd = 2       shap=+0.0487   Minimal payload — pure probing
+  Init_Win_bytes_backward = 0   shap=+0.0363   Target never opened a receive window
```

**DDoS → HIGH (100% confidence)**
```
+  Destination Port = 80         shap=+0.0680   HTTP — common DDoS target
-  Init_Win_bytes_backward = -1  shap=-0.0567   Connection never established by target
+  Init_Win_bytes_forward = 256  shap=+0.0424   Minimal window size — DDoS tool signature
+  Bwd Packets/s = 0             shap=+0.0412   Target completely unresponsive
```

### Global SHAP Feature Importance (top 10)

```
Destination Port          0.0534
Init_Win_bytes_backward   0.0363
PSH Flag Count            0.0257
Bwd Packet Length Std     0.0250
Packet Length Mean        0.0227
Total Length of Bwd Pkts  0.0223
Total Length of Fwd Pkts  0.0217
Init_Win_bytes_forward    0.0210
Bwd Packet Length Mean    0.0206
Fwd Packet Length Mean    0.0204
```

SHAP global importance is more reliable than the Random Forest's built-in `feature_importances_`, which is biased toward high-cardinality features. SHAP importance reflects actual contribution to predictions averaged across all samples and all classes.

---

## Cross-Dataset Validation

A 3-experiment generalization study evaluating model behavior on UNSW-NB15 — a completely different dataset from an Australian university lab with different infrastructure, different attack tools, and 10 attack categories (257,673 flows).

### Feature Coverage Analysis

Only 15 of 38 CIC-IDS2017 features have direct equivalents in UNSW-NB15. The 23 missing features include the top SHAP contributors.

```
Covered  : 15/38 (39.5%)
  Flow Duration, Total Fwd/Bwd Packets, Total Length Fwd/Bwd,
  Flow Bytes/s, Flow Packets/s, Flow IAT Mean/Std,
  Bwd IAT Mean/Std, Fwd/Bwd Packet Length Mean,
  Init_Win_bytes_forward/backward

Zeroed   : 23/38
  Destination Port, all TCP flags (SYN/ACK/RST/FIN/PSH),
  Flow IAT Max/Min, Fwd IAT Mean/Std/Max/Min,
  Bwd IAT Max/Min, Fwd/Bwd Packet Length Std,
  Packet Length Mean/Std, Average Packet Size,
  Fwd/Bwd Packets/s, Active Mean, Idle Mean
```

### Experiment Results

| Experiment | Features | Trained On | Tested On | Accuracy |
|---|---|---|---|---|
| 1 — Full model, zeroed features | 38 | CIC-IDS2017 | UNSW-NB15 | 36% — collapse |
| 2 — Shared features only | 15 | CIC-IDS2017 | UNSW-NB15 | 36% — collapse |
| 3 — Native UNSW model | 28 | UNSW-NB15 | UNSW-NB15 | **90%** |

### Interpretation

Experiments 1 and 2 both collapse to predicting LOW for all flows. This is not a model failure — it is **covariate shift**: the same feature values carry different statistical distributions across environments captured on different infrastructure with different tools. The model learned what "attack" looks like in one specific lab environment. UNSW-NB15 traffic, even for the same attack categories, has different baseline statistics.

Experiment 3 confirms the pipeline generalizes correctly. 90% accuracy when trained natively on UNSW-NB15 establishes the performance ceiling and proves the architecture works across domains when trained appropriately.

### UNSW-NB15 Native Model Results

```
              precision    recall  f1-score   support

         LOW       0.96      0.88      0.92     18,600
      MEDIUM       0.67      0.77      0.72      8,182
        HIGH       0.94      0.95      0.95     24,753

    accuracy                           0.90     51,535
```

MEDIUM F1 of 0.72 reflects genuine difficulty separating Reconnaissance, Fuzzers, and Analysis categories — behaviorally adjacent attack types that are harder to distinguish even for a domain-trained model.

### Production Implication

Zero-shot cross-dataset transfer fails due to covariate shift. The correct production approach is domain-specific training. The pipeline supports this without architectural changes: re-running `triage_engine.py` on target-environment data produces a new model in minutes.

---

## Web Interface

Flask-based SOC dashboard. Dark theme, minimal, functional.

**Features:**
- CSV upload via file picker or drag-and-drop
- Summary cards: total flows, HIGH / MEDIUM / LOW counts
- Alert table sorted by severity — HIGH first
- Confidence breakdown per alert (H% / M% / L%)
- Click any row to expand SHAP explanation with directional bar chart
- Export full triage report as a timestamped PDF

**Run:**
```bash
python3 src/webapp/app.py
```

Open `http://localhost:5000` and upload any CICFlowMeter-format CSV. Capped at 5,000 flows per upload.

### PDF Report

The exported PDF includes an executive summary with severity counts, a breakdown table with action guidance per severity level, and per-alert SHAP explanations for the first 100 alerts. Each alert shows the top 5 features that drove the decision, their values, SHAP magnitudes, and direction of influence.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/laymouna7/soc-alert-triage.git
cd soc-alert-triage

# Install dependencies
pip install pandas "numpy<2" scikit-learn "shap==0.45.1" joblib flask kagglehub reportlab

# Download dataset and build combined CSV
python3 src/data_loader.py

# Analyze features
python3 src/feature_analyzer.py

# Train the triage model
python3 src/triage_engine.py

# Run SHAP explainer
python3 src/shap_explainer.py

# Run cross-dataset validation study
python3 src/cross_validation.py

# Launch web interface
python3 src/webapp/app.py
```

### Requirements

```
Python 3.10+
pandas
numpy < 2.0
scikit-learn
shap == 0.45.1
joblib
flask
kagglehub
reportlab
```

---

## Limitations & Honest Assessment

**Dataset is controlled.** CIC-IDS2017 was generated in a lab with clean labels and no adversarial evasion. Real-world traffic is noisier, labels are uncertain, and attackers actively try to blend in with legitimate traffic.

**Environment-specific baseline.** The model learned what normal traffic looks like on one university lab network. Cross-dataset validation confirmed this — the model collapses on UNSW-NB15 due to covariate shift in feature distributions.

**Static model.** The model does not update in production. Concept drift — where attack patterns evolve over time — is not addressed. Periodic retraining on fresh labeled data is required.

**No adversarial robustness.** An attacker aware of the model's features could craft flows that evade detection by mimicking benign statistical patterns — for example, throttling a DoS attack to stay below the IAT threshold.

**Retraining is straightforward.** The pipeline is designed for it. Re-running `triage_engine.py` with new data produces an updated model without architectural changes.

---

## Roadmap

- [x] Data pipeline with balanced sampling
- [x] Feature engineering — 38 SOC-relevant features
- [x] Random Forest triage engine with class balancing
- [x] SHAP explainability layer — per-flow and global
- [x] Flask web interface with dark SOC theme
- [x] Cross-dataset validation study — UNSW-NB15, 3 experiments
- [x] PDF report export
- [ ] IoT domain fine-tuning (N-BaIoT dataset)
- [ ] Domain model selection in UI (enterprise vs IoT)
- [ ] DVC integration for dataset and model versioning
- [ ] Streaming triage mode — real-time flow processing

---

## References

- Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization.* ICISSP.
- Moustafa, N., & Slay, J. (2015). *UNSW-NB15: A Comprehensive Data Set for Network Intrusion Detection Systems.* MilCIS.
- Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS.
- Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5–32.
- CICFlowMeter — Network Traffic Flow Generator: https://www.unb.ca/cic/research/applications.html

---

## Author

Built as part of a network security / applied ML project.
Pipeline: data acquisition → feature engineering → ML triage → SHAP explainability → cross-dataset validation → SOC web interface with PDF export.
