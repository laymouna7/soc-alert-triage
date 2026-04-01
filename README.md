# SOC Alert Triage System
### Explainable AI-Based Network Intrusion Detection & Prioritization

---

## Overview

Security Operations Centers (SOCs) are flooded with thousands of network alerts daily. Analysts waste hours triaging false positives while real threats slip through. Alert fatigue is one of the leading causes of security breaches that should have been caught.

This project builds an intelligent, explainable triage system that automatically classifies network flows into priority levels — **LOW**, **MEDIUM**, or **HIGH** — and provides a human-readable justification for every decision. The analyst stays in control. The system tells them where to look first and why.

The core design principles:

- **Human-in-the-loop** — the system augments analysts, never replaces them
- **Explainability first** — every alert comes with feature-level reasoning via SHAP
- **Reproducible** — data, models, and pipelines are versioned and documented
- **Production-aware** — design decisions account for real deployment constraints

---

## Demo

Upload any CICFlowMeter-format CSV to the web interface and get back:

- A summary dashboard: total flows, HIGH / MEDIUM / LOW counts
- A sortable alert table with severity badges and confidence scores
- Per-alert SHAP explanations: which features drove the decision and by how much
- Click any row to expand the full feature contribution breakdown

---

## Architecture
```
[1] Data Layer
    Raw network captures (PCAP) → CICFlowMeter → CSV flow records
    Download and sampling via kagglehub

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

[5] Web Interface
    Flask backend — CSV upload, triage, SHAP computation
    Dark-theme SOC-style UI
    Expandable per-alert explanations
```

---

## Project Structure
```
soc-alert-triage/
├── data/
│   ├── combined_dataset.csv          # 597,646 flows, 15 attack types
│   ├── combined_with_priority.csv    # With severity labels
│   └── label_distribution.csv       # Class distribution summary
│
├── models/
│   ├── triage_model.pkl              # Trained Random Forest
│   └── feature_names.pkl            # Ordered feature list (38 features)
│
├── src/
│   ├── data_loader.py                # Dataset download and balanced sampling
│   ├── feature_analyzer.py          # Feature analysis by category and attack type
│   ├── triage_engine.py             # Model training, evaluation, single-flow triage
│   ├── shap_explainer.py            # SHAP explainer, global and per-flow explanations
│   └── webapp/
│       ├── app.py                    # Flask application
│       └── templates/
│           └── index.html            # SOC dashboard UI
│
├── notebooks/                        # Exploration and analysis
├── docs/
│   ├── vision.md
│   └── soc_context.md
├── requirements.txt
└── README.md
```

---

## Dataset

**CIC-IDS2017** — Canadian Institute for Cybersecurity Intrusion Detection Dataset 2017

The dataset was generated in a controlled lab environment with real network infrastructure. Normal traffic was captured over multiple days, then attacks were injected using real offensive tools. Traffic was processed with CICFlowMeter to produce per-flow statistical features — the same format used by production network monitoring tools like Zeek and NetFlow.

Each row represents one network flow: a complete conversation between two endpoints, summarized as 79 behavioral statistics. No payload content is used — only metadata.

### Attack Types

| Attack | Category | Severity | Description |
|---|---|---|---|
| BENIGN | Normal | LOW | Baseline network traffic |
| PortScan | Reconnaissance | MEDIUM | Port probing via SYN packets to discover open services |
| Bot | Persistence | MEDIUM | Compromised host communicating with C2 server |
| FTP-Patator | Brute Force | MEDIUM | Automated FTP credential attacks |
| SSH-Patator | Brute Force | MEDIUM | Automated SSH credential attacks |
| DoS Hulk | Denial of Service | HIGH | HTTP flood — 182,000+ packets/second |
| DoS GoldenEye | Denial of Service | HIGH | Slow HTTP — holds connections open to exhaust pool |
| DoS slowloris | Denial of Service | HIGH | Partial HTTP headers — starves server threads |
| DoS Slowhttptest | Denial of Service | HIGH | Slow HTTP body — similar to slowloris |
| DDoS | Denial of Service | HIGH | Distributed flood from multiple sources |
| Web Attack - Brute Force | Web Exploitation | HIGH | Application-level credential brute force |
| Web Attack - XSS | Web Exploitation | HIGH | Cross-site scripting injection |
| Web Attack - Sql Injection | Web Exploitation | HIGH | SQL injection against web applications |
| Infiltration | Post-Compromise | HIGH | Attacker operating inside the network |
| Heartbleed | Exploitation | HIGH | OpenSSL memory disclosure vulnerability |

### Dataset Statistics
```
Total flows after sampling  : 597,646
Benign                      : 40,000  (6.69%)
Attacks                     : 557,646 (93.31%)

Severity distribution:
  HIGH    : 382,915  (64.1%)
  MEDIUM  : 174,731  (29.2%)
  LOW     :  40,000   (6.7%)
```

### Why Balanced Sampling

The original dataset contains over 2.8 million flows, heavily dominated by benign traffic in some files. A naive 10,000-row-per-file approach captured almost no attacks (8 attacks in 80,000 rows). The loader now reads each file in full, keeps all attack rows, and caps benign rows at 5,000 per file — ensuring attack patterns are well-represented without making the dataset unmanageable.

---

## Feature Engineering

78 numeric features were available. 38 were selected based on:

- Statistical discriminability between benign and attack classes
- Security domain relevance (known attack signatures)
- Elimination of redundant, near-zero, and duplicate columns

### Selected Feature Categories

**Basic Flow** — `Destination Port`, `Flow Duration`, `Total Fwd/Bwd Packets`, `Total Length of Fwd/Bwd Packets`

**Rate Metrics** — `Flow Bytes/s`, `Flow Packets/s`, `Fwd Packets/s`, `Bwd Packets/s`

**Inter-Arrival Times (IAT)** — `Flow IAT Mean/Std/Max/Min`, `Fwd IAT Mean/Std/Max/Min`, `Bwd IAT Mean/Std/Max/Min`

**TCP Flags** — `SYN`, `ACK`, `RST`, `FIN`, `PSH` flag counts

**Packet Size** — `Fwd/Bwd Packet Length Mean/Std`, `Packet Length Mean/Std`, `Average Packet Size`

**Connection Behavior** — `Init_Win_bytes_forward`, `Init_Win_bytes_backward`, `Active Mean`, `Idle Mean`

### Attack Signatures in Feature Space

| Attack | Key Signal | Values |
|---|---|---|
| DoS Hulk | Flow Packets/s | 182,511 vs baseline 62 |
| DoS GoldenEye | Flow IAT Min | 11,318,405 µs — deliberately slow |
| PortScan | Average Packet Size | 5 bytes — empty probes |
| PortScan | Total Backward Packets | 1 — target never responds |
| DDoS | Total Backward Packets | 0 — target overwhelmed |
| FTP-Patator | Flow IAT Min | 79 µs — automated brute force |
| Heartbleed | Total Backward Packets | 1,897 — massive server memory dump |
| Infiltration | ACK Flag Count | 0.83 — established internal connections |

### Dropped Features

| Category | Reason |
|---|---|
| Bulk features (Fwd/Bwd Avg Bytes/Bulk etc.) | Near-zero across entire dataset |
| Subflow features | Redundant with total packet counts |
| `Fwd Header Length.1` | Exact duplicate of `Fwd Header Length` |
| `Packet Length Variance` | Redundant — variance = std² |
| URG / CWE / ECE flags | Near-zero everywhere, no discriminative value |

---

## Severity Mapping

Attack labels were mapped to three SOC-relevant severity levels based on the nature of the threat and the urgency of analyst response:
```
BENIGN                    → LOW    (no action required)
PortScan                  → MEDIUM (reconnaissance, early warning)
Bot                       → MEDIUM (suspicious, may escalate)
FTP-Patator               → MEDIUM (active brute force, monitor)
SSH-Patator               → MEDIUM (active brute force, monitor)
DoS Hulk                  → HIGH   (active service disruption)
DoS GoldenEye             → HIGH   (active service disruption)
DoS slowloris             → HIGH   (active service disruption)
DoS Slowhttptest          → HIGH   (active service disruption)
DDoS                      → HIGH   (distributed, harder to mitigate)
Web Attack - Brute Force  → HIGH   (active exploitation)
Web Attack - XSS          → HIGH   (active exploitation)
Web Attack - Sql Injection→ HIGH   (active exploitation)
Infiltration              → HIGH   (attacker inside network)
Heartbleed                → HIGH   (data exfiltration risk)
```

The distinction between MEDIUM and HIGH reflects operational priority: MEDIUM attacks are active but detectable early and not yet causing damage. HIGH attacks require immediate response.

---

## Model

### Architecture

**Random Forest Classifier**
- 100 estimators (decision trees)
- `class_weight='balanced'` — penalizes mistakes on minority classes proportionally
- `max_depth=20` — prevents overfitting
- `min_samples_leaf=5` — ensures generalization at leaf level
- `n_jobs=-1` — parallel training across all CPU cores

### Why Random Forest

- Tabular data: RF consistently outperforms neural networks on pre-engineered flow features
- No normalization required: RF handles mixed feature scales natively
- Feature importance: available directly from the model, validated by SHAP
- Interpretability: individual trees can be inspected; SHAP values are exact
- Class imbalance: `class_weight='balanced'` handles minority classes without oversampling artifacts

### Handling Class Imbalance

Without correction, a model predicting HIGH for every flow would achieve 64.1% accuracy. `class_weight='balanced'` computes per-class weights as:
```
weight(class) = total_samples / (n_classes × samples_in_class)
```

Mistakes on LOW (6.7% of data) are penalized ~9× more than mistakes on HIGH during training.

### Results

Evaluated on 119,530 held-out flows (20% stratified split, random_state=42):
```
              precision    recall  f1-score   support

         LOW       0.99      1.00      0.99      8,000
      MEDIUM       1.00      1.00      1.00     34,946
        HIGH       1.00      1.00      1.00     76,584

    accuracy                           1.00    119,530
```

**Confusion Matrix:**
```
         LOW   MEDIUM   HIGH
LOW     7,977      10     13
MEDIUM     31  34,909      6
HIGH       62       0  76,522
```

106 misclassifications out of 119,530 flows.

### On the Accuracy Score

Near-perfect accuracy on CIC-IDS2017 is a known property of the dataset, not evidence of overfitting. Attack signatures in this dataset are statistically well-separated — DoS Hulk generates 182,000 packets/second against a benign baseline of 62. Any well-implemented classifier will approach these numbers.

Real-world deployment would face noisier data, adversarial evasion, and concept drift. This system is designed with those constraints in mind: the explainability layer allows analysts to verify model reasoning, and the pipeline supports retraining on new data without architectural changes.

---

## Explainability

### SHAP (SHapley Additive exPlanations)

SHAP values are grounded in cooperative game theory. Each feature is treated as a player in a coalition game, and its contribution to the prediction is computed as the average marginal contribution across all possible feature coalitions.

For a single flow prediction:
- The model has a baseline prediction (average over background sample)
- Each feature pushes the prediction up or down from that baseline
- The sum of all SHAP values equals the difference between the prediction and baseline
- Positive SHAP = pushed toward the predicted class
- Negative SHAP = pushed away from the predicted class

### Implementation

`shap.TreeExplainer` is used, which computes exact SHAP values by traversing the Random Forest tree structure directly — not an approximation. A background sample of 500 flows is used as the reference distribution.

### Example Explanations

**BENIGN → LOW (100% confidence)**
```
+ Destination Port = 443       shap=+0.1551  HTTPS — overwhelmingly benign
+ Bwd Packet Length Mean = 95  shap=+0.0749  Normal server response size
+ Average Packet Size = 79     shap=+0.0689  Typical TLS traffic
+ Init_Win_bytes_backward = 972 shap=+0.0407  Normal TCP window negotiation
```

**PortScan → MEDIUM (100% confidence)**
```
+ PSH Flag Count = 1           shap=+0.0795  Scanner forcing data through
+ Packet Length Mean = 3.33    shap=+0.0607  Near-empty probe packets
+ Total Length of Fwd = 2      shap=+0.0487  Minimal payload
+ Init_Win_bytes_backward = 0  shap=+0.0363  Target never opened receive window
```

**DDoS → HIGH (100% confidence)**
```
+ Destination Port = 80        shap=+0.0680  HTTP — common DDoS target
- Init_Win_bytes_backward = -1  shap=-0.0567  Connection never established
+ Init_Win_bytes_forward = 256 shap=+0.0424  Minimal window — tool signature
+ Bwd Packets/s = 0            shap=+0.0412  Target completely unresponsive
```

### Global Feature Importance (SHAP vs RF Built-in)

SHAP global importance is computed as mean absolute SHAP values across all classes and samples. This is more reliable than the Random Forest's built-in `feature_importances_`, which is biased toward high-cardinality features.

Top features by SHAP importance:
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

---

## Web Interface

Flask-based SOC dashboard. Dark theme, minimal, functional.

**Features:**
- CSV upload via file picker or drag-and-drop
- Summary cards: total flows, HIGH / MEDIUM / LOW counts
- Alert table sorted by severity (HIGH first)
- Confidence breakdown per alert (H% / M% / L%)
- Click any row to expand SHAP explanation
- Visual SHAP bar chart per feature, direction-colored

**Run:**
```bash
python3 src/webapp/app.py
```
Open `http://localhost:5000`

Upload any CICFlowMeter-format CSV. The system caps at 5,000 flows per upload for response time.

---

## Installation

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
```

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/soc-alert-triage.git
cd soc-alert-triage

# Install dependencies
pip install pandas "numpy<2" scikit-learn "shap==0.45.1" joblib flask kagglehub

# Download dataset and build combined CSV
python3 src/data_loader.py

# Analyze features
python3 src/feature_analyzer.py

# Train the triage model
python3 src/triage_engine.py

# Run SHAP explainer (optional — validates explanations)
python3 src/shap_explainer.py

# Launch web interface
python3 src/webapp/app.py
```

---

## Limitations & Honest Assessment

**Dataset is controlled.** CIC-IDS2017 was generated in a lab with clean labels and no adversarial evasion. Real-world network traffic is noisier, labels are uncertain, and attackers actively try to blend in with legitimate traffic.

**Environment-specific baseline.** The model learned what "normal" looks like on one university lab network. Deployed on a hospital, bank, or cloud environment, the benign traffic distribution would differ. False positive rates would increase without retraining.

**Static model.** The model does not update in production. Concept drift — where attack patterns evolve over time — is not addressed. Periodic retraining on fresh labeled data would be required.

**No adversarial robustness.** An attacker aware of the model's features could craft flows that evade detection — for example, a DoS attack that throttles packet rate to stay below the IAT threshold.

**Retraining is straightforward.** The pipeline is designed for it. Given labeled samples from a new environment, re-running `triage_engine.py` with updated data produces a new model without architectural changes.

---

## Roadmap

- [ ] Cross-dataset validation on CIC-IDS2018 and UNSW-NB15
- [ ] Domain-specific fine-tuning on IoT traffic (N-BaIoT dataset)
- [ ] PDF report export from web interface
- [ ] Domain model selection in UI (enterprise vs IoT)
- [ ] DVC integration for dataset and model versioning
- [ ] Streaming triage mode (process flows in real time via file watch)

---

## References

- Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization.* ICISSP.
- Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS.
- Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5–32.
- CICFlowMeter — Network Traffic Flow Generator: https://www.unb.ca/cic/research/applications.html

---

## Author

Built as part of a network security / applied ML project.  
Pipeline: data acquisition → feature engineering → ML triage → SHAP explainability → SOC web interface.
