# Enterprise Telecom Churn Decision Intelligence Suite

An end-to-end Machine Learning operations (MLOps) pipeline designed to identify retail telecom subscribers at a high risk of canceling their services. This application integrates an advanced automated data preprocessing matrix with an optimized XGBoost predictive engine, wrapped inside a live Streamlit operational intelligence dashboard.

## 🏢 Business Impact & Executive Summary
In the telecommunications sector, acquiring new subscribers is up to 5x more expensive than retaining existing ones. This suite provides actionable decision intelligence by:
* **Quantifying Financial Risk:** Dynamically computing Monthly Recurring Revenue (MRR) at stake for every high-risk profile.
* **Proactive Retention Targeting:** Segmenting customer accounts based on statistical churn probability to optimize targeted marketing campaigns.

## 🏗️ Architecture & Technical Design

The project is structured into three modular, production-ready pipeline components:
1. **Data Preprocessing Automation Layer:** A unified pipeline utilizing standard scaling for continuous numerical values (`Tenure`, `MonthlyBilling`) and ordinal encoding for structural components (`Contract Type`).
2. **Predictive Analytics Core:** Powered by high-performance gradient boosting via an optimized XGBoost algorithm.
3. **Operational Intelligence Layer:** An asynchronous Streamlit application serving real-time predictions and financial revenue updates.

### Project Structure
```text
├── data/
│   └── telecom_sample.csv                  # Mock data for pipeline verification
├── app.py                                  # Live web interface and scoring script
├── train.py                                # Production training & serialization pipeline
├── telecom_churn_production_pipeline.pkl   # Serialized unified MLOps pipeline binary
└── requirements.txt                        # Enterprise environment dependencies
```

## 🚀 Execution & Quickstart

### Environment Prerequisites
To provision the deployment platform parameters locally, execute the environmental dependencies configuration:
```bash
pip install -r requirements.txt
```

### Model Re-Training & Pipeline Compilation
To re-compile the end-to-end preprocessing sequence and update the classifier logic binary, run the pipeline training script:
```bash
python train.py
```

### Running the Live Scoring Interface
To initialize the live user profile dashboard interface, launch the Streamlit server locally:
```bash
streamlit run app.py
```

## 📊 Core Performance Metrics
The system evaluation layer quantifies efficiency utilizing industry-standard validation splits:
* **Stratified Data Separation:** Ensures identical distributions of churn attributes across training and validation sets to neutralize class imbalance bias.
* **Probability Inference Calibration:** Computes raw continuous probabilities rather than basic binary flags, providing high granularity for customer success workflows.
